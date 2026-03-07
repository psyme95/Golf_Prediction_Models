"""
Seasonal Round 2 Model Training (Python)
Mirrors: Weekly_Modelling/Script/Seasonal_Rd2_Model_Training.R

Trains a single ensemble for the Win market using Round 2 in-tournament features.
Same modelling approach as seasonal_model_training.py:
  - Optuna tuning per model (warm-starts from saved params)
  - RepeatedStratifiedKFold OOF predictions
  - LogisticRegression meta-model using OOF scores + Betfair Rd2 odds

Input files (from shared R input directory):
  Full_{TOUR}_Historical_Predictions.xlsx  — historical model predictions + outcomes
  {TOUR}.xlsx                              — raw data with Rd2Pos, Rd2Lead, AvPosn, _Top5, Betfair_rd2

Output:
  Weekly_Modelling_Python/Output/Models/{TOUR}_R2_{SEASON}_trained.pkl

Run: python seasonal_rd2_model_training.py
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:  # IPython / Jupyter — assume CWD is the Script dir
    sys.path.insert(0, str(Path.cwd()))
from config import (
    MODELS_DIR,
    N_CV_REPEATS,
    N_CV_SPLITS,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    RD2_MODEL_VARS,
    SEASON_SUFFIX,
    TOUR_CONFIG,
    TRAINING_YEARS,
)
# Re-use tuning and OOF helpers from the main training module
from seasonal_model_training import (
    build_final_models,
    fit_meta_model,
    generate_oof,
    tss_optimal,
    tune_lgbm,
    tune_lgbm_dart,
    tune_logistic,
    tune_rf,
    tune_xgb,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Rd2 Optuna uses fewer trials — 6 features, less tuning needed
RD2_OPTUNA_TRIALS = max(30, OPTUNA_TRIALS // 2)


# ===== DATA PREPARATION =====

def load_rd2_training_data(tour_key: str, tour_info: dict) -> pd.DataFrame:
    """
    Joins historical predictions with raw tour data to obtain Rd2 features.
    Mirrors the join logic in Seasonal_Rd2_Model_Training.R.
    """
    preds_path = tour_info["rd2_predictions_file"]
    raw_path   = tour_info["rd2_raw_file"]

    if not preds_path.exists():
        raise FileNotFoundError(f"Historical predictions not found: {preds_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw tour file not found: {raw_path}")

    preds = pd.read_excel(preds_path)
    raw   = pd.read_excel(raw_path)

    # Rename columns starting with '_' (R convention)
    raw.columns   = [f"X{c}" if c.startswith("_") else c for c in raw.columns]
    preds.columns = [f"X{c}" if c.startswith("_") else c for c in preds.columns]

    rd2_cols = ["eventID", "playerID", "Rd2Pos", "Rd2Lead", "AvPosn", "XTop5", "Betfair_rd2"]
    available_rd2 = [c for c in rd2_cols if c in raw.columns]
    raw_clean = raw[available_rd2].dropna()

    merged = preds.merge(
        raw_clean,
        left_on=["EventID", "PlayerID"],
        right_on=["eventID", "playerID"],
        how="inner",
    ).dropna()

    # Rename XTop5 → Top5 (mirrors R: rename(Top5 = `_Top5`))
    if "XTop5" in merged.columns:
        merged = merged.rename(columns={"XTop5": "Top5"})

    print(f"  Joined data: {len(merged):,} rows")
    return merged


def get_rd2_training_years(df: pd.DataFrame) -> list:
    """
    Auto-detect training seasons from Test_Year column.
    Mirrors TRAIN_SEASONS logic: last TRAINING_YEARS complete seasons.
    """
    if "Test_Year" not in df.columns:
        # Fallback: use Date column
        df = df.copy()
        if "Date" in df.columns:
            df["Test_Year"] = pd.to_datetime(df["Date"]).dt.year
        else:
            raise ValueError("No Test_Year or Date column found in Rd2 data")

    max_year = int(df["Test_Year"].max())
    seasons = list(range(max_year - TRAINING_YEARS + 1, max_year + 1))
    print(f"  Rd2 training seasons: {seasons}")
    return seasons


# ===== TRAINING =====

def train_rd2(tour_key: str, tour_info: dict):
    print(f"\n{'='*60}")
    print(f"TRAINING RD2: {tour_info['name']}")
    print(f"{'='*60}")

    df = load_rd2_training_data(tour_key, tour_info)
    seasons = get_rd2_training_years(df)

    if "Test_Year" not in df.columns and "Date" in df.columns:
        df["Test_Year"] = pd.to_datetime(df["Date"]).dt.year

    train_df = df[df["Test_Year"].isin(seasons)].copy()

    # Rename Normalised_Probability → GLM_Odds_Probability_Median
    # and Model_Score → Model_Score_Median (mirrors R rename in prediction script)
    rename_map = {}
    if "Normalised_Probability" in train_df.columns:
        rename_map["Normalised_Probability"] = "GLM_Odds_Probability_Median"
    if "Model_Score" in train_df.columns:
        rename_map["Model_Score"] = "Model_Score_Median"
    train_df = train_df.rename(columns=rename_map)

    available_vars = [v for v in RD2_MODEL_VARS if v in train_df.columns]
    missing = [v for v in RD2_MODEL_VARS if v not in train_df.columns]
    if missing:
        print(f"  Warning: missing Rd2 vars: {missing}")

    target_col = "Market_Win"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available: {list(train_df.columns)}")

    df_clean = train_df[available_vars + [target_col, "Betfair_rd2"]].dropna()
    X     = df_clean[available_vars].values.astype(float)
    y     = df_clean[target_col].values.astype(int)
    odds  = df_clean["Betfair_rd2"].values.astype(float)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw   = n_neg / max(n_pos, 1)
    print(f"  Samples: {len(y):,}  |  Positives: {n_pos} ({100*n_pos/len(y):.1f}%)")

    def load_warm(model_name):
        p = MODELS_DIR / f"{tour_key}_R2_{model_name}_best_params.pkl"
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass
        return None

    print(f"  Tuning {RD2_OPTUNA_TRIALS} Optuna trials per model...")

    logistic_params = tune_logistic( X, y, RD2_OPTUNA_TRIALS, warm_params=load_warm("logistic"))
    rf_params       = tune_rf(       X, y, RD2_OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("rf"))
    lgbm_params     = tune_lgbm(     X, y, RD2_OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("lgbm"))
    xgb_params      = tune_xgb(      X, y, RD2_OPTUNA_TRIALS, scale_pos_weight=spw,
                                     warm_params=load_warm("xgb"))
    dart_params     = tune_lgbm_dart(X, y, RD2_OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("lgbm_dart"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, params in [("logistic", logistic_params), ("rf", rf_params),
                          ("lgbm", lgbm_params), ("xgb", xgb_params),
                          ("lgbm_dart", dart_params)]:
        joblib.dump(params, MODELS_DIR / f"{tour_key}_R2_{name}_best_params.pkl")

    mf_map = {"sqrt": "sqrt", "log2": "log2", "frac03": 0.3, "frac05": 0.5}
    rf_p = {k: v for k, v in rf_params.items() if k != "max_features"}
    rf_p["max_features"] = mf_map.get(rf_params.get("max_features", "sqrt"), "sqrt")

    model_configs = {
        "logistic": (
            LogisticRegression,
            dict(**logistic_params, penalty="elasticnet", solver="saga",
                 class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
        ),
        "rf": (
            RandomForestClassifier,
            dict(**rf_p, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        ),
        "lgbm": (
            lgb.LGBMClassifier,
            dict(**lgbm_params, class_weight="balanced",
                 random_state=RANDOM_SEED, verbose=-1),
        ),
        "xgb": (
            xgb.XGBClassifier,
            dict(**xgb_params, scale_pos_weight=spw, eval_metric="logloss",
                 use_label_encoder=False, random_state=RANDOM_SEED, verbosity=0),
        ),
        "lgbm_dart": (
            lgb.LGBMClassifier,
            dict(**dart_params, boosting_type="dart", class_weight="balanced",
                 random_state=RANDOM_SEED, verbose=-1),
        ),
    }

    print(f"  Generating OOF ({N_CV_SPLITS}-fold × {N_CV_REPEATS} repeats)...")
    oof_matrix = generate_oof(model_configs, model_configs, X, y)

    model_names = list(model_configs.keys())
    metrics = {}
    for j, name in enumerate(model_names):
        oof_j = np.clip(oof_matrix[:, j], 1e-15, 1 - 1e-15)
        ll  = log_loss(y, oof_j)
        auc = roc_auc_score(y, oof_j)
        tss = tss_optimal(y, oof_j)
        metrics[name] = {"log_loss": ll, "roc_auc": auc, "tss": tss}
        print(f"    {name:12s}  log_loss={ll:.4f}  AUC={auc:.4f}  TSS={tss:.4f}")

    meta_model, meta_scaler = fit_meta_model(oof_matrix, y)

    print("  Fitting final models on full training data...")
    final_models = build_final_models(model_configs, X, y)

    package = {
        "models": final_models,
        "model_names": model_names,
        "meta_model": meta_model,
        "meta_scaler": meta_scaler,
        "model_vars": available_vars,
        "metrics": metrics,
        "tour_name": tour_key,
        "train_seasons": seasons,
        "trained_at": datetime.now(),
        "n_samples": len(y),
        "n_positives": n_pos,
    }

    out_path = MODELS_DIR / f"{tour_key}_R2_{SEASON_SUFFIX}_trained.pkl"
    joblib.dump(package, out_path)
    print(f"\n  Rd2 model saved: {out_path.name}")

    return package


# ===== MAIN =====

def main():
    start = datetime.now()
    print("=== RD2 MODEL TRAINING (PYTHON) ===")
    print(f"Season: {SEASON_SUFFIX}  |  Optuna trials: {RD2_OPTUNA_TRIALS}")

    results = {}
    for tour_key, tour_info in TOUR_CONFIG.items():
        try:
            results[tour_key] = train_rd2(tour_key, tour_info)
        except Exception as e:
            print(f"\nError training {tour_key} Rd2: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n=== RD2 TRAINING COMPLETE ({elapsed:.1f} min) ===")
    for tour_key, res in results.items():
        if res:
            print(f"  {tour_key}: {res['n_samples']:,} samples  |  "
                  f"seasons {res['train_seasons']}")
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
