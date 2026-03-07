"""
Seasonal Model Training (Python)
Mirrors: Weekly_Modelling/Script/Seasonal_Model_Training.R

Key differences from R version:
  - 5 sklearn/lgbm/xgb models replace GAM + RF + ANN + GBM + XGBoost
      * LogisticRegression (ElasticNet)  replaces GAM
      * RandomForestClassifier           unchanged
      * LGBMClassifier                   replaces old GBM
      * XGBClassifier                    unchanged
      * LGBMClassifier (DART boosting)   replaces ANN (sklearn MLP lacks class_weight)
  - Optuna tunes hyperparameters once per season (replaces 30 arbitrary runs)
  - class_weight / scale_pos_weight handles imbalance instead of TSS threshold tricks
  - RepeatedStratifiedKFold OOF predictions feed a LogisticRegression meta-model
    that performs ensemble weighting + probability calibration in one step
  - Average Precision (PR-AUC) is the Optuna objective — rewards precision at the
    top of the ranked list, directly aligned with back/lay betting on a few players
  - Training summary Excel matches R output structure for direct comparison

Run: python seasonal_model_training.py
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:  # IPython / Jupyter — assume CWD is the Script dir
    sys.path.insert(0, str(Path.cwd()))
from config import (
    BASE_MODEL_VARS,
    BETTING_MARKETS,
    MODELS_DIR,
    N_CV_REPEATS,
    N_CV_SPLITS,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    SEASON_SUFFIX,
    TOUR_CONFIG,
    TRAINING_YEARS,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===== METRICS =====

def tss_optimal(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """TSS at the optimal decision threshold (mirrors biomod2 evaluation)."""
    thresholds = np.unique(y_prob)
    best = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        if (tp + fn) > 0 and (tn + fp) > 0:
            tss = tp / (tp + fn) + tn / (tn + fp) - 1
            if tss > best:
                best = tss
    return best


# ===== OPTUNA TUNING =====

def _cv_log_loss(model, X: np.ndarray, y: np.ndarray, seed: int) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss", n_jobs=-1)
    return -float(scores.mean())


def _cv_avg_precision(model, X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """
    Average Precision (PR-AUC) via cross-validation, returned negated so
    direction="minimize" in Optuna.
    AP rewards placing actual positives at the very top of the ranked list —
    directly aligned with back/lay betting where only a few players are acted on.
    Baseline for a random model = prevalence (e.g. ~0.008 for Winner), so
    absolute AP values will look small; a Winner AP of 0.05 is ~6× better than random.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    return -float(scores.mean())   # negate → minimise negative AP


def tune_logistic(X, y, n_trials, warm_params=None, seed=RANDOM_SEED):
    def objective(trial):
        C        = trial.suggest_float("C", 1e-3, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
            class_weight="balanced", max_iter=2000, random_state=seed,
        )
        return _cv_avg_precision(model, X, y, seed)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        study.enqueue_trial(warm_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_rf(X, y, n_trials, n_pos=None, warm_params=None, seed=RANDOM_SEED):
    _mf_map = {"sqrt": "sqrt", "log2": "log2", "frac03": 0.3, "frac05": 0.5}
    # Cap upper bound at 2×positives so trees can actually isolate them.
    # E.g. Winner (31 pos) → max leaf 62; Top20 (692 pos) → capped at 200.
    _n_pos = int(y.sum()) if n_pos is None else n_pos
    _leaf_max = min(200, max(5, _n_pos * 2))

    def objective(trial):
        mf_key = trial.suggest_categorical("max_features", list(_mf_map.keys()))
        model = RandomForestClassifier(
            n_estimators        = trial.suggest_int("n_estimators", 300, 1500),
            max_features        = _mf_map[mf_key],
            min_samples_leaf    = trial.suggest_int("min_samples_leaf", 1, _leaf_max),
            max_depth           = trial.suggest_categorical("max_depth",
                                      [None, 5, 10, 15, 20]),
            class_weight        = "balanced",
            n_jobs              = -1,
            random_state        = seed,
        )
        return _cv_avg_precision(model, X, y, seed)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        study.enqueue_trial(warm_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgbm(X, y, n_trials, n_pos=None, warm_params=None, seed=RANDOM_SEED):
    _n_pos = int(y.sum()) if n_pos is None else n_pos
    _child_max = min(200, max(5, _n_pos * 2))

    def objective(trial):
        model = lgb.LGBMClassifier(
            n_estimators        = trial.suggest_int("n_estimators", 100, 1000),
            num_leaves          = trial.suggest_int("num_leaves", 20, 150),
            learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples   = trial.suggest_int("min_child_samples", 1, _child_max),
            subsample           = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha           = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda          = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            class_weight        = "balanced",
            random_state        = seed,
            verbose             = -1,
        )
        return _cv_avg_precision(model, X, y, seed)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        study.enqueue_trial(warm_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgb(X, y, n_trials, scale_pos_weight, warm_params=None, seed=RANDOM_SEED):
    def objective(trial):
        model = xgb.XGBClassifier(
            n_estimators        = trial.suggest_int("n_estimators", 50, 500),
            max_depth           = trial.suggest_int("max_depth", 3, 8),
            learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample           = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight    = trial.suggest_int("min_child_weight", 5, 50),
            reg_alpha           = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda          = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            scale_pos_weight    = scale_pos_weight,
            eval_metric         = "logloss",
            use_label_encoder   = False,
            random_state        = seed,
            verbosity           = 0,
        )
        return _cv_avg_precision(model, X, y, seed)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        study.enqueue_trial(warm_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_lgbm_dart(X, y, n_trials, n_pos=None, warm_params=None, seed=RANDOM_SEED):
    _n_pos = int(y.sum()) if n_pos is None else n_pos
    _child_max = min(200, max(5, _n_pos * 2))

    def objective(trial):
        model = lgb.LGBMClassifier(
            boosting_type       = "dart",
            n_estimators        = trial.suggest_int("n_estimators", 100, 500),
            num_leaves          = trial.suggest_int("num_leaves", 20, 100),
            learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples   = trial.suggest_int("min_child_samples", 1, _child_max),
            subsample           = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree    = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            drop_rate           = trial.suggest_float("drop_rate", 0.05, 0.3),
            class_weight        = "balanced",
            random_state        = seed,
            verbose             = -1,
        )
        return _cv_avg_precision(model, X, y, seed)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        study.enqueue_trial(warm_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ===== OOF GENERATION =====

def generate_oof(models_fitted_on_subsets, model_configs, X, y, seed=RANDOM_SEED):
    """
    Generate out-of-fold predictions for ALL models using RepeatedStratifiedKFold.
    Each sample receives N_CV_REPEATS predictions; we average them.

    models_fitted_on_subsets: dict {name: (model_class, params)}
    Returns oof_matrix: np.ndarray of shape (n_samples, n_models)
    """
    n = len(y)
    n_models = len(model_configs)
    oof_sum    = np.zeros((n, n_models))
    oof_counts = np.zeros((n, n_models), dtype=int)

    cv = RepeatedStratifiedKFold(n_splits=N_CV_SPLITS, n_repeats=N_CV_REPEATS,
                                 random_state=seed)

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        for j, (name, (cls, params)) in enumerate(model_configs.items()):
            m = cls(**params)
            m.fit(X_tr, y_tr)
            oof_sum[val_idx, j]    += m.predict_proba(X_val)[:, 1]
            oof_counts[val_idx, j] += 1

    oof_matrix = oof_sum / np.maximum(oof_counts, 1)
    return oof_matrix


# ===== BUILD FINAL MODELS =====

def build_final_models(model_configs: dict, X: np.ndarray, y: np.ndarray) -> dict:
    final = {}
    for name, (cls, params) in model_configs.items():
        m = cls(**params)
        m.fit(X, y)
        final[name] = m
    return final


# ===== META-MODEL (calibration + weighting) =====

def fit_meta_model(oof_matrix, y, seed=RANDOM_SEED):
    """
    LogisticRegression on OOF predictions only.
    Learns ensemble weights and calibrates probabilities using player-skill
    signals exclusively — market odds are intentionally excluded.
    """
    meta_X = oof_matrix
    scaler = StandardScaler()
    meta_X_scaled = scaler.fit_transform(meta_X)
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    meta.fit(meta_X_scaled, y)
    return meta, scaler


# ===== MARKET TRAINING FUNCTION =====

def train_market(market_name, market_config, train_df, tour_key, model_vars):
    print(f"\n  --- {market_name} ---")

    target_col = market_config["target_col"]
    odds_col   = market_config["odds_col"]

    # Subset to available columns
    available_vars = [v for v in model_vars if v in train_df.columns]
    missing = [v for v in model_vars if v not in train_df.columns]
    if missing:
        print(f"    Warning: missing vars (skipped): {missing}")

    df = train_df[available_vars + [target_col, odds_col]].dropna()
    X   = df[available_vars].values.astype(float)
    y   = df[target_col].values.astype(int)
    odds = df[odds_col].values.astype(float)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw   = n_neg / max(n_pos, 1)   # scale_pos_weight for XGBoost
    print(f"    Samples: {len(y):,}  |  Positives: {n_pos} ({100*n_pos/len(y):.1f}%)")

    # --- Load warm-start params from last season's best (if saved) ---
    def load_warm(model_name):
        p = MODELS_DIR / f"{tour_key}_{market_name}_{model_name}_best_params.pkl"
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass
        return None

    # --- Tune each model ---
    print(f"    Tuning {OPTUNA_TRIALS} Optuna trials per model...")

    logistic_params = tune_logistic( X, y, OPTUNA_TRIALS, warm_params=load_warm("logistic"))
    rf_params       = tune_rf(       X, y, OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("rf"))
    lgbm_params     = tune_lgbm(     X, y, OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("lgbm"))
    xgb_params      = tune_xgb(      X, y, OPTUNA_TRIALS, scale_pos_weight=spw,
                                     warm_params=load_warm("xgb"))
    dart_params     = tune_lgbm_dart(X, y, OPTUNA_TRIALS, n_pos=n_pos, warm_params=load_warm("lgbm_dart"))

    # Save best params for next season warm-start
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, params in [("logistic", logistic_params), ("rf", rf_params),
                          ("lgbm", lgbm_params), ("xgb", xgb_params),
                          ("lgbm_dart", dart_params)]:
        joblib.dump(params, MODELS_DIR / f"{tour_key}_{market_name}_{name}_best_params.pkl")

    # --- Build model configurations ---
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

    # --- OOF predictions ---
    print(f"    Generating OOF predictions "
          f"({N_CV_SPLITS}-fold × {N_CV_REPEATS} repeats)...")
    oof_matrix = generate_oof(model_configs, model_configs, X, y)

    # --- Per-model OOF metrics ---
    model_names = list(model_configs.keys())
    metrics = {}
    for j, name in enumerate(model_names):
        oof_j = np.clip(oof_matrix[:, j], 1e-15, 1 - 1e-15)
        ll  = log_loss(y, oof_j)
        auc = roc_auc_score(y, oof_j)
        ap  = average_precision_score(y, oof_j)
        tss = tss_optimal(y, oof_j)
        metrics[name] = {"log_loss": ll, "roc_auc": auc, "avg_precision": ap, "tss": tss}
        print(f"      {name:12s}  log_loss={ll:.4f}  AUC={auc:.4f}  AP={ap:.4f}  TSS={tss:.4f}")

    # --- Meta-model (calibration) ---
    meta_model, meta_scaler = fit_meta_model(oof_matrix, y)

    # --- Final models on full data ---
    print("    Fitting final models on full training data...")
    final_models = build_final_models(model_configs, X, y)

    return {
        "models": final_models,
        "model_names": model_names,
        "meta_model": meta_model,
        "meta_scaler": meta_scaler,
        "model_vars": available_vars,
        "odds_col": odds_col,
        "market_size": market_config["market_size"],
        "metrics": metrics,
        "n_samples": len(y),
        "n_positives": n_pos,
        "trained_at": datetime.now(),
    }


# ===== TRAINING DATA SELECTION =====

def get_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Last TRAINING_YEARS complete calendar years (mirrors R date-based filter)."""
    max_year = int(df["Date"].dt.year.max())
    end_year   = max_year - 1
    start_year = max_year - TRAINING_YEARS
    mask = (df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)
    subset = df[mask]
    print(f"  Training window: {start_year}–{end_year}  "
          f"({mask.sum():,} rows from {len(df):,})")
    return subset


def get_market_vars(market_config: dict) -> list:
    """Base vars only — market odds are excluded from base model features.
    Odds are used solely in the meta-model for calibration (via implied_odds)."""
    odds_cols = {"Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"}
    return [v for v in BASE_MODEL_VARS if v not in odds_cols]


# ===== SAVE TRAINING SUMMARY =====

def save_summary(all_market_results: dict, tour_key: str):
    rows = []
    for market_name, res in all_market_results.items():
        for model_name, m in res["metrics"].items():
            rows.append({
                "Tour":       tour_key,
                "Market":     market_name,
                "Model":      model_name,
                "Log_Loss":       round(m["log_loss"],       5),
                "ROC_AUC":        round(m["roc_auc"],        4),
                "Avg_Precision":  round(m["avg_precision"],  4),
                "TSS":            round(m["tss"],             4),
                "N_Samples":  res["n_samples"],
                "N_Positives": res["n_positives"],
            })
    summary_df = pd.DataFrame(rows)
    path = MODELS_DIR / f"{tour_key}_Training_Summary_{SEASON_SUFFIX}.xlsx"
    summary_df.to_excel(path, index=False)
    print(f"\n  Training summary saved: {path.name}")
    return summary_df


# ===== PROCESS SINGLE TOUR =====

def process_tour(tour_key: str, tour_info: dict):
    print(f"\n{'='*60}")
    print(f"TRAINING: {tour_info['name']}")
    print(f"{'='*60}")

    hist_path = tour_info["historical_file"]
    if not hist_path.exists():
        print(f"  Historical file not found: {hist_path}")
        return None

    df = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date"])
    print(f"  Loaded {len(df):,} rows from {hist_path.name}")

    train_df = get_training_data(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_market_results = {}

    for market_name, market_config in BETTING_MARKETS.items():
        model_vars = get_market_vars(market_config)
        result = train_market(market_name, market_config, train_df, tour_key, model_vars)
        all_market_results[market_name] = result

    # Save model package
    package = {
        "markets": all_market_results,
        "tour_key": tour_key,
        "season": SEASON_SUFFIX,
        "trained_at": datetime.now(),
    }
    model_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"
    joblib.dump(package, model_path)
    print(f"\n  Models saved: {model_path.name}")

    save_summary(all_market_results, tour_key)
    return package


# ===== MAIN =====

def main():
    start = datetime.now()
    print("=== GOLF TOURNAMENT MODEL TRAINING (PYTHON) ===")
    print(f"Season: {SEASON_SUFFIX}  |  Training years: {TRAINING_YEARS}")
    print(f"Optuna trials per model: {OPTUNA_TRIALS}")
    print(f"OOF: {N_CV_SPLITS}-fold × {N_CV_REPEATS} repeats")

    results = {}
    for tour_key, tour_info in TOUR_CONFIG.items():
        try:
            results[tour_key] = process_tour(tour_key, tour_info)
        except Exception as e:
            print(f"\nError processing {tour_key}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n=== TRAINING COMPLETE  ({elapsed:.1f} min) ===")

    for tour_key, res in results.items():
        if res:
            markets = list(res["markets"].keys())
            print(f"  {tour_key}: {len(markets)} markets trained — {markets}")

    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
