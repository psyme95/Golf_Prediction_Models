"""
3. Weekly Round 2 Predictions (Python)
Mirrors: Weekly_Modelling/Script/3. Weekly Rd2 Model Predictions.R

Reads the weekly Rd2 CSV (prepared manually with Rd2 data appended to
predictions from Script 2), applies the trained Rd2 ensemble + meta-model,
and outputs calibrated probabilities normalised within the current field.

Input:
  {SHARED_INPUT_DIR}/This_Week_Rd2_{TOUR}.csv
    Expected columns: Surname, Firstname,
                      Normalised_Probability, Model_Score  ← from Script 2 output
                      Rd2Pos, Rd2Lead, AvPosn, Top5, Betfair_rd2

Output columns match R output for direct comparison:
  Surname, Firstname, GLM_Odds_Probability_Median, Model_Score_Median,
  Rd2_Model_Score, Rd2_Calibrated_Prob, Rd2_Normalised_Prob,
  Rd2_Model_Odds, Rd2_Normalised_Odds

Run: python 3_weekly_rd2_model_predictions.py
"""

import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    RD2_MODEL_VARS,
    SEASON_SUFFIX,
    TOUR_CONFIG,
)

warnings.filterwarnings("ignore")


# ===== DATA PREPARATION =====

def prepare_rd2_data(weekly_path: Path, tour_key: str) -> pd.DataFrame:
    print(f"\n--- Preparing {tour_key} Rd2 data ---")
    print(f"  Reading: {weekly_path}")

    if not weekly_path.exists():
        raise FileNotFoundError(f"Rd2 weekly file not found: {weekly_path}")

    df = pd.read_csv(weekly_path)
    print(f"  {len(df)} rows loaded")

    # Rename to match model variable names (mirrors R rename in prediction script)
    rename_map = {}
    if "Normalised_Probability" in df.columns:
        rename_map["Normalised_Probability"] = "GLM_Odds_Probability_Median"
    if "Model_Score" in df.columns:
        rename_map["Model_Score"] = "Model_Score_Median"
    df = df.rename(columns=rename_map)

    # Report missing data before dropping
    check_cols = ["Top5", "Rd2Pos", "AvPosn", "Betfair_rd2", "Rd2Lead"]
    missing_rows = df[df[check_cols].isna().any(axis=1)]
    if len(missing_rows) > 0:
        id_cols = [c for c in ["Surname", "Firstname"] if c in df.columns]
        print(f"\n  WARNING: {len(missing_rows)} players with missing data:")
        print(missing_rows[id_cols + check_cols].to_string(index=False))

    df = df.dropna(subset=check_cols)
    print(f"  Final dataset: {len(df)} players")
    return df


# ===== PREDICTION =====

def predict_rd2(df: pd.DataFrame, model_pkg: dict) -> pd.DataFrame:
    model_vars = model_pkg["model_vars"]

    available = [v for v in model_vars if v in df.columns]
    missing   = [v for v in model_vars if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required Rd2 model variables: {missing}")
    if "Betfair_rd2" not in df.columns:
        raise ValueError("Missing 'Betfair_rd2' column for calibration")

    X    = df[available].values.astype(float)
    odds = df["Betfair_rd2"].values.astype(float)

    # Individual model predictions
    model_preds = np.column_stack([
        m.predict_proba(X)[:, 1]
        for m in model_pkg["models"].values()
    ])
    raw_score = model_preds.mean(axis=1)

    # Meta-model calibration (OOF-trained, skill signals only)
    meta_X_scaled = model_pkg["meta_scaler"].transform(model_preds)
    calibrated_prob = model_pkg["meta_model"].predict_proba(meta_X_scaled)[:, 1]

    # Tournament-level normalisation (win market: probabilities sum to 1)
    prob_sum     = calibrated_prob.sum()
    norm_prob    = calibrated_prob / prob_sum if prob_sum > 0 else calibrated_prob

    result = df.copy()
    result["Rd2_Model_Score"]      = raw_score.round(5)
    result["Rd2_Calibrated_Prob"]  = calibrated_prob.round(6)
    result["Rd2_Normalised_Prob"]  = norm_prob.round(6)
    result["Rd2_Model_Odds"]       = (1.0 / calibrated_prob.clip(1e-8)).round(2)
    result["Rd2_Normalised_Odds"]  = (1.0 / norm_prob.clip(1e-8)).round(2)

    print(f"\n  Prediction summary:")
    print(f"    Calibrated prob range: "
          f"{calibrated_prob.min():.4f} – {calibrated_prob.max():.4f}")
    print(f"    Model odds range: "
          f"{result['Rd2_Model_Odds'].min():.1f} – {result['Rd2_Model_Odds'].max():.1f}")

    return result


# ===== EXPORT =====

def export_rd2_predictions(df: pd.DataFrame, tour_key: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%d-%m-%Y")
    out_path = PREDICTIONS_DIR / f"{tour_key}_R2_Predictions_{date_str}.xlsx"

    # Column selection mirrors R output
    keep_cols = []
    for col in ["Surname", "Firstname", "GLM_Odds_Probability_Median", "Model_Score_Median",
                "Rd2_Model_Score", "Rd2_Calibrated_Prob", "Rd2_Normalised_Prob",
                "Rd2_Model_Odds", "Rd2_Normalised_Odds"]:
        if col in df.columns:
            keep_cols.append(col)

    df[keep_cols].to_excel(out_path, index=False)
    print(f"  Saved: {out_path.name}")
    return out_path


# ===== MAIN =====

def main():
    date_str = datetime.now().strftime("%d-%m-%Y")
    print("=== RD2 PREDICTION SCRIPT (PYTHON) ===")
    print(f"Date: {date_str}  |  Season: {SEASON_SUFFIX}")

    output_files = {}

    for tour_key, tour_info in TOUR_CONFIG.items():
        try:
            model_path = MODELS_DIR / f"{tour_key}_R2_{SEASON_SUFFIX}_trained.pkl"
            if not model_path.exists():
                print(f"\nModel not found for {tour_key}: {model_path}")
                continue

            model_pkg = joblib.load(model_path)
            print(f"\nLoaded {tour_key} Rd2 model  "
                  f"(trained: {model_pkg['trained_at'].strftime('%Y-%m-%d %H:%M')}  "
                  f"| seasons: {model_pkg['train_seasons']})")

            df = prepare_rd2_data(tour_info["rd2_weekly_file"], tour_key)
            result = predict_rd2(df, model_pkg)
            output_files[tour_key] = export_rd2_predictions(result, tour_key)

        except Exception as e:
            print(f"\nError processing {tour_key} Rd2: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== COMPLETE ===")
    for tour_key, path in output_files.items():
        print(f"  {tour_key}: {path}")


if __name__ == "__main__":
    main()
