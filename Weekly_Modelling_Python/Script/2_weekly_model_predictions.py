"""
2. Weekly Model Predictions (Python)
Mirrors: Weekly_Modelling/Script/2. Weekly Model Predictions.R

Loads trained model packages and generates Win / Top5 / Top10 / Top20 predictions
for the upcoming tournament field.

Output columns match R output for direct comparison:
  Surname, Firstname, Rating, Market_Odds, Model_Score,
  Probability, Normalised_Probability, Normalised_Model_Odds

Run: python 2_weekly_model_predictions.py
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:  # IPython / Jupyter — assume CWD is the Script dir
    sys.path.insert(0, str(Path.cwd()))
from config import (
    BASE_MODEL_VARS,
    BETTING_MARKETS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    SEASON_SUFFIX,
    TOUR_CONFIG,
)

warnings.filterwarnings("ignore")


# ===== HELPERS =====

def get_market_vars(market_config: dict) -> list:
    odds_cols = {"Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"}
    return [v for v in BASE_MODEL_VARS if v not in odds_cols]


def ensemble_predict(market_pkg: dict, X: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """
    1. Get predictions from each base model.
    2. Stack with implied odds as meta-features.
    3. Apply meta-model to produce calibrated probabilities.
    """
    model_preds = np.column_stack([
        model.predict_proba(X)[:, 1]
        for model in market_pkg["models"].values()
    ])
    meta_X_scaled = market_pkg["meta_scaler"].transform(model_preds)
    return (
        market_pkg["meta_model"].predict_proba(meta_X_scaled)[:, 1],
        model_preds.mean(axis=1),   # raw model score (mean of individual model probs)
    )


# ===== PREDICTION FOR ONE MARKET =====

def predict_market(market_name: str, market_pkg: dict, newdat: pd.DataFrame) -> pd.DataFrame:
    market_config = BETTING_MARKETS[market_name]
    model_vars    = market_pkg["model_vars"]
    odds_col      = market_pkg["odds_col"]

    available = [v for v in model_vars if v in newdat.columns]
    missing   = [v for v in model_vars if v not in newdat.columns]
    if missing:
        print(f"    Warning: missing vars (skipped): {missing}")

    df = newdat[available + [odds_col]].copy()
    df = df.dropna()
    X    = df[available].values.astype(float)
    odds = df[odds_col].values.astype(float)

    proba, raw_score = ensemble_predict(market_pkg, X, odds)

    # Tournament-level normalisation: probabilities sum to market_size
    market_size = market_pkg["market_size"]
    prob_sum    = proba.sum()
    norm_prob   = (proba / prob_sum) * market_size if prob_sum > 0 else proba

    results = newdat.loc[df.index].copy()
    results["Model_Score"]            = raw_score.round(5)
    results["Probability"]            = proba.round(6)
    results["Normalised_Probability"] = norm_prob.round(6)
    results["Model_Odds"]             = (1.0 / proba.clip(1e-8)).round(2)
    results["Normalised_Model_Odds"]  = (1.0 / norm_prob.clip(1e-8)).round(2)

    # Sort ascending by market odds
    results = results.sort_values(odds_col)
    return results


# ===== EXPORT =====

def export_predictions(tour_results: dict, tour_key: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%d-%m-%Y")
    out_path = PREDICTIONS_DIR / f"{tour_key}_Predictions_{date_str}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for market_name, df in tour_results.items():
            if df is None or len(df) == 0:
                continue

            odds_col = BETTING_MARKETS[market_name]["odds_col"]

            # Select and rename columns to match R output
            cols_available = []
            rename_map = {}
            for col in ["surname", "firstname"]:
                if col in df.columns:
                    cols_available.append(col)
                    rename_map[col] = col.capitalize()

            out_df = pd.DataFrame()
            out_df["Surname"]   = df.get("surname",   df.get("Surname",   ""))
            out_df["Firstname"] = df.get("firstname", df.get("Firstname", ""))
            if "rating" in df.columns:
                out_df["Rating"] = df["rating"]
            out_df["Market_Odds"]           = df[odds_col]
            out_df["Model_Score"]           = df["Model_Score"]
            out_df["Probability"]           = df["Probability"]
            out_df["Normalised_Probability"] = df["Normalised_Probability"]
            out_df["Normalised_Model_Odds"] = df["Normalised_Model_Odds"]

            out_df.to_excel(writer, sheet_name=f"{market_name}_Market", index=False)

    print(f"  Predictions saved: {out_path.name}")
    return out_path


# ===== PROCESS ONE TOUR =====

def process_tour(tour_key: str, tour_info: dict):
    print(f"\n{'='*60}")
    print(f"PREDICTING: {tour_info['name']}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"
    if not model_path.exists():
        print(f"  Model file not found: {model_path}")
        return None

    weekly_path = tour_info["weekly_file"]
    if not weekly_path.exists():
        print(f"  Weekly file not found: {weekly_path}")
        return None

    package = joblib.load(model_path)
    print(f"  Model trained: {package['trained_at'].strftime('%Y-%m-%d %H:%M')}")

    newdat = pd.read_excel(weekly_path).dropna()
    print(f"  Field size: {len(newdat)} players")

    market_results = {}
    for market_name, market_pkg in package["markets"].items():
        print(f"  Processing {market_name}...")
        df = predict_market(market_name, market_pkg, newdat)
        market_results[market_name] = df
        print(f"    {len(df)} players  |  "
              f"prob range: {df['Probability'].min():.4f}–{df['Probability'].max():.4f}")

    return market_results


# ===== MAIN =====

def main():
    print("=== GOLF TOURNAMENT PREDICTIONS (PYTHON) ===")
    print(f"Season: {SEASON_SUFFIX}  |  Date: {datetime.now().strftime('%d-%m-%Y')}")

    all_results = {}
    output_files = {}

    for tour_key, tour_info in TOUR_CONFIG.items():
        try:
            results = process_tour(tour_key, tour_info)
            if results:
                all_results[tour_key] = results
                output_files[tour_key] = export_predictions(results, tour_key)
        except Exception as e:
            print(f"\nError processing {tour_key}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== PREDICTION SUMMARY ===")
    for tour_key, results in all_results.items():
        print(f"\n{TOUR_CONFIG[tour_key]['name']}:")
        for market_name, df in results.items():
            if df is not None:
                print(f"  {market_name}: {len(df)} players")
        if tour_key in output_files:
            print(f"  Output: {output_files[tour_key].name}")

    if not all_results:
        print("\nNo predictions generated. Check that:")
        print("  1. Processed weekly files exist in Weekly_Modelling_Python/Input/")
        print("  2. Trained models exist in Weekly_Modelling_Python/Output/Models/")
        print("  3. Run 1_weekly_data_preprocessing.py and seasonal_model_training.py first")

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
