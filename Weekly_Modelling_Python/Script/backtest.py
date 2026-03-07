"""
Backtesting Script
Evaluates the trained model on out-of-sample events for a given year.

The processed historical data contains both training-era events (2024-2025)
and current-year events (2026). The model was trained on 2024-2025, so 2026
events are a clean out-of-sample test.

For each event the script:
  1. Extracts the field (all players in that event from the processed data)
  2. Applies the trained model exactly as the weekly prediction script does
  3. Compares predictions against actual outcomes for the back betting strategy

Back betting strategy
---------------------
Bet when Normalised_Model_Odds < Market_Odds (model thinks player underpriced).
Fixed stake: £10.

  Winner market: P&L = (market_odds - 1) × 10 on win, -10 on loss.
  Top5/Top10/Top20: P&L taken directly from the pre-computed profit column
    (Top5_Profit / Top10_Profit / Top20_Profit) which incorporates dead heat
    rules and is based on a £10 stake.

ROI = total_P&L / (n_bets × 10)

Model quality metrics: AUC, Average Precision, TSS, Log-loss, Brier score

Output Excel (one per tour):
  Summary          — model metrics + back strategy results by market
  Event_Results    — per-event P&L with running cumulative column
  All_Predictions  — every player prediction + outcome + per-bet P&L
  Calibration      — binned predicted vs actual rates for calibration plots

Run: python backtest.py
     python backtest.py --year 2025
     python backtest.py --tour PGA
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))
from config import (
    BASE_MODEL_VARS,
    BETTING_MARKETS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEASON_SUFFIX,
    TOUR_CONFIG,
    TRAINING_YEARS,
)
from seasonal_model_training import tss_optimal

warnings.filterwarnings("ignore")


# ===== BETTING CONSTANTS =====
BACK_STAKE = 10.0     # £ per back bet


# ===== ARGS =====

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest golf prediction models")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to backtest (default: auto-detect from data)")
    parser.add_argument("--tour", type=str, default=None,
                        help="Tour to backtest: PGA or Euro (default: both)")
    return parser.parse_args()


def get_backtest_year(df: pd.DataFrame) -> int:
    max_year = int(df["Date"].dt.year.max())
    if (df["Date"].dt.year == max_year).sum() > 0:
        return max_year
    return max_year - 1


# ===== PROFIT COLUMN JOIN =====

PROFIT_COLS    = ["Top5_Profit", "Top10_Profit", "Top20_Profit"]
PROFIT_JOIN_ON = ["eventID", "surname", "firstname"]


def join_profit_cols(df: pd.DataFrame, profit_file: Path) -> pd.DataFrame:
    """
    Left join Top5/10/20_Profit onto df from the raw profit_file.

    Rows with missing odds in the source will have NaN in the profit columns;
    apply_back_strategy falls back to the formula for those rows.
    Returns df unchanged if the file is missing or contains no usable columns.
    """
    if profit_file is None or not profit_file.exists():
        return df

    try:
        available_join = [c for c in PROFIT_JOIN_ON if c in df.columns]
        if not available_join:
            return df

        raw = pd.read_excel(profit_file, usecols=available_join + PROFIT_COLS)
        raw = raw.dropna(subset=available_join)

        # Drop any profit cols already present in df to avoid _x/_y conflicts
        existing = [c for c in PROFIT_COLS if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

        df = df.merge(raw[available_join + PROFIT_COLS], on=available_join, how="left")
    except Exception as e:
        print(f"  Warning: could not join profit columns from {profit_file.name}: {e}")

    return df


# ===== PREDICTION (mirrors 2_weekly_model_predictions.py) =====

def predict_event(event_df: pd.DataFrame, market_name: str,
                  market_pkg: dict) -> pd.DataFrame | None:
    model_vars = market_pkg["model_vars"]
    odds_col   = market_pkg["odds_col"]

    available = [v for v in model_vars if v in event_df.columns]
    df = event_df[available + [odds_col]].copy()
    for col in available + [odds_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if len(df) == 0:
        return None

    X    = df[available].values.astype(float)
    odds = df[odds_col].values.astype(float)

    model_preds = np.column_stack([
        m.predict_proba(X)[:, 1] for m in market_pkg["models"].values()
    ])
    raw_score = model_preds.mean(axis=1)

    meta_X_scaled = market_pkg["meta_scaler"].transform(model_preds)
    proba         = market_pkg["meta_model"].predict_proba(meta_X_scaled)[:, 1]

    market_size = market_pkg["market_size"]
    prob_sum    = proba.sum()
    norm_prob   = (proba / prob_sum) * market_size if prob_sum > 0 else proba

    result = event_df.loc[df.index].copy()
    result["Model_Score"]            = raw_score.round(5)
    result["Probability"]            = proba.round(6)
    result["Normalised_Probability"] = norm_prob.round(6)
    result["Normalised_Model_Odds"]  = (1.0 / norm_prob.clip(1e-8)).round(2)
    return result


# ===== BACK BETTING STRATEGY =====

def apply_back_strategy(df: pd.DataFrame, odds_col: str,
                        target_col: str,
                        profit_col: str | None = None) -> pd.DataFrame:
    """
    Back when Normalised_Model_Odds < Market_Odds.
    Fixed stake BACK_STAKE per bet.

    P&L source:
      - If profit_col is provided and present in df: use it directly (incorporates
        dead heat rules and is based on BACK_STAKE already).
      - Otherwise: compute from market odds (no dead heat adjustment).
    """
    df = df.copy()
    mkt    = df[odds_col].to_numpy(dtype=float)
    actual = df[target_col].to_numpy(dtype=float)
    model  = df["Normalised_Model_Odds"].to_numpy(dtype=float)
    is_back = model < mkt

    if profit_col and profit_col in df.columns:
        # Pre-computed P&L already has dead heat adjustments baked in
        precomputed = df[profit_col].to_numpy(dtype=float)
        back_pnl = np.where(is_back, precomputed, 0.0)
    else:
        back_pnl = np.where(
            is_back & (actual == 1),  (mkt - 1) * BACK_STAKE,
            np.where(is_back, -BACK_STAKE, 0.0)
        )

    df["Back_Bet"] = is_back
    df["Back_PnL"] = back_pnl
    return df


# ===== STRATEGY SUMMARY =====

def back_summary(df: pd.DataFrame, target_col: str) -> dict:
    bets = df[df["Back_Bet"]]
    if len(bets) == 0:
        return {"Back_NBets": 0, "Back_NWon": 0, "Back_HitRate": np.nan,
                "Back_PnL": 0.0, "Back_ROI": np.nan}
    n_bets = len(bets)
    n_won  = int(bets[target_col].sum())
    pnl    = float(bets["Back_PnL"].sum())
    return {
        "Back_NBets":   n_bets,
        "Back_NWon":    n_won,
        "Back_HitRate": round(n_won / n_bets, 4),
        "Back_PnL":     round(pnl, 2),
        "Back_ROI":     round(pnl / (n_bets * BACK_STAKE), 4),
    }


# ===== MODEL QUALITY METRICS =====

def compute_discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if y_true.sum() == 0:
        return {"AUC": np.nan, "Avg_Precision": np.nan, "TSS": np.nan,
                "Log_Loss": np.nan, "Brier": np.nan}
    y_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "AUC":           round(roc_auc_score(y_true, y_prob), 4),
        "Avg_Precision": round(average_precision_score(y_true, y_prob), 4),
        "TSS":           round(tss_optimal(y_true, y_prob), 4),
        "Log_Loss":      round(log_loss(y_true, y_clip), 5),
        "Brier":         round(brier_score_loss(y_true, y_prob), 5),
    }


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "Bin_Low":        round(lo, 2),
            "Bin_High":       round(hi, 2),
            "N":              int(mask.sum()),
            "Mean_Predicted": round(float(y_prob[mask].mean()), 4),
            "Actual_Rate":    round(float(y_true[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


# ===== MAIN BACKTEST LOOP =====

def run_backtest(tour_key: str, tour_info: dict, backtest_year: int):
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {tour_info['name']}  |  Year: {backtest_year}")
    print(f"{'='*60}")

    hist_path  = tour_info["historical_file"]
    model_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"

    if not hist_path.exists():
        print(f"  Historical file not found: {hist_path}"); return None
    if not model_path.exists():
        print(f"  Model file not found: {model_path}"); return None

    df      = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = join_profit_cols(df, tour_info.get("profit_file"))
    package = joblib.load(model_path)

    backtest_df = df[df["Date"].dt.year == backtest_year].copy()
    if len(backtest_df) == 0:
        print(f"  No data found for {backtest_year}"); return None

    events = backtest_df["eventID"].unique()
    print(f"  Events: {len(events)}  |  Players: {len(backtest_df):,}")

    all_predictions = []
    event_summaries = []

    for event_id in sorted(events):
        event_df   = backtest_df[backtest_df["eventID"] == event_id].copy()
        event_date = event_df["Date"].iloc[0].strftime("%Y-%m-%d")

        for market_name, market_pkg in package["markets"].items():
            market_config = BETTING_MARKETS[market_name]
            target_col    = market_config["target_col"]
            odds_col      = market_config["odds_col"]
            profit_col    = market_config["profit_col"]

            if target_col not in event_df.columns:
                continue

            preds = predict_event(event_df, market_name, market_pkg)
            if preds is None or len(preds) == 0:
                continue

            preds["Actual"] = preds[target_col].astype(int)
            preds = apply_back_strategy(preds, odds_col, target_col, profit_col)

            bs = back_summary(preds, target_col)
            event_summaries.append({
                "Tour":      tour_key,
                "EventID":   event_id,
                "Date":      event_date,
                "Market":    market_name,
                "FieldSize": len(preds),
                "Positives": int(preds["Actual"].sum()),
                **bs,
            })

            preds["Tour"]    = tour_key
            preds["EventID"] = event_id
            preds["Date"]    = event_date
            preds["Market"]  = market_name
            all_predictions.append(preds)

    if not all_predictions:
        print("  No predictions generated."); return None

    pred_df      = pd.concat(all_predictions, ignore_index=True)
    event_sum_df = pd.DataFrame(event_summaries).sort_values(["Market", "Date", "EventID"])

    # Cumulative P&L per market, sorted by date
    for market_name in event_sum_df["Market"].unique():
        mask = event_sum_df["Market"] == market_name
        event_sum_df.loc[mask, "Back_Cumulative_PnL"] = (
            event_sum_df.loc[mask, "Back_PnL"].cumsum().round(2)
        )

    # ===== AGGREGATE METRICS =====
    summary_rows = []
    calib_sheets = {}

    for market_name, market_config in BETTING_MARKETS.items():
        target_col = market_config["target_col"]

        mdf = pred_df[pred_df["Market"] == market_name]
        if len(mdf) == 0 or target_col not in mdf.columns:
            continue

        y_true = mdf["Actual"].values
        y_prob = mdf["Normalised_Probability"].values

        disc = compute_discrimination(y_true, y_prob)
        bs   = back_summary(mdf, target_col)
        calib_sheets[market_name] = calibration_bins(y_true, y_prob)

        n_events   = int(mdf["EventID"].nunique())
        prevalence = round(float(y_true.mean()), 4)

        print(f"\n  {market_name} ({n_events} events, prevalence={prevalence:.1%})")
        print(f"    Model:  AUC={disc['AUC']:.4f}  AP={disc['Avg_Precision']:.4f}  "
              f"TSS={disc['TSS']:.4f}  log_loss={disc['Log_Loss']:.4f}")
        print(f"    Back:   {bs['Back_NBets']} bets  won={bs['Back_NWon']}  "
              f"P&L=£{bs['Back_PnL']:.2f}  ROI={bs['Back_ROI']:.1%}")

        summary_rows.append({
            "Tour":        tour_key,
            "Year":        backtest_year,
            "Market":      market_name,
            "N_Events":    n_events,
            "N_Players":   len(mdf),
            "Prevalence":  prevalence,
            "AP_Baseline": prevalence,
            **disc,
            **bs,
        })

    return {
        "summary":         pd.DataFrame(summary_rows),
        "event_results":   event_sum_df,
        "all_predictions": pred_df,
        "calibration":     calib_sheets,
    }


# ===== EXPORT =====

def export_results(results: dict, tour_key: str, backtest_year: int):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{tour_key}_Backtest_{backtest_year}.xlsx"

    id_cols   = [c for c in ["Date", "EventID", "Market", "surname", "firstname",
                              "posn", "rating"] if c in results["all_predictions"].columns]
    odds_cols = [c for c in ["Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"]
                 if c in results["all_predictions"].columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    bet_cols  = ["Back_Bet", "Back_PnL"]
    all_cols  = id_cols + odds_cols + pred_cols + bet_cols
    export_df = results["all_predictions"][
        [c for c in all_cols if c in results["all_predictions"].columns]
    ]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(      writer, sheet_name="Summary",         index=False)
        results["event_results"].to_excel(writer, sheet_name="Event_Results",   index=False)
        export_df.to_excel(               writer, sheet_name="All_Predictions", index=False)
        for market_name, calib_df in results["calibration"].items():
            calib_df.to_excel(writer, sheet_name=f"Calib_{market_name}"[:31], index=False)

    print(f"\n  Results saved: {out_path.name}")
    return out_path


# ===== MAIN =====

def main():
    args = parse_args()
    print("=== GOLF MODEL BACKTESTING ===")
    print(f"Season: {SEASON_SUFFIX}  |  Training years: {TRAINING_YEARS}")
    print(f"Back stake: £{BACK_STAKE}  |  Place markets use pre-computed dead-heat P&L")

    tours_to_run = {k: v for k, v in TOUR_CONFIG.items()
                    if args.tour is None or k == args.tour}

    for tour_key, tour_info in tours_to_run.items():
        try:
            if args.year:
                backtest_year = args.year
            else:
                hist_path = tour_info["historical_file"]
                if not hist_path.exists():
                    print(f"\nHistorical file not found: {hist_path}"); continue
                df_tmp = pd.read_excel(hist_path, usecols=["Date"])
                df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])
                backtest_year = get_backtest_year(df_tmp)

            print(f"\nBacktest year: {backtest_year}")
            results = run_backtest(tour_key, tour_info, backtest_year)
            if results:
                export_results(results, tour_key, backtest_year)

        except Exception as e:
            print(f"\nError backtesting {tour_key}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== BACKTESTING COMPLETE ===")


if __name__ == "__main__":
    main()
