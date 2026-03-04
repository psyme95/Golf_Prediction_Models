"""
Backtesting Script
Evaluates the trained model on out-of-sample events for a given year.

The processed historical data contains both training-era events (2024-2025)
and current-year events (2026). The model was trained on 2024-2025, so 2026
events are a clean out-of-sample test.

For each event the script:
  1. Extracts the field (all players in that event from the processed data)
  2. Applies the trained model exactly as the weekly prediction script does
  3. Compares predictions against actual outcomes

Metrics output per tour, per market:
  - Discrimination: AUC, Average Precision, TSS
  - Calibration:    Log-loss, Brier score
  - Betting:        ROI, hit rate, P&L on flat-stake value bets
                    (value bet = Normalised_Model_Probability > Market_Implied_Probability)

Output Excel (one per tour):
  Summary          — key metrics by market
  Event_Results    — per-event P&L breakdown
  All_Predictions  — every player prediction with outcome
  Calibration      — binned predicted vs actual rates for calibration plots

Run: python backtest.py
     python backtest.py --year 2025   (to backtest a specific year)
"""

import argparse
import sys
import warnings
from datetime import datetime
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

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BETTING_MARKETS,
    BASE_MODEL_VARS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEASON_SUFFIX,
    TOUR_CONFIG,
    TRAINING_YEARS,
)
from seasonal_model_training import tss_optimal

warnings.filterwarnings("ignore")


# ===== CONFIGURATION =====

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest golf prediction models")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to backtest (default: auto-detect from data)")
    parser.add_argument("--tour", type=str, default=None,
                        help="Tour to backtest: PGA or Euro (default: both)")
    return parser.parse_args()


def get_backtest_year(df: pd.DataFrame) -> int:
    """
    Auto-detect the backtest year: the most recent year in the data
    that falls outside the training window.
    Training window: last TRAINING_YEARS complete calendar years before max year.
    E.g. max_year=2026 → trains 2024-2025 → backtest year = 2026.
    """
    max_year = int(df["Date"].dt.year.max())
    # Backtest year is max_year if it contains data, otherwise max_year-1
    if (df["Date"].dt.year == max_year).sum() > 0:
        return max_year
    return max_year - 1


def get_market_vars(market_config: dict) -> list:
    odds_cols = {"Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"}
    base_no_odds = [v for v in BASE_MODEL_VARS if v not in odds_cols]
    return base_no_odds + market_config["model_odds_cols"]


# ===== PREDICTION (mirrors 2_weekly_model_predictions.py) =====

def predict_event(event_df: pd.DataFrame, market_name: str,
                  market_pkg: dict) -> pd.DataFrame | None:
    """Apply trained model to a single event's field."""
    model_vars = market_pkg["model_vars"]
    odds_col   = market_pkg["odds_col"]

    available = [v for v in model_vars if v in event_df.columns]
    df = event_df[available + [odds_col]].dropna()

    if len(df) == 0:
        return None

    X    = df[available].values.astype(float)
    odds = df[odds_col].values.astype(float)

    model_preds = np.column_stack([
        m.predict_proba(X)[:, 1] for m in market_pkg["models"].values()
    ])
    raw_score = model_preds.mean(axis=1)

    implied_odds  = 1.0 / np.clip(odds, 1e-8, None)
    meta_X        = np.column_stack([model_preds, implied_odds])
    meta_X_scaled = market_pkg["meta_scaler"].transform(meta_X)
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


# ===== METRICS =====

def compute_discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if y_true.sum() == 0:
        return {"auc": np.nan, "avg_precision": np.nan, "tss": np.nan,
                "log_loss": np.nan, "brier": np.nan}
    y_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "auc":           round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "tss":           round(tss_optimal(y_true, y_prob), 4),
        "log_loss":      round(log_loss(y_true, y_clip), 5),
        "brier":         round(brier_score_loss(y_true, y_prob), 5),
    }


def compute_betting_pnl(df: pd.DataFrame, odds_col: str,
                        target_col: str) -> dict:
    """
    Flat-stake value betting: bet 1 unit when
    Normalised_Probability > market implied probability.
    P&L = (market_odds - 1) on win, -1 on loss.
    """
    df = df.copy()
    df["_implied"] = 1.0 / df[odds_col].clip(lower=1e-8)
    df["_value"]   = df["Normalised_Probability"] > df["_implied"]

    bets = df[df["_value"]]
    if len(bets) == 0:
        return {"n_bets": 0, "n_won": 0, "hit_rate": np.nan,
                "total_pnl": 0.0, "roi": np.nan}

    pnl    = bets.apply(lambda r: r[odds_col] - 1 if r[target_col] == 1 else -1, axis=1)
    n_bets = len(bets)
    n_won  = int(bets[target_col].sum())
    total  = float(pnl.sum())

    return {
        "n_bets":    n_bets,
        "n_won":     n_won,
        "hit_rate":  round(n_won / n_bets, 4),
        "total_pnl": round(total, 2),
        "roi":       round(total / n_bets, 4),
    }


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    """Bin predicted probabilities and compare against actual rates."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "Bin_Low":          round(lo, 2),
            "Bin_High":         round(hi, 2),
            "N":                int(mask.sum()),
            "Mean_Predicted":   round(float(y_prob[mask].mean()), 4),
            "Actual_Rate":      round(float(y_true[mask].mean()), 4),
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
        print(f"  Historical file not found: {hist_path}")
        return None
    if not model_path.exists():
        print(f"  Model file not found: {model_path}")
        return None

    df = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    package    = joblib.load(model_path)

    backtest_df = df[df["Date"].dt.year == backtest_year].copy()
    if len(backtest_df) == 0:
        print(f"  No data found for {backtest_year}")
        return None

    events = backtest_df["eventID"].unique()
    print(f"  Events in {backtest_year}: {len(events)}")
    print(f"  Players: {len(backtest_df):,}")

    all_predictions = []
    event_summaries = []

    for event_id in sorted(events):
        event_df   = backtest_df[backtest_df["eventID"] == event_id].copy()
        event_date = event_df["Date"].iloc[0].strftime("%Y-%m-%d")

        for market_name, market_pkg in package["markets"].items():
            market_config = BETTING_MARKETS[market_name]
            target_col    = market_config["target_col"]
            odds_col      = market_config["odds_col"]

            if target_col not in event_df.columns:
                continue

            preds = predict_event(event_df, market_name, market_pkg)
            if preds is None or len(preds) == 0:
                continue

            # Attach actual outcome
            preds["Actual"] = preds[target_col].astype(int)

            # Value bet P&L for this event
            bet = compute_betting_pnl(preds, odds_col, target_col)

            event_summaries.append({
                "Tour":      tour_key,
                "EventID":   event_id,
                "Date":      event_date,
                "Market":    market_name,
                "FieldSize": len(preds),
                "Positives": int(preds["Actual"].sum()),
                **{f"Bet_{k}": v for k, v in bet.items()},
            })

            preds["Tour"]    = tour_key
            preds["EventID"] = event_id
            preds["Date"]    = event_date
            preds["Market"]  = market_name
            all_predictions.append(preds)

    if not all_predictions:
        print("  No predictions generated.")
        return None

    pred_df = pd.concat(all_predictions, ignore_index=True)
    event_df_summary = pd.DataFrame(event_summaries)

    # ===== AGGREGATE METRICS =====
    summary_rows = []
    calib_sheets = {}

    for market_name, market_config in BETTING_MARKETS.items():
        target_col = market_config["target_col"]
        odds_col   = market_config["odds_col"]

        mdf = pred_df[pred_df["Market"] == market_name]
        if len(mdf) == 0 or target_col not in mdf.columns:
            continue

        y_true = mdf["Actual"].values
        y_prob = mdf["Normalised_Probability"].values

        disc = compute_discrimination(y_true, y_prob)
        bet  = compute_betting_pnl(mdf, odds_col, target_col)
        calib_sheets[market_name] = calibration_bins(y_true, y_prob)

        n_events = mdf["EventID"].nunique()
        prevalence = round(float(y_true.mean()), 4)

        print(f"\n  {market_name} ({n_events} events, prevalence={prevalence:.1%})")
        print(f"    AUC={disc['auc']:.4f}  AP={disc['avg_precision']:.4f}  "
              f"TSS={disc['tss']:.4f}  log_loss={disc['log_loss']:.4f}")
        print(f"    Value bets: {bet['n_bets']}  Won: {bet['n_won']}  "
              f"ROI: {bet['roi']:.1%}  P&L: {bet['total_pnl']:.1f} units")

        summary_rows.append({
            "Tour":         tour_key,
            "Year":         backtest_year,
            "Market":       market_name,
            "N_Events":     n_events,
            "N_Players":    len(mdf),
            "Prevalence":   prevalence,
            "AP_Baseline":  prevalence,  # random ranker baseline
            **disc,
            **{f"Bet_{k}": v for k, v in bet.items()},
        })

    return {
        "summary":        pd.DataFrame(summary_rows),
        "event_results":  event_df_summary,
        "all_predictions": pred_df,
        "calibration":    calib_sheets,
    }


# ===== EXPORT =====

def export_results(results: dict, tour_key: str, backtest_year: int):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{tour_key}_Backtest_{backtest_year}.xlsx"

    # Columns to include in all_predictions sheet
    id_cols   = [c for c in ["Date", "EventID", "Market", "surname", "firstname",
                              "posn", "rating"] if c in results["all_predictions"].columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    odds_cols = [c for c in results["all_predictions"].columns
                 if c.endswith("_odds") or c in ("Win_odds", "Top5_odds",
                                                   "Top10_odds", "Top20_odds")]
    export_cols = id_cols + [c for c in odds_cols
                             if c in results["all_predictions"].columns] + pred_cols

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(writer, sheet_name="Summary", index=False)
        results["event_results"].to_excel(writer, sheet_name="Event_Results", index=False)
        results["all_predictions"][
            [c for c in export_cols if c in results["all_predictions"].columns]
        ].to_excel(writer, sheet_name="All_Predictions", index=False)

        for market_name, calib_df in results["calibration"].items():
            sheet = f"Calib_{market_name}"[:31]  # Excel sheet name limit
            calib_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"\n  Results saved: {out_path.name}")
    return out_path


# ===== MAIN =====

def main():
    args = parse_args()

    print("=== GOLF MODEL BACKTESTING ===")
    print(f"Model season: {SEASON_SUFFIX}  |  Training years: {TRAINING_YEARS}")

    tours_to_run = {k: v for k, v in TOUR_CONFIG.items()
                    if args.tour is None or k == args.tour}

    for tour_key, tour_info in tours_to_run.items():
        try:
            # Determine backtest year
            if args.year:
                backtest_year = args.year
            else:
                hist_path = tour_info["historical_file"]
                if not hist_path.exists():
                    print(f"\nHistorical file not found: {hist_path}")
                    continue
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
