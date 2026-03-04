"""
Walk-Forward Backtesting Script

Trains on a rolling 2-year window, predicts on the next year, slides forward
through all available data. No pre-trained models required — this script trains
and tests every window from scratch (with warm-starting between adjacent windows).

Example: data available 2018–2026
  Window 1:  Train 2018–2019  →  Test 2020
  Window 2:  Train 2019–2020  →  Test 2021
  Window 3:  Train 2020–2021  →  Test 2022
  Window 4:  Train 2021–2022  →  Test 2023
  Window 5:  Train 2022–2023  →  Test 2024
  Window 6:  Train 2023–2024  →  Test 2025
  Window 7:  Train 2024–2025  →  Test 2026

Caching
-------
Trained model bundles are saved to Output/WalkForward/Models/{tour}_{year}/
so the script can be interrupted and resumed without retraining completed windows.
Use --force-retrain to ignore the cache and retrain everything.

Optuna trials
-------------
Default is 30 trials per model per market (vs 75 in production) to keep runtimes
manageable across many windows. Use --trials 75 for full production quality.
Warm-start: best params from window N are passed as starting suggestions to window N+1.

Output
------
Output/WalkForward/Results/{tour}_WalkForward_Backtest.xlsx
  Summary         — overall metrics + P&L across all windows, per market
  Season_Summary  — same broken down by test year
  Event_Results   — per-event P&L with cumulative P&L columns
  All_Predictions — every player prediction + outcome + bet P&L
  Calib_*         — calibration bins (predicted vs actual) per market

Run:
  python walk_forward_backtest.py                         # both tours, all windows, 30 trials
  python walk_forward_backtest.py --tour PGA
  python walk_forward_backtest.py --trials 75             # full Optuna
  python walk_forward_backtest.py --min-year 2022         # start from 2022 test window
  python walk_forward_backtest.py --force-retrain         # ignore cached models
"""

import argparse
import shutil
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

sys.path.insert(0, str(Path(__file__).parent))

# Import seasonal_model_training as a module so we can monkey-patch
# OPTUNA_TRIALS and MODELS_DIR per window without touching other state.
import seasonal_model_training as smt

from config import (
    BASE_DIR,
    BETTING_MARKETS,
    PREDICTIONS_DIR,
    TOUR_CONFIG,
    TRAINING_YEARS,
)
from backtest import (
    BACK_STAKE,
    LAY_ODDS_MULTIPLIER,
    LAY_TOTAL_LIABILITY,
    apply_back_strategy,
    apply_lay_strategy,
    back_summary,
    calibration_bins,
    compute_discrimination,
    lay_summary,
    predict_event,
)
from seasonal_model_training import get_market_vars, tss_optimal

warnings.filterwarnings("ignore")

# ===== PATHS =====
WF_DIR        = BASE_DIR / "Output" / "WalkForward"
WF_MODELS_DIR = WF_DIR / "Models"
WF_RESULTS_DIR = WF_DIR / "Results"

# ===== CONSTANTS =====
DEFAULT_WF_TRIALS = 30
MIN_POSITIVES     = 10   # skip a market window if training set has fewer positives


# ===== CLI =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for golf prediction models"
    )
    parser.add_argument("--tour", default=None,
                        help="Tour to backtest: PGA or Euro (default: both)")
    parser.add_argument("--trials", type=int, default=DEFAULT_WF_TRIALS,
                        help=f"Optuna trials per model per market "
                             f"(default: {DEFAULT_WF_TRIALS})")
    parser.add_argument("--min-year", type=int, default=None, dest="min_year",
                        help="Earliest test year to evaluate (default: all available)")
    parser.add_argument("--force-retrain", action="store_true", dest="force_retrain",
                        help="Ignore cached models and retrain all windows")
    return parser.parse_args()


# ===== WALK-FORWARD WINDOW LOGIC =====

def get_windows(df: pd.DataFrame, min_test_year: int = None):
    """
    Build list of (train_start, train_end, test_year) tuples.
    Requires at least TRAINING_YEARS of prior data for each test year.
    """
    years = sorted(df["Date"].dt.year.dropna().unique())
    windows = []
    for test_year in years:
        train_end   = test_year - 1
        train_start = test_year - TRAINING_YEARS
        if train_start < years[0]:
            continue   # not enough history
        if min_test_year and test_year < min_test_year:
            continue
        windows.append((int(train_start), int(train_end), int(test_year)))
    return windows


# ===== WINDOW TRAINING =====

def train_window(tour_key: str, train_df: pd.DataFrame, test_year: int,
                 n_trials: int, prev_window_dir: Path = None) -> dict:
    """
    Train all four betting markets for one walk-forward window.

    Monkey-patches smt.OPTUNA_TRIALS and smt.MODELS_DIR so tuning uses
    the walk-forward trial budget and saves/loads warm params to the
    window-specific cache directory rather than the production MODELS_DIR.

    Warm-start: best params from prev_window_dir are copied into this
    window's directory before training begins.

    Returns dict: market_name → market_package (or empty dict if no markets trained).
    """
    window_dir = WF_MODELS_DIR / f"{tour_key}_{test_year}"
    window_dir.mkdir(parents=True, exist_ok=True)

    # Copy warm params from previous window so Optuna starts nearby
    if prev_window_dir and prev_window_dir.exists():
        for f in prev_window_dir.glob("*_best_params.pkl"):
            dest = window_dir / f.name
            if not dest.exists():
                shutil.copy(f, dest)

    # Temporarily redirect module constants so train_market uses our settings
    orig_trials    = smt.OPTUNA_TRIALS
    orig_models_dir = smt.MODELS_DIR
    smt.OPTUNA_TRIALS = n_trials
    smt.MODELS_DIR    = window_dir

    try:
        package = {}
        for market_name, market_config in BETTING_MARKETS.items():
            target_col = market_config["target_col"]

            # Guard: skip if training set is too sparse for this market
            if target_col in train_df.columns:
                n_pos = int(train_df[target_col].sum())
                if n_pos < MIN_POSITIVES:
                    print(f"    {market_name}: skipped "
                          f"({n_pos} positives < {MIN_POSITIVES} minimum)")
                    continue

            model_vars = get_market_vars(market_config)
            result = smt.train_market(
                market_name, market_config, train_df, tour_key, model_vars
            )
            if result is not None:
                package[market_name] = result
    finally:
        # Always restore, even if training raises
        smt.OPTUNA_TRIALS = orig_trials
        smt.MODELS_DIR    = orig_models_dir

    return package


# ===== WINDOW BACKTESTING =====

def backtest_window(tour_key: str, test_df: pd.DataFrame, test_year: int,
                    package: dict):
    """
    Apply all trained markets to every event in test_df.
    Returns (all_predictions list, event_summaries list).
    """
    all_predictions = []
    event_summaries = []

    events = test_df["eventID"].unique()
    print(f"  Test {test_year}: {len(events)} events  |  {len(test_df):,} player-rows")

    for event_id in sorted(events):
        event_df   = test_df[test_df["eventID"] == event_id].copy()
        event_date = event_df["Date"].iloc[0].strftime("%Y-%m-%d")

        for market_name, market_pkg in package.items():
            market_config = BETTING_MARKETS[market_name]
            target_col    = market_config["target_col"]
            odds_col      = market_config["odds_col"]
            market_size   = market_config["market_size"]

            if target_col not in event_df.columns:
                continue

            preds = predict_event(event_df, market_name, market_pkg)
            if preds is None or len(preds) == 0:
                continue

            preds["Actual"]    = preds[target_col].astype(int)
            preds              = apply_back_strategy(preds, odds_col, target_col)
            preds              = apply_lay_strategy( preds, odds_col, target_col, market_size)
            preds["Combo_PnL"] = preds["Back_PnL"] + preds["Lay_PnL"]

            bs = back_summary(preds, target_col)
            ls = lay_summary( preds, target_col)

            event_summaries.append({
                "Tour":      tour_key,
                "Test_Year": test_year,
                "EventID":   event_id,
                "Date":      event_date,
                "Market":    market_name,
                "FieldSize": len(preds),
                "Positives": int(preds["Actual"].sum()),
                **bs,
                **ls,
                "Combo_PnL": round(float(preds["Combo_PnL"].sum()), 2),
            })

            preds["Tour"]      = tour_key
            preds["Test_Year"] = test_year
            preds["EventID"]   = event_id
            preds["Date"]      = event_date
            preds["Market"]    = market_name
            all_predictions.append(preds)

    return all_predictions, event_summaries


# ===== AGGREGATE ACROSS WINDOWS =====

def aggregate_results(all_preds_list: list, event_summaries_list: list,
                      tour_key: str) -> dict:
    """Combine results from all walk-forward windows into summary DataFrames."""
    if not all_preds_list:
        return None

    pred_df  = pd.concat(all_preds_list, ignore_index=True)
    event_df = (
        pd.DataFrame(event_summaries_list)
        .sort_values(["Market", "Test_Year", "Date", "EventID"])
        .reset_index(drop=True)
    )

    # Cumulative P&L per market (chronological across all windows)
    for mkt in event_df["Market"].unique():
        mask = event_df["Market"] == mkt
        for col, cum_col in [
            ("Back_PnL",  "Back_Cumulative_PnL"),
            ("Lay_PnL",   "Lay_Cumulative_PnL"),
            ("Combo_PnL", "Combo_Cumulative_PnL"),
        ]:
            event_df.loc[mask, cum_col] = (
                event_df.loc[mask, col].cumsum().round(2).to_numpy()
            )

    # Per-season summary (test year × market)
    season_rows = []
    for test_year in sorted(pred_df["Test_Year"].unique()):
        for market_name, market_config in BETTING_MARKETS.items():
            target_col = market_config["target_col"]
            mdf = pred_df[
                (pred_df["Test_Year"] == test_year) &
                (pred_df["Market"]    == market_name)
            ]
            if len(mdf) == 0 or target_col not in mdf.columns:
                continue
            y_true = mdf["Actual"].values
            y_prob = mdf["Normalised_Probability"].values
            disc = compute_discrimination(y_true, y_prob)
            bs   = back_summary(mdf, target_col)
            ls   = lay_summary( mdf, target_col)
            season_rows.append({
                "Tour":      tour_key,
                "Test_Year": test_year,
                "Market":    market_name,
                "N_Events":  int(mdf["EventID"].nunique()),
                "N_Players": len(mdf),
                "Prevalence": round(float(y_true.mean()), 4),
                **disc, **bs, **ls,
                "Combo_PnL": round(float(mdf["Combo_PnL"].sum()), 2),
            })

    # Overall summary across all windows (market level)
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
        ls   = lay_summary( mdf, target_col)
        summary_rows.append({
            "Tour":         tour_key,
            "Market":       market_name,
            "N_Test_Years": int(pred_df[pred_df["Market"] == market_name]["Test_Year"].nunique()),
            "N_Events":     int(mdf["EventID"].nunique()),
            "N_Players":    len(mdf),
            "Prevalence":   round(float(y_true.mean()), 4),
            **disc, **bs, **ls,
            "Combo_PnL": round(float(mdf["Combo_PnL"].sum()), 2),
        })
        calib_sheets[market_name] = calibration_bins(y_true, y_prob)

    return {
        "summary":         pd.DataFrame(summary_rows),
        "season_summary":  pd.DataFrame(season_rows),
        "event_results":   event_df,
        "all_predictions": pred_df,
        "calibration":     calib_sheets,
    }


# ===== EXPORT =====

def export_results(results: dict, tour_key: str) -> Path:
    WF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WF_RESULTS_DIR / f"{tour_key}_WalkForward_Backtest.xlsx"

    id_cols   = [c for c in ["Test_Year", "Date", "EventID", "Market",
                              "surname", "firstname", "posn", "rating"]
                 if c in results["all_predictions"].columns]
    odds_cols = [c for c in ["Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"]
                 if c in results["all_predictions"].columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    bet_cols  = ["Back_Bet", "Back_PnL",
                 "Lay_Bet", "Lay_Stake", "Lay_Liability", "Lay_PnL",
                 "Combo_PnL"]
    all_cols  = id_cols + odds_cols + pred_cols + bet_cols
    export_df = results["all_predictions"][
        [c for c in all_cols if c in results["all_predictions"].columns]
    ]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(        writer, sheet_name="Summary",        index=False)
        results["season_summary"].to_excel( writer, sheet_name="Season_Summary", index=False)
        results["event_results"].to_excel(  writer, sheet_name="Event_Results",  index=False)
        export_df.to_excel(                 writer, sheet_name="All_Predictions", index=False)
        for mkt, calib_df in results["calibration"].items():
            calib_df.to_excel(writer, sheet_name=f"Calib_{mkt}"[:31], index=False)

    print(f"\n  Saved: {out_path}")
    return out_path


# ===== TOUR RUNNER =====

def run_tour(tour_key: str, tour_info: dict, n_trials: int,
             min_test_year: int, force_retrain: bool):
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {tour_info['name']}")
    print(f"{'='*60}")

    hist_path = tour_info["historical_file"]
    if not hist_path.exists():
        print(f"  Processed file not found: {hist_path}")
        return None

    df = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date"])
    print(f"  Loaded {len(df):,} rows  |  years: "
          f"{int(df['Date'].dt.year.min())}–{int(df['Date'].dt.year.max())}")

    windows = get_windows(df, min_test_year=min_test_year)
    if not windows:
        print("  No valid walk-forward windows found."); return None

    print(f"\n  Walk-forward plan ({len(windows)} windows):")
    for ts, te, ty in windows:
        cached = (WF_MODELS_DIR / f"{tour_key}_{ty}" / "market_bundle.pkl").exists()
        status = " [cached]" if cached and not force_retrain else ""
        print(f"    Train {ts}–{te}  →  Test {ty}{status}")

    all_preds_list    = []
    event_summ_list   = []
    prev_window_dir   = None

    for train_start, train_end, test_year in windows:
        print(f"\n  --- Train {train_start}–{train_end} → Test {test_year} ---")

        bundle_path = WF_MODELS_DIR / f"{tour_key}_{test_year}" / "market_bundle.pkl"

        if bundle_path.exists() and not force_retrain:
            print(f"  Loading cached bundle for {test_year}...")
            package = joblib.load(bundle_path)
        else:
            train_df = df[
                (df["Date"].dt.year >= train_start) &
                (df["Date"].dt.year <= train_end)
            ].copy()
            print(f"  Training rows: {len(train_df):,}")

            package = train_window(
                tour_key, train_df, test_year, n_trials, prev_window_dir
            )
            if not package:
                print(f"  No markets trained — skipping test year {test_year}")
                prev_window_dir = WF_MODELS_DIR / f"{tour_key}_{test_year}"
                continue

            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(package, bundle_path)
            print(f"  Bundle saved: {bundle_path.parent.name}/market_bundle.pkl")

        prev_window_dir = WF_MODELS_DIR / f"{tour_key}_{test_year}"

        # Backtest on test year
        test_df = df[df["Date"].dt.year == test_year].copy()
        if len(test_df) == 0:
            print(f"  No test data for {test_year}"); continue

        window_preds, window_events = backtest_window(
            tour_key, test_df, test_year, package
        )
        all_preds_list.extend(window_preds)
        event_summ_list.extend(window_events)

    results = aggregate_results(all_preds_list, event_summ_list, tour_key)
    if results is None:
        print("  No results to aggregate."); return None

    # Console summary
    print(f"\n  === AGGREGATE WALK-FORWARD RESULTS ({len(windows)} windows) ===")
    print(f"  {'Market':<8}  {'AUC':>6}  {'AP':>6}  {'TSS':>6}  "
          f"{'Back_PnL':>10}  {'Lay_PnL':>10}  {'Combo':>10}")
    for _, row in results["summary"].iterrows():
        print(
            f"  {row['Market']:<8}  {row['AUC']:>6.4f}  {row['Avg_Precision']:>6.4f}  "
            f"{row['TSS']:>6.4f}  £{row['Back_PnL']:>9.2f}  "
            f"£{row['Lay_PnL']:>9.2f}  £{row['Combo_PnL']:>9.2f}"
        )

    return results


# ===== MAIN =====

def main():
    args = parse_args()

    print("=== GOLF MODEL WALK-FORWARD BACKTESTING ===")
    print(f"Training window: {TRAINING_YEARS} years  |  Optuna trials: {args.trials}")
    print(f"Back stake: £{BACK_STAKE}  |  Lay total liability: £{LAY_TOTAL_LIABILITY}  "
          f"|  Lay odds multiplier: {LAY_ODDS_MULTIPLIER}×")
    if args.force_retrain:
        print("  ** Force retrain: ignoring all cached models **")

    tours_to_run = {
        k: v for k, v in TOUR_CONFIG.items()
        if args.tour is None or k == args.tour
    }

    for tour_key, tour_info in tours_to_run.items():
        results = run_tour(
            tour_key, tour_info,
            n_trials=args.trials,
            min_test_year=args.min_year,
            force_retrain=args.force_retrain,
        )
        if results:
            export_results(results, tour_key)


if __name__ == "__main__":
    main()
