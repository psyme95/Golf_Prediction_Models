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

Back betting
------------
Bet when Normalised_Model_Odds < Market_Odds (fixed £10 stake).
  Winner:          P&L = (market_odds - 1) × 10 on win, -10 on loss.
  Top5/Top10/Top20: P&L from pre-computed profit column (dead-heat adjusted).

Grid search
-----------
After all windows are processed a parameter grid is swept over the combined
All_Predictions data. Dimensions:
  Edge threshold  — ratio of market odds to model odds required before betting
  Min/Max odds    — odds range filter
  Min rating      — player rating floor filter
Results are written to the Strategy_Grid sheet of the output workbook.

Output
------
Output/WalkForward/Results/{tour}_WalkForward_Backtest.xlsx
  Summary         — overall metrics + back P&L across all windows, per market
  Season_Summary  — same broken down by test year
  Event_Results   — per-event P&L with cumulative P&L column
  All_Predictions — every player prediction + outcome + bet P&L
  Calib_*         — calibration bins (predicted vs actual) per market
  Strategy_Grid   — grid search results sorted by ROI

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
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

try:
    sys.path.insert(0, str(Path(__file__).parent))
except NameError:
    # IPython/Jupyter or PyCharm console: __file__ is unavailable.
    # Search common locations relative to CWD (project root, Weekly_Modelling_Python, or Script).
    _cwd = Path.cwd()
    _candidates = [_cwd, _cwd / "Weekly_Modelling_Python" / "Script", _cwd / "Script"]
    _script_dir = next((p for p in _candidates if (p / "seasonal_model_training.py").exists()), _cwd)
    sys.path.insert(0, str(_script_dir))

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
    apply_back_strategy,
    back_summary,
    calibration_bins,
    compute_discrimination,
    join_profit_cols,
    predict_event,
)
from seasonal_model_training import get_market_vars, tss_optimal

warnings.filterwarnings("ignore")

# ===== PATHS =====
WF_DIR         = BASE_DIR / "Output" / "WalkForward"
WF_MODELS_DIR  = WF_DIR / "Models"
WF_RESULTS_DIR = WF_DIR / "Results"

# ===== CONSTANTS =====
DEFAULT_WF_TRIALS = 30
MIN_POSITIVES     = 10   # skip a market window if training set has fewer positives

# ===== GRID SEARCH PARAMETERS =====
GRID_EDGE_THRESHOLDS = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
GRID_MIN_ODDS        = [1.0, 3.0, 5.0, 10.0]
GRID_MAX_ODDS        = [9999, 10, 25, 50, 100, 200, 500]   # 9999 = no cap
GRID_MIN_RATING      = [None, 50, 55, 60, 65, 70]           # None = no filter


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
    orig_trials     = smt.OPTUNA_TRIALS
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
                "Test_Year": test_year,
                "EventID":   event_id,
                "Date":      event_date,
                "Market":    market_name,
                "FieldSize": len(preds),
                "Positives": int(preds["Actual"].sum()),
                **bs,
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

    # Cumulative back P&L per market (chronological across all windows)
    for mkt in event_df["Market"].unique():
        mask = event_df["Market"] == mkt
        event_df.loc[mask, "Back_Cumulative_PnL"] = (
            event_df.loc[mask, "Back_PnL"].cumsum().round(2).to_numpy()
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
            season_rows.append({
                "Tour":       tour_key,
                "Test_Year":  test_year,
                "Market":     market_name,
                "N_Events":   int(mdf["EventID"].nunique()),
                "N_Players":  len(mdf),
                "Prevalence": round(float(y_true.mean()), 4),
                **disc, **bs,
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
        summary_rows.append({
            "Tour":          tour_key,
            "Market":        market_name,
            "N_Test_Years":  int(pred_df[pred_df["Market"] == market_name]["Test_Year"].nunique()),
            "N_Events":      int(mdf["EventID"].nunique()),
            "N_Players":     len(mdf),
            "Prevalence":    round(float(y_true.mean()), 4),
            **disc, **bs,
        })
        calib_sheets[market_name] = calibration_bins(y_true, y_prob)

    return {
        "summary":         pd.DataFrame(summary_rows),
        "season_summary":  pd.DataFrame(season_rows),
        "event_results":   event_df,
        "all_predictions": pred_df,
        "calibration":     calib_sheets,
    }


# ===== GRID SEARCH =====

def run_grid_search(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep edge/odds/rating filters across the combined walk-forward predictions.

    For each combination the strategy bets on every row passing the filter
    and computes P&L using the same profit columns used in the main backtest
    (dead-heat adjusted for place markets, formula-based for Winner).
    Results are sorted by ROI%.
    """
    df = pred_df.copy()

    # Build a single Market_Odds column
    conditions = [df["Market"] == "Winner", df["Market"] == "Top5", df["Market"] == "Top10"]
    choices    = [df.get("Win_odds", np.nan), df.get("Top5_odds", np.nan),
                  df.get("Top10_odds", np.nan)]
    df["_Market_Odds"] = np.select(conditions, choices, default=df.get("Top20_odds", np.nan))
    df["_Edge"]        = df["_Market_Odds"] / df["Normalised_Model_Odds"].clip(1e-8)

    # Pre-compute per-row "unit P&L" — what the P&L would be if we bet this row
    # Uses profit columns (dead-heat aware) for place markets, formula for Winner.
    df["_Unit_PnL"] = np.nan
    for market_name, market_config in BETTING_MARKETS.items():
        mask       = df["Market"] == market_name
        profit_col = market_config["profit_col"]
        if profit_col and profit_col in df.columns:
            df.loc[mask, "_Unit_PnL"] = df.loc[mask, profit_col]
        else:
            odds   = df.loc[mask, "_Market_Odds"].to_numpy(dtype=float)
            actual = df.loc[mask, "Actual"].to_numpy(dtype=float)
            df.loc[mask, "_Unit_PnL"] = np.where(
                actual == 1, (odds - 1) * BACK_STAKE, -BACK_STAKE
            )

    market_dfs = {m: df[df["Market"] == m] for m in BETTING_MARKETS}

    combos = [
        (mkt, edge, mn_o, mx_o, mn_r)
        for mkt, edge, mn_o, mx_o, mn_r
        in product(list(BETTING_MARKETS.keys()),
                   GRID_EDGE_THRESHOLDS, GRID_MIN_ODDS, GRID_MAX_ODDS, GRID_MIN_RATING)
        if mn_o < mx_o
    ]
    total = len(combos)
    print(f"\n  Grid search: {total:,} combinations across "
          f"{len(pred_df):,} predictions...")

    results = []
    for i, (mkt, edge, mn_o, mx_o, mn_r) in enumerate(combos):
        if mkt not in market_dfs:
            continue
        sub  = market_dfs[mkt]
        mask = (sub["_Edge"] >= edge) & (sub["_Market_Odds"] >= mn_o) & (sub["_Market_Odds"] <= mx_o)
        if mn_r is not None:
            mask = mask & (sub["rating"] >= mn_r)
        filtered = sub[mask]
        if len(filtered) == 0:
            continue

        pnl_vals     = filtered["_Unit_PnL"].values
        actuals      = filtered["Actual"].values
        odds         = filtered["_Market_Odds"].values
        n_bets       = len(filtered)
        total_staked = n_bets * BACK_STAKE
        total_pnl    = pnl_vals.sum()
        roi          = total_pnl / total_staked if total_staked > 0 else 0

        event_pnl = pd.Series(pnl_vals, index=filtered.index).groupby(
            filtered["EventID"]).sum()
        n_events  = len(event_pnl)
        epnl_std  = event_pnl.std()
        sharpe    = (event_pnl.mean() / epnl_std * np.sqrt(n_events)) if epnl_std > 0 else 0
        cum       = event_pnl.cumsum().values
        max_dd    = (cum - np.maximum.accumulate(cum)).min()

        results.append({
            "Market":         mkt,
            "Edge_Threshold": edge,
            "Min_Odds":       mn_o,
            "Max_Odds":       mx_o if mx_o < 9999 else "None",
            "Min_Rating":     mn_r if mn_r is not None else "None",
            "N_Bets":         n_bets,
            "N_Won":          int(actuals.sum()),
            "Strike_Rate_%":  round(actuals.mean() * 100, 2),
            "Total_Staked":   round(total_staked, 2),
            "Total_PnL":      round(total_pnl, 2),
            "ROI_%":          round(roi * 100, 2),
            "Avg_Odds":       round(odds.mean(), 2),
            "Sharpe":         round(sharpe, 3),
            "Max_Drawdown":   round(max_dd, 2),
        })

        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,}/{total:,} done, {len(results):,} valid so far...")

    if not results:
        return pd.DataFrame()

    grid_df = pd.DataFrame(results).sort_values("ROI_%", ascending=False).reset_index(drop=True)
    n_profitable = (grid_df["Total_PnL"] > 0).sum()
    print(f"  Grid complete: {len(grid_df):,} valid strategies, "
          f"{n_profitable:,} profitable ({100*n_profitable/len(grid_df):.1f}%)")
    print(f"  Best ROI: {grid_df['ROI_%'].max():.1f}%  |  "
          f"Best P&L: £{grid_df['Total_PnL'].max():,.0f}")
    return grid_df


def _write_grid_sheet(wb, grid_df: pd.DataFrame, sheet_name: str = "Strategy_Grid"):
    """Write the grid search results to a styled Excel sheet."""
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    thin_side  = Side(style="thin", color="CCCCCC")
    std_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    center     = Alignment(horizontal="center", vertical="center")
    HDR_FONT   = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    HDR_FILL   = PatternFill("solid", start_color="1F4E79")
    BODY_FONT  = Font(name="Arial", size=9)
    POS_FILL   = PatternFill("solid", start_color="C6EFCE")
    NEG_FILL   = PatternFill("solid", start_color="FFC7CE")
    MID_FILL   = PatternFill("solid", start_color="FFEB9C")

    cols = list(grid_df.columns)
    for col_idx, col_name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = HDR_FONT
        cell.fill      = HDR_FILL
        cell.alignment = center
        cell.border    = std_border

    pnl_col_idx = cols.index("Total_PnL") + 1
    roi_col_idx = cols.index("ROI_%") + 1

    for row_idx, row in grid_df.iterrows():
        excel_row = row_idx + 2
        for col_idx, val in enumerate(row.values, 1):
            cell = ws.cell(row=excel_row, column=col_idx, value=val)
            cell.font      = BODY_FONT
            cell.border    = std_border
            cell.alignment = center
        pnl = row["Total_PnL"]
        ws.cell(row=excel_row, column=pnl_col_idx).fill = (
            POS_FILL if pnl > 0 else NEG_FILL if pnl < -200 else MID_FILL
        )
        ws.cell(row=excel_row, column=roi_col_idx).fill = (
            POS_FILL if row["ROI_%"] > 0 else NEG_FILL
        )

    col_widths = {
        "Market": 10, "Edge_Threshold": 15, "Min_Odds": 11, "Max_Odds": 11,
        "Min_Rating": 12, "N_Bets": 9, "N_Won": 8, "Strike_Rate_%": 14,
        "Total_Staked": 14, "Total_PnL": 13, "ROI_%": 9,
        "Avg_Odds": 11, "Sharpe": 10, "Max_Drawdown": 15,
    }
    for col_idx, col_name in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 12)

    ws.freeze_panes = "A2"


# ===== EXPORT =====

def export_results(results: dict, tour_key: str, grid_df: pd.DataFrame = None) -> Path:
    WF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WF_RESULTS_DIR / f"{tour_key}_WalkForward_Backtest.xlsx"

    id_cols   = [c for c in ["Test_Year", "Date", "EventID", "Market",
                              "surname", "firstname", "posn", "rating"]
                 if c in results["all_predictions"].columns]
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
        results["summary"].to_excel(        writer, sheet_name="Summary",         index=False)
        results["season_summary"].to_excel( writer, sheet_name="Season_Summary",  index=False)
        results["event_results"].to_excel(  writer, sheet_name="Event_Results",   index=False)
        export_df.to_excel(                 writer, sheet_name="All_Predictions", index=False)
        for mkt, calib_df in results["calibration"].items():
            calib_df.to_excel(writer, sheet_name=f"Calib_{mkt}"[:31], index=False)

    # Append the grid search sheet using openpyxl directly (preserves other sheets)
    if grid_df is not None and len(grid_df) > 0:
        wb = load_workbook(out_path)
        _write_grid_sheet(wb, grid_df)
        wb.save(out_path)

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
    df = join_profit_cols(df, tour_info.get("profit_file"))
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

    all_preds_list  = []
    event_summ_list = []
    prev_window_dir = None

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
    print(f"  {'Market':<8}  {'AUC':>6}  {'AP':>6}  {'TSS':>6}  {'Back_PnL':>10}  {'Back_ROI':>9}")
    for _, row in results["summary"].iterrows():
        print(
            f"  {row['Market']:<8}  {row['AUC']:>6.4f}  {row['Avg_Precision']:>6.4f}  "
            f"{row['TSS']:>6.4f}  £{row['Back_PnL']:>9.2f}  {row['Back_ROI']:>8.1%}"
        )

    # Grid search over all windows combined
    grid_df = run_grid_search(results["all_predictions"])

    return results, grid_df


# ===== MAIN =====

def main():
    args = parse_args()

    print("=== GOLF MODEL WALK-FORWARD BACKTESTING ===")
    print(f"Training window: {TRAINING_YEARS} years  |  Optuna trials: {args.trials}")
    print(f"Back stake: £{BACK_STAKE}  |  Place markets use pre-computed dead-heat P&L")
    if args.force_retrain:
        print("  ** Force retrain: ignoring all cached models **")

    tours_to_run = {
        k: v for k, v in TOUR_CONFIG.items()
        if args.tour is None or k == args.tour
    }

    for tour_key, tour_info in tours_to_run.items():
        outcome = run_tour(
            tour_key, tour_info,
            n_trials=args.trials,
            min_test_year=args.min_year,
            force_retrain=args.force_retrain,
        )
        if outcome:
            results, grid_df = outcome
            export_results(results, tour_key, grid_df)


if __name__ == "__main__":
    main()
