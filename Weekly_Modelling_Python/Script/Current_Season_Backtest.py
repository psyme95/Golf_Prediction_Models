"""
Seasonal Model Backtester
=========================
Loads a pre-trained seasonal model bundle (.pkl) and backtests it against
events that have already occurred in the current year.

Unlike walk_forward_backtest.py this script never trains — it only evaluates.
It handles both bundle formats:
  - Flat:    {market_name: market_pkg, ...}          (walk-forward output)
  - Wrapped: {"markets": {market_name: market_pkg}}  (seasonal training output)

Usage:
  python seasonal_backtest.py --tour Euro
  python seasonal_backtest.py --tour PGA
  python seasonal_backtest.py --tour Euro --year 2026
  python seasonal_backtest.py --tour Euro --no-grid   # skip grid search

Paths (hardcoded):
  Models : Output/Models/{tour}_Trained_Models_S26.pkl
  Data   : Input/{tour}_Processed.xlsx
  Lay    : Input/{tour}.xlsx

Output:
  Output/SeasonalBacktest/{tour}_Seasonal_Backtest_{year}.xlsx
    Summary         — overall metrics + P&L per market
    Event_Results   — per-event P&L with cumulative P&L column
    All_Predictions — every player prediction + outcome + bet P&L
    Back_Strategy_Grid / Lay_Strategy_Grid  (unless --no-grid)
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import shared config and all backtest logic from walk_forward_backtest ──
try:
    from config import BASE_DIR, BETTING_MARKETS, TOUR_CONFIG
except ImportError:
    sys.exit("ERROR: config.py not found. Run this script from the repo root.")

try:
    import walk_forward_backtest as wfb
except ImportError:
    sys.exit("ERROR: walk_forward_backtest.py not found. Run this script from the repo root.")

# ── Fixed paths ──
SCRIPT_DIR           = Path(__file__).resolve().parent
REPO_DIR             = SCRIPT_DIR.parent                          # ..\Weekly_Modelling_Python
MODELS_DIR           = REPO_DIR / "Output" / "Models"
INPUT_DIR            = REPO_DIR / "Input"
BACKTEST_RESULTS_DIR = REPO_DIR / "Output" / "SeasonalBacktest"

MODEL_PATHS = {
    "Euro": MODELS_DIR / "Euro_Trained_Models_S26.pkl",
    "PGA":  MODELS_DIR / "PGA_Trained_Models_S26.pkl",
}

DATA_PATHS = {
    "Euro": INPUT_DIR / "Euro_Processed.xlsx",
    "PGA":  INPUT_DIR / "PGA_Processed.xlsx",
}

LAY_ODDS_SOURCE = {
    "Euro": INPUT_DIR / "Euro.xlsx",
    "PGA":  INPUT_DIR / "PGA.xlsx",
}


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest a pre-trained seasonal golf model against completed events."
    )
    parser.add_argument(
        "--tour", default=None, choices=list(wfb.TOUR_CONFIG.keys()),
        help="Tour to evaluate: PGA or Euro (default: both)"
    )
    parser.add_argument(
        "--year", type=int, default=2026, dest="year",

        help="Year to evaluate (default: 2026)"
    )
    parser.add_argument(
        "--no-grid", action="store_true", dest="no_grid",
        help="Skip back/lay strategy grid search"
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# BUNDLE LOADING
# ══════════════════════════════════════════════════════════════════

def load_bundle(model_path: Path) -> dict:
    """
    Load a model bundle and normalise it to a flat dict:
        {market_name: market_pkg, ...}

    Handles two formats:
      - Flat    (walk-forward / joblib output):
            {"Winner": {...}, "Top5": {...}, ...}
      - Wrapped (seasonal training output):
            {"markets": {"Winner": {...}, "Top5": {...}, ...}}
    """
    if not model_path.exists():
        sys.exit(f"ERROR: Model file not found: {model_path}")

    print(f"  Loading model bundle: {model_path.name}")
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        sys.exit(f"ERROR: Expected a dict in the pkl, got {type(bundle)}")

    # Unwrap if the seasonal training script adds a "markets" wrapper
    if "markets" in bundle and isinstance(bundle["markets"], dict):
        print("  Detected wrapped bundle format — unwrapping 'markets' key")
        bundle = bundle["markets"]

    valid_markets = set(BETTING_MARKETS.keys())
    found = [k for k in bundle.keys() if k in valid_markets]
    unknown = [k for k in bundle.keys() if k not in valid_markets]

    if not found:
        sys.exit(
            f"ERROR: No recognised market keys found in bundle.\n"
            f"  Bundle keys : {list(bundle.keys())}\n"
            f"  Expected any of: {list(valid_markets)}"
        )
    if unknown:
        print(f"  WARNING: Ignoring unrecognised bundle keys: {unknown}")

    print(f"  Markets loaded: {found}")
    return {k: bundle[k] for k in found}


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_tour_data(tour_key: str, year: int) -> pd.DataFrame:
    """
    Load the processed data file, join Lay odds, and filter to
    completed events in `year` (events with at least one non-null posn).
    """
    hist_path = DATA_PATHS[tour_key]
    lay_path  = LAY_ODDS_SOURCE[tour_key]

    if not hist_path.exists():
        sys.exit(f"ERROR: Processed data file not found: {hist_path}")

    df = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date"])

    # Join Betfair Lay odds (stripped during preprocessing)
    df = wfb.join_lay_odds(df, lay_path)

    year_df = df[df["Date"].dt.year == year].copy()

    if len(year_df) == 0:
        sys.exit(f"ERROR: No data found for {tour_key} year {year}")

    # Filter to completed events only (posn is populated)
    if "posn" in year_df.columns:
        completed_events = (
            year_df.groupby("eventID")["posn"]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").notna().any())
        )
        completed_ids = completed_events[completed_events].index
        incomplete_ids = completed_events[~completed_events].index

        if len(incomplete_ids) > 0:
            print(f"  Excluding {len(incomplete_ids)} incomplete/upcoming events")

        year_df = year_df[year_df["eventID"].isin(completed_ids)].copy()
    else:
        print("  WARNING: 'posn' column not found — including all events in year")

    n_events = year_df["eventID"].nunique()
    print(f"  {year} data: {n_events} completed events  |  {len(year_df):,} player-rows")

    if n_events == 0:
        sys.exit(f"ERROR: No completed events found for {tour_key} {year}")

    return year_df


# ══════════════════════════════════════════════════════════════════
# AGGREGATION  (simplified — no multi-window season summary needed)
# ══════════════════════════════════════════════════════════════════

def aggregate_results(all_preds_list: list, event_summaries_list: list,
                      tour_key: str, year: int) -> dict:
    """Combine per-event results into summary DataFrames."""
    if not all_preds_list:
        return None

    pred_df  = pd.concat(all_preds_list, ignore_index=True)
    event_df = (
        pd.DataFrame(event_summaries_list)
        .sort_values(["Market", "Date", "EventID"])
        .reset_index(drop=True)
    )

    # Cumulative P&L per market
    for mkt in event_df["Market"].unique():
        mask = event_df["Market"] == mkt
        event_df.loc[mask, "Back_Cumulative_PnL"] = (
            event_df.loc[mask, "Back_PnL"].cumsum().round(2).to_numpy()
        )
        event_df.loc[mask, "Lay_Cumulative_PnL_FixedLiab"] = (
            event_df.loc[mask, "Lay_PnL_FixedLiab"].cumsum().round(2).to_numpy()
        )
        event_df.loc[mask, "Lay_Cumulative_PnL_FixedStake"] = (
            event_df.loc[mask, "Lay_PnL_FixedStake"].cumsum().round(2).to_numpy()
        )

    # Overall summary per market
    summary_rows = []
    for market_name, market_config in BETTING_MARKETS.items():
        target_col = market_config["target_col"]
        mdf = pred_df[pred_df["Market"] == market_name]
        if len(mdf) == 0 or target_col not in mdf.columns:
            continue
        y_true = mdf["Actual"].values
        y_prob = mdf["Normalised_Probability"].values
        disc = wfb.compute_discrimination(y_true, y_prob)
        bs   = wfb.back_summary(mdf, target_col)
        ls   = wfb.lay_summary(mdf, target_col, market_name)
        summary_rows.append({
            "Tour":       tour_key,
            "Year":       year,
            "Market":     market_name,
            "N_Events":   int(mdf["EventID"].nunique()),
            "N_Players":  len(mdf),
            "Prevalence": round(float(y_true.mean()), 4),
            **disc, **bs, **ls,
        })

    return {
        "summary":         pd.DataFrame(summary_rows),
        "event_results":   event_df,
        "all_predictions": pred_df,
    }


# ══════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════

def export_results(results: dict, tour_key: str, year: int,
                   back_grid_df: pd.DataFrame = None,
                   lay_grid_df: pd.DataFrame = None) -> Path:
    BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKTEST_RESULTS_DIR / f"{tour_key}_Seasonal_Backtest_{year}.xlsx"

    pred_df   = results["all_predictions"]
    id_cols   = [c for c in ["Date", "EventID", "Market", "playerID",
                              "surname", "firstname", "posn", "rating"]
                 if c in pred_df.columns]
    lay_cols  = [c for c in wfb.LAY_ODDS_COLS.values() if c in pred_df.columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    bet_cols  = ["Back_Bet", "DeadHeat_RF", "Back_PnL",
                 "Lay_Bet", "Lay_PnL_FixedLiab", "Lay_PnL_FixedStake"]
    all_cols  = id_cols + lay_cols + pred_cols + bet_cols
    export_df = pred_df[[c for c in all_cols if c in pred_df.columns]]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(      writer, sheet_name="Summary",         index=False)
        results["event_results"].to_excel(writer, sheet_name="Event_Results",   index=False)
        export_df.to_excel(               writer, sheet_name="All_Predictions", index=False)

    # Grid sheets use the same styled writer from wfb
    if (back_grid_df is not None and len(back_grid_df) > 0) or \
       (lay_grid_df  is not None and len(lay_grid_df)  > 0):
        from openpyxl import load_workbook
        wb = load_workbook(out_path)
        if back_grid_df is not None and len(back_grid_df) > 0:
            wfb._write_grid_sheet(wb, back_grid_df,
                                  sheet_name="Back_Strategy_Grid",
                                  pnl_cols=["Total_PnL"],
                                  roi_cols=["ROI_%"])
        if lay_grid_df is not None and len(lay_grid_df) > 0:
            wfb._write_grid_sheet(wb, lay_grid_df,
                                  sheet_name="Lay_Strategy_Grid",
                                  pnl_cols=["FL_Total_PnL", "FS_Total_PnL"],
                                  roi_cols=["FL_ROI_%", "FS_ROI_%"])
        wb.save(out_path)

    print(f"\n  Saved: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    tours = [args.tour] if args.tour else list(MODEL_PATHS.keys())

    for tour in tours:
        model_path = MODEL_PATHS[tour]
        data_path  = DATA_PATHS[tour]

        print(f"\n=== GOLF MODEL SEASONAL BACKTESTER: {tour} ===")
        print(f"  Year  : {args.year}")
        print(f"  Model : {model_path}")
        print(f"  Data  : {data_path}")
        print("=" * 45)

        package  = load_bundle(model_path)
        test_df  = load_tour_data(tour, args.year)

        all_preds, event_summaries = wfb.backtest_window(
            tour, test_df, args.year, package
        )

        if not all_preds:
            print(f"\nWARNING: No predictions generated for {tour} — check Lay odds columns.")
            continue

        results = aggregate_results(all_preds, event_summaries, tour, args.year)
        if results is None:
            print(f"No results to aggregate for {tour}."); continue

        print(f"\n  === RESULTS: {tour} {args.year} ===")
        print(f"  {'Market':<8}  {'AUC':>6}  {'AP':>6}  {'TSS':>6}  "
              f"{'Back_PnL':>10}  {'Back_ROI':>9}  {'Back_N':>7}")
        for _, row in results["summary"].iterrows():
            print(
                f"  {row['Market']:<8}  {row['AUC']:>6.4f}  {row['Avg_Precision']:>6.4f}  "
                f"{row['TSS']:>6.4f}  £{row['Back_PnL']:>9.2f}  "
                f"{row['Back_ROI']:>8.1%}  {row['Back_NBets']:>7}"
            )

        back_grid_df = lay_grid_df = None
        if not args.no_grid:
            back_grid_df = wfb.run_back_grid_search(results["all_predictions"])
            lay_grid_df  = wfb.run_lay_grid_search(results["all_predictions"])

        export_results(results, tour, args.year, back_grid_df, lay_grid_df)


if __name__ == "__main__":
    main()