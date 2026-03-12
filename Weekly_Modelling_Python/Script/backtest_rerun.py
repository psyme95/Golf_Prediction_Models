"""
Backtest Rerun Script

Re-runs betting strategy analysis and grid searches on saved All_Predictions data
from a completed walk-forward backtest.  No model training or new predictions are
generated — only the P&L / summary / grid layers are recomputed.

Useful for comparing model variants that produced different All_Predictions files
without waiting for a full retrain.

Input:   {tour}_WalkForward_Backtest.xlsx  (must contain an All_Predictions sheet)
Output:  {tour}_WalkForward_Rerun.xlsx     (same directory as input by default)

Run:
    python backtest_rerun.py --tour PGA
    python backtest_rerun.py --tour Euro
    python backtest_rerun.py                              # both tours, auto-discovered
    python backtest_rerun.py --file path/to/file.xlsx
    python backtest_rerun.py --tour PGA --out my_out.xlsx
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

# Reuse all analysis helpers from the main backtest script
import walk_forward_backtest as wfb
from walk_forward_backtest import (
    BETTING_MARKETS,
    LAY_ODDS_COLS,
    WF_RESULTS_DIR,
    apply_back_strategy,
    apply_lay_strategy,
    back_summary,
    calibration_bins,
    compute_discrimination,
    lay_summary,
    run_back_grid_search,
    run_lay_grid_search,
    _write_grid_sheet,
)

warnings.filterwarnings("ignore")


# ===== LOAD =====

def load_predictions(path: Path) -> pd.DataFrame:
    """Load the All_Predictions sheet from an existing backtest Excel file."""
    print(f"  Loading: {path.name}")
    df = pd.read_excel(path, sheet_name="All_Predictions")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Normalised_Model_Odds", "Normalised_Probability",
                "Probability", "Model_Score", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for lay_col in LAY_ODDS_COLS.values():
        if lay_col in df.columns:
            df[lay_col] = pd.to_numeric(df[lay_col], errors="coerce")
    # Normalise event ID column name to match raw data convention used in
    # walk_forward_backtest.py (apply_back_strategy / compute_reduction_factors)
    if "EventID" in df.columns and "eventID" not in df.columns:
        df = df.rename(columns={"EventID": "eventID"})
    print(f"  Rows: {len(df):,}  |  Markets: {sorted(df['Market'].unique())}")
    print(f"  Years: {sorted(df['Test_Year'].unique())}")
    return df


# ===== STRATEGY =====

def reapply_strategies(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-apply back and lay betting strategies from scratch using the model odds
    and lay odds already present in the predictions DataFrame.

    Overwrites any existing Back_Bet / Lay_Bet / P&L columns so that strategy
    parameter changes in walk_forward_backtest.py are picked up automatically.
    """
    parts = []
    for market_name in BETTING_MARKETS:
        mdf = pred_df[pred_df["Market"] == market_name].copy()
        if len(mdf) == 0:
            continue

        lay_odds_col = LAY_ODDS_COLS.get(market_name)
        if lay_odds_col not in mdf.columns or mdf[lay_odds_col].isna().all():
            print(f"  {market_name}: no Lay odds — strategy not applied")
            parts.append(mdf)
            continue

        mdf = apply_back_strategy(mdf, lay_odds_col, "Actual", market_name)
        mdf = apply_lay_strategy(mdf, lay_odds_col, "Actual", market_name)
        parts.append(mdf)

    return pd.concat(parts, ignore_index=True) if parts else pred_df


# ===== AGGREGATE =====

def aggregate(pred_df: pd.DataFrame, tour_key: str) -> dict:
    """
    Rebuild summary, season_summary, event_results, and calibration DataFrames
    from a flat All_Predictions DataFrame.  Mirrors aggregate_results() in
    walk_forward_backtest.py but uses "Actual" as the universal target column.
    """
    # --- Event-level summaries ---
    event_rows = []
    for (t, test_year, event_id, market_name), grp in pred_df.groupby(
        ["Tour", "Test_Year", "eventID", "Market"]
    ):
        if len(grp) == 0:
            continue
        date = grp["Date"].iloc[0].strftime("%Y-%m-%d") if "Date" in grp.columns else ""
        bs = back_summary(grp, "Actual")
        ls = lay_summary(grp, "Actual", market_name)
        event_rows.append({
            "Tour":      t,
            "Test_Year": test_year,
            "EventID":   event_id,
            "Date":      date,
            "Market":    market_name,
            "FieldSize": len(grp),
            "Positives": int(grp["Actual"].sum()),
            **bs, **ls,
        })

    event_df = (
        pd.DataFrame(event_rows)
        .sort_values(["Market", "Test_Year", "Date", "EventID"])
        .reset_index(drop=True)
    )
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

    # --- Season summaries ---
    season_rows = []
    for test_year in sorted(pred_df["Test_Year"].unique()):
        for market_name in BETTING_MARKETS:
            mdf = pred_df[
                (pred_df["Test_Year"] == test_year) &
                (pred_df["Market"] == market_name)
            ]
            if len(mdf) == 0:
                continue
            y_true = mdf["Actual"].values
            y_prob = mdf["Normalised_Probability"].values
            disc = compute_discrimination(y_true, y_prob)
            season_rows.append({
                "Tour":       tour_key,
                "Test_Year":  test_year,
                "Market":     market_name,
                "N_Events":   int(mdf["eventID"].nunique()),
                "N_Players":  len(mdf),
                "Prevalence": round(float(y_true.mean()), 4),
                **disc,
                **back_summary(mdf, "Actual"),
                **lay_summary(mdf, "Actual", market_name),
            })

    # --- Overall summary + calibration ---
    summary_rows = []
    calib_sheets = {}
    for market_name in BETTING_MARKETS:
        mdf = pred_df[pred_df["Market"] == market_name]
        if len(mdf) == 0:
            continue
        y_true = mdf["Actual"].values
        y_prob = mdf["Normalised_Probability"].values
        disc = compute_discrimination(y_true, y_prob)
        summary_rows.append({
            "Tour":         tour_key,
            "Market":       market_name,
            "N_Test_Years": int(pred_df[pred_df["Market"] == market_name]["Test_Year"].nunique()),
            "N_Events":     int(mdf["eventID"].nunique()),
            "N_Players":    len(mdf),
            "Prevalence":   round(float(y_true.mean()), 4),
            **disc,
            **back_summary(mdf, "Actual"),
            **lay_summary(mdf, "Actual", market_name),
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

def export(results: dict, tour_key: str,
           back_grid_df: pd.DataFrame, lay_grid_df: pd.DataFrame,
           out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df   = results["all_predictions"].rename(columns={"eventID": "EventID"})
    id_cols   = [c for c in ["Test_Year", "Date", "EventID", "Market",
                              "playerID", "surname", "firstname", "posn", "rating"]
                 if c in pred_df.columns]
    lay_cols  = [c for c in LAY_ODDS_COLS.values() if c in pred_df.columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    bet_cols  = ["Back_Bet", "DeadHeat_RF", "Back_PnL",
                 "Lay_Bet", "Lay_PnL_FixedLiab", "Lay_PnL_FixedStake"]
    all_cols  = id_cols + lay_cols + pred_cols + bet_cols
    export_df = pred_df[[c for c in all_cols if c in pred_df.columns]]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(        writer, sheet_name="Summary",         index=False)
        results["season_summary"].to_excel( writer, sheet_name="Season_Summary",  index=False)
        results["event_results"].to_excel(  writer, sheet_name="Event_Results",   index=False)
        export_df.to_excel(                 writer, sheet_name="All_Predictions", index=False)
        for market_name, calib_df in results["calibration"].items():
            calib_df.to_excel(writer, sheet_name=f"Calib_{market_name}", index=False)

    wb = load_workbook(out_path)
    if back_grid_df is not None and len(back_grid_df) > 0:
        _write_grid_sheet(wb, back_grid_df,
                          sheet_name="Back_Strategy_Grid",
                          pnl_cols=["Total_PnL"],
                          roi_cols=["ROI_%"])
    if lay_grid_df is not None and len(lay_grid_df) > 0:
        _write_grid_sheet(wb, lay_grid_df,
                          sheet_name="Lay_Strategy_Grid",
                          pnl_cols=["FL_Total_PnL", "FS_Total_PnL"],
                          roi_cols=["FL_ROI_%", "FS_ROI_%"])
    wb.save(out_path)

    print(f"\n  Saved: {out_path}")
    return out_path


# ===== RUNNER =====

def _run(in_path: Path, tour_key: str, out_path: Path | None):
    print(f"\n{'='*60}")
    print(f"BACKTEST RERUN: {tour_key}")
    print(f"{'='*60}")

    if not in_path.exists():
        print(f"  File not found: {in_path}")
        return

    pred_df = load_predictions(in_path)

    print("\n  Re-applying strategies...")
    pred_df = reapply_strategies(pred_df)

    print("\n  Aggregating results...")
    results = aggregate(pred_df, tour_key)

    print(f"\n  === RESULTS ===")
    print(f"  {'Market':<8}  {'AUC':>6}  {'AP':>6}  {'TSS':>6}  {'Back_PnL':>10}  {'Back_ROI':>9}")
    for _, row in results["summary"].iterrows():
        print(
            f"  {row['Market']:<8}  {row['AUC']:>6.4f}  {row['Avg_Precision']:>6.4f}  "
            f"{row['TSS']:>6.4f}  £{row['Back_PnL']:>9.2f}  {row['Back_ROI']:>8.1%}"
        )

    grid_back_df = run_back_grid_search(pred_df)
    grid_lay_df  = run_lay_grid_search(pred_df)

    if out_path is None:
        out_path = in_path.parent / in_path.name.replace("_Backtest.xlsx", "_Rerun.xlsx")
        if out_path == in_path:   # file wasn't named *_Backtest.xlsx
            stem = in_path.stem
            out_path = in_path.parent / f"{stem}_Rerun.xlsx"

    export(results, tour_key, grid_back_df, grid_lay_df, out_path)


# ===== MAIN =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run backtest analysis from saved All_Predictions data"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tour", default=None,
        help="Tour key (PGA or Euro) — auto-finds the backtest Excel in WalkForward/Results/",
    )
    group.add_argument(
        "--file", default=None,
        help="Path to a specific backtest Excel file containing an All_Predictions sheet",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file path (default: same directory as input, *_Rerun.xlsx)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else None

    if args.file:
        in_path  = Path(args.file)
        tour_key = in_path.stem.split("_")[0]
        _run(in_path, tour_key, out_path)

    elif args.tour:
        in_path = WF_RESULTS_DIR / f"{args.tour}_WalkForward_Backtest.xlsx"
        _run(in_path, args.tour, out_path)

    else:
        # Auto-discover all backtest Excel files
        files = sorted(WF_RESULTS_DIR.glob("*_WalkForward_Backtest.xlsx"))
        if not files:
            print(f"No backtest Excel files found in {WF_RESULTS_DIR}")
            sys.exit(1)
        for f in files:
            tour = f.stem.split("_")[0]
            _run(f, tour, out_path)


if __name__ == "__main__":
    main()
