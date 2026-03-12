"""
Backtest Rerun Script

Re-applies betting strategies to a saved All_Predictions sheet and writes
a new Excel file with P&L summaries.

Run:
    python backtest_rerun.py --tour PGA
    python backtest_rerun.py --tour Euro
    python backtest_rerun.py --file path/to/file.xlsx
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

import walk_forward_backtest as wfb

warnings.filterwarnings("ignore")


def load_predictions(path: Path) -> pd.DataFrame:
    print(f"Loading: {path.name}")
    df = pd.read_excel(path, sheet_name="All_Predictions")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Normalised_Model_Odds", "Normalised_Probability",
                "Probability", "Model_Score", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in wfb.LAY_ODDS_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # walk_forward_backtest.py expects lowercase "eventID"
    if "EventID" in df.columns:
        df = df.rename(columns={"EventID": "eventID"})

    print(f"  {len(df):,} rows  |  markets: {sorted(df['Market'].unique())}")
    print(f"  years: {sorted(df['Test_Year'].unique())}")
    return df


def apply_strategies(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for market in wfb.BETTING_MARKETS:
        mdf = df[df["Market"] == market].copy()
        if mdf.empty:
            continue
        lay_col = wfb.LAY_ODDS_COLS.get(market)
        if lay_col not in mdf.columns or mdf[lay_col].isna().all():
            print(f"  {market}: no lay odds, skipping")
            parts.append(mdf)
            continue
        mdf = wfb.apply_back_strategy(mdf, lay_col, "Actual", market)
        mdf = wfb.apply_lay_strategy(mdf, lay_col, "Actual", market)
        parts.append(mdf)
    return pd.concat(parts, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market in wfb.BETTING_MARKETS:
        mdf = df[df["Market"] == market]
        if mdf.empty:
            continue
        row = {"Market": market, **wfb.back_summary(mdf, "Actual"),
               **wfb.lay_summary(mdf, "Actual", market)}
        rows.append(row)
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    # Restore capital-E EventID for the output file
    out_df = df.rename(columns={"eventID": "EventID"})
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        out_df.to_excel(writer, sheet_name="All_Predictions", index=False)
    print(f"\nSaved: {out_path}")


def run(in_path: Path, out_path: Path | None = None):
    df      = load_predictions(in_path)
    df      = apply_strategies(df)
    summary = summarise(df)

    print(f"\n{'Market':<8}  {'Back_PnL':>10}  {'Back_ROI':>9}  "
          f"{'Back_NBets':>10}  {'Lay_PnL_FL':>10}  {'Lay_PnL_FS':>10}")
    for _, r in summary.iterrows():
        print(f"  {r['Market']:<8}  £{r['Back_PnL']:>9.2f}  {r['Back_ROI']:>8.1%}  "
              f"{r['Back_NBets']:>10}  £{r['Lay_PnL_FixedLiab']:>9.2f}  "
              f"£{r['Lay_PnL_FixedStake']:>9.2f}")

    if out_path is None:
        stem     = in_path.stem.replace("_Backtest", "")
        out_path = in_path.parent / f"{stem}_Rerun.xlsx"

    save(df, summary, out_path)


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tour", choices=["PGA", "Euro"])
    group.add_argument("--file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    in_path  = Path(args.file) if args.file else (
        wfb.WF_RESULTS_DIR / f"{args.tour}_WalkForward_Backtest.xlsx"
    )
    out_path = Path(args.out) if args.out else None
    run(in_path, out_path)


if __name__ == "__main__":
    main()
