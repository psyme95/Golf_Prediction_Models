"""Back-betting screen: does any back strategy survive a realistic back price?

Back bets were originally priced at the Betfair lay quote, which is better than
any price that can actually be backed at. This re-runs the back strategy grid
from saved walk-forward predictions at back_price = lay * (1 - spread) for a
range of spreads, without retraining. spread=0 reproduces the original numbers.

Run from the repo root:  python -m analysis.back_screen [tag]
"""
import sys

import numpy as np
import pandas as pd

from golfmodel.backtest import _kelly_back_stake, _with_lay_odds, run_back_grid
from golfmodel.config import (BACK_SPREAD, BACK_STAKE, BACKTESTS_DIR,
                              LAY_ODDS_COLS, TOURS)

SPREADS = [0.00, 0.02, 0.05, 0.10]

# Betfair price ladder: (exclusive upper bound, tick size). One tick below the
# lay quote is the tightest a back price can physically be, a hard floor on
# BACK_SPREAD, never an estimate of it. Thin golf place markets sit wider.
LADDER = [(2, .01), (3, .02), (4, .05), (6, .1), (10, .2),
          (20, .5), (30, 1), (50, 2), (100, 5), (1000, 10)]


def tick_size(prices: np.ndarray) -> np.ndarray:
    bounds = np.array([b for b, _ in LADDER])
    ticks  = np.array([t for _, t in LADDER])
    return ticks[np.clip(np.searchsorted(bounds, prices, side="right"), 0, len(ticks) - 1)]


def tick_floor(preds: pd.DataFrame) -> pd.DataFrame:
    """One-tick spread as a fraction of price, per market, the floor on BACK_SPREAD."""
    rows = []
    for market_name, lay_col in LAY_ODDS_COLS.items():
        p = pd.to_numeric(preds.loc[preds["Market"] == market_name, lay_col],
                          errors="coerce").dropna()
        p = p[p > 1.01]
        if p.empty:
            continue
        s = tick_size(p.to_numpy(dtype=float)) / p.to_numpy(dtype=float)
        rows.append({"Market": market_name, "Median_Lay": round(float(p.median()), 2),
                     "OneTick_%": round(float(np.median(s)) * 100, 2),
                     "OneTick_p75_%": round(float(np.percentile(s, 75)) * 100, 2),
                     "TwoTick_%": round(float(np.median(s)) * 200, 2)})
    return pd.DataFrame(rows)


def reprice(preds: pd.DataFrame, spread: float) -> pd.DataFrame:
    """Recompute every column run_back_grid consumes at a discounted back price."""
    df = _with_lay_odds(preds)
    back = df["_Lay_Odds"].to_numpy(dtype=float) * (1 - spread)
    df["_Back_Odds"] = back           # grid odds bands filter on the price bet at
    p      = df["Probability"].to_numpy(dtype=float)
    rf     = df["DeadHeat_RF"].to_numpy(dtype=float)
    actual = df["Actual"].to_numpy(dtype=float)

    df["Edge_Raw_Back"]  = p * back
    df["Edge_Norm_Back"] = back / np.clip(df["Normalised_Model_Odds"], 1e-8, None)
    df["_Back_PnL_Pot"] = np.where(actual == 1, BACK_STAKE * (rf * back - 1), -BACK_STAKE)
    kb = _kelly_back_stake(p, back)
    df["_Back_Kelly_Stake"]   = kb
    df["_Back_Kelly_PnL_Pot"] = np.where(actual == 1, kb * (rf * back - 1), -kb)
    return df


def wf_path(tour: str, tag: str | None = None):
    name = (f"{tour}_WalkForward_Backtest_{tag}.xlsx" if tag
            else f"{tour}_WalkForward_Backtest.xlsx")
    return BACKTESTS_DIR / name


def screen(tour: str, tag: str | None = None) -> pd.DataFrame:
    preds = pd.read_excel(wf_path(tour, tag), sheet_name="All_Predictions")
    out = []
    for s in SPREADS:
        grid = run_back_grid(reprice(preds, s))
        grid.insert(0, "Spread", s)
        out.append(grid)
    return pd.concat(out, ignore_index=True)


def _self_check(tour: str, tag: str | None = None) -> None:
    """The recomputation must reproduce the workbook's own Back_PnL at whichever
    spread that workbook was generated with.

    Pre-BACK_SPREAD workbooks were priced at the raw lay quote (spread 0); ones
    generated since are priced at lay*(1-BACK_SPREAD). Rather than hard-coding
    either, find the spread that reproduces the file and fail if none does,
    which still catches any real break in the repricing.
    """
    preds = pd.read_excel(wf_path(tour, tag), sheet_name="All_Predictions")
    for s in sorted({0.0, BACK_SPREAD, *SPREADS}):
        bet = reprice(preds, s)
        bet = bet[bet["Back_Bet"].astype(bool)]
        if bet.empty:
            continue
        diff = (bet["_Back_PnL_Pot"] - bet["Back_PnL"]).abs().max()
        if diff < 0.02:
            print(f"  self-check OK ({tour}, {len(bet):,} back bets priced at "
                  f"spread={s:.2f}, max diff {diff:.4f})")
            return
    raise AssertionError(f"{tour}: no spread reproduces the workbook's Back_PnL")


def _check_ticks() -> None:
    got = tick_size(np.array([1.5, 2.5, 3.5, 5.0, 8.0, 15.0, 25.0, 40.0, 75.0, 500.0]))
    assert list(got) == [.01, .02, .05, .1, .2, .5, 1, 2, 5, 10], got
    assert tick_size(np.array([2.0]))[0] == .02 and tick_size(np.array([1.99]))[0] == .01


def survivors(grid: pd.DataFrame, spread: float, frac_years: float = 0.8) -> pd.DataFrame:
    """Decision rule: positive P&L, >=200 bets, >=80% of test years profitable.

    The year requirement is a FRACTION, not a count: a full 5-window run needs
    4/5, but a run truncated to 2 windows needs 2/2. Hard-coding ">=4" against a
    2-year run makes the rule unsatisfiable and reports a meaningless zero.
    """
    d = grid[grid["Spread"] == spread].copy()
    yp = d["Years_Pos"].map(lambda v: int(str(v).split("/")[0]))
    yn = d["Years_Pos"].map(lambda v: int(str(v).split("/")[1]))
    return d[(d["Total_PnL"] > 0) & (d["N_Bets"] >= 200) & (yp >= np.ceil(frac_years * yn))]


if __name__ == "__main__":
    _check_ticks()
    tag   = sys.argv[1] if len(sys.argv) > 1 else None
    tours = [t for t in TOURS if wf_path(t, tag).exists()]
    if not tours:
        raise SystemExit(f"no walk-forward workbooks found for tag={tag!r}")

    frames, floors = {}, {}
    for tour in tours:
        _self_check(tour, tag)
        frames[tour] = screen(tour, tag)
        preds = pd.read_excel(wf_path(tour, tag), sheet_name="All_Predictions")
        floors[tour] = tick_floor(preds).assign(Tour=tour)
        print(f"  {tour}: {len(frames[tour]):,} grid rows across {len(SPREADS)} spreads")

    out = BACKTESTS_DIR / (f"Back_Screen_{tag}.xlsx" if tag else "Back_Screen.xlsx")
    with pd.ExcelWriter(out) as xl:
        for tour, grid in frames.items():
            grid.to_excel(xl, sheet_name=tour, index=False)
        pd.concat(floors.values(), ignore_index=True).to_excel(
            xl, sheet_name="Tick_Floor", index=False)

    key = ["Market", "Edge_Basis", "Filter_Type", "Filter_Value"]
    print()
    print("  Strategies passing (positive, n>=200, >=4/5 years):")
    for sp in SPREADS:
        per  = {t: survivors(g, sp) for t, g in frames.items()}
        sets = [set(map(tuple, d[key].values)) for d in per.values()]
        both = set.intersection(*sets) if len(sets) > 1 else set()
        counts = "   ".join(f"{t} {len(d):3d}" for t, d in per.items())
        print(f"    spread={sp:.2f}:  {counts}   both tours {len(both)}")
        for b in sorted(both):
            print(f"        {b}")
    print(f"\nWrote {out}")
