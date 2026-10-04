"""Each-way screen: is there a back edge in extra-place concessions?

Bookmaker track, not the exchange: Betfair has no each-way market, so this runs
on its own venue, bankroll and commission (none).

`Places` and `Terms` vary player-by-player WITHIN an event, because Win_odds is a
best-price-shopped quote and the each-way terms belong to whichever book was
topping the market. Favourites attract the extra-place concessions. That variation
is the only thing here a model could exploit, so the screen sweeps it directly.

EW_Profit is already a settled, dead-heat-aware result for a £5 win + £5 place
bet, so no model and no re-settlement is needed to answer the question.

Run from the repo root:  python -m analysis.ew_screen
"""
import numpy as np
import pandas as pd

from golfmodel.backtest import _grid_metrics
from golfmodel.config import BACKTESTS_DIR, RAW_DIR, TOURS
from golfmodel.data import load_raw

EW_STAKE = 10.0          # £5 win + £5 place
PLACES_GRID = [5, 6, 7, 8, 10, 12]
EXTRA_GRID  = [0, 1, 2, 3]          # places above the event's own median
ODDS_GRID   = [10, 20, 50, 100, 200, 500]
RATING_GRID = [55, 60, 65, 70, 75]
TOPK_GRID   = [5, 10, 20, 30]


def place_odds(win_odds, terms):
    return (win_odds - 1.0) / terms + 1.0


def settle_ew(df: pd.DataFrame) -> pd.Series:
    """Rebuild EW_Profit from first principles, dead heats included."""
    posn   = pd.to_numeric(df["posn"], errors="coerce")
    places = pd.to_numeric(df["Places"], errors="coerce")
    terms  = pd.to_numeric(df["Terms"], errors="coerce")
    win    = pd.to_numeric(df["Win_odds"], errors="coerce")
    n_tied = df.groupby(["eventID", "posn"])["posn"].transform("size")

    f_win   = (((1 - posn + 1) / n_tied).clip(0, 1))
    f_place = (((places - posn + 1) / n_tied).clip(0, 1))
    return (5 * f_win * win - 5) + (5 * f_place * place_odds(win, terms) - 5)


def prepare(tour: str) -> pd.DataFrame:
    df = load_raw(RAW_DIR / f"{tour}.xlsx")
    df = df[df["posn"].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Places", "Terms", "Win_odds", "EW_Profit"])

    df["EventID"]   = df["eventID"]
    df["Test_Year"] = df["Date"].dt.year
    df["_Lay_Odds"] = df["Win_odds"]                  # the price actually bet at
    df["Actual"]    = (df["posn"] <= df["Places"]).astype(int)
    df["_EW_PnL"]   = df["EW_Profit"]
    df["_Zero"]     = 0.0
    # The concession: places offered ABOVE what the rest of this field is getting.
    df["Extra_Places"] = df["Places"] - df.groupby("EventID")["Places"].transform("median")
    return df


def _sweeps(base: pd.DataFrame):
    yield "All", "-", base
    for k in PLACES_GRID:
        yield "Places >=", k, base[base["Places"] >= k]
    for d in EXTRA_GRID:
        yield "Extra places >=", d, base[base["Extra_Places"] >= d]
    for t in (4, 5):
        yield "Terms ==", t, base[base["Terms"] == t]
    for o in ODDS_GRID:
        yield "Odds <", o, base[base["Win_odds"] < o]
    for o in ODDS_GRID:
        yield "Odds >=", o, base[base["Win_odds"] >= o]
    if "rating" in base.columns:
        for r in RATING_GRID:
            yield "Rating >=", r, base[base["rating"] >= r]
    # Ties are broken on price, NOT row order: 53.6% of rows sit at
    # Extra_Places == 0 and the file is stored in finishing order, so
    # rank(method="first") would select the best finishers outright.
    order = base.sort_values(["Extra_Places", "Win_odds"], ascending=[False, True],
                             kind="mergesort")
    rank = (order.groupby("EventID").cumcount() + 1).reindex(base.index)
    for k in TOPK_GRID:
        yield "Top K extra", k, base[rank <= k]


def screen(tour: str) -> pd.DataFrame:
    df = prepare(tour)
    rows = []
    for ftype, fval, band in _sweeps(df):
        if len(band) < 50:
            continue
        rows.append({
            "Filter_Type": ftype, "Filter_Value": fval,
            # Bookmaker bets pay no exchange commission.
            **_grid_metrics(band, "_EW_PnL", len(band) * EW_STAKE,
                            "_Zero", "_Zero", commission=0.0),
        })
    return pd.DataFrame(rows).drop(columns=["Kelly_PnL", "Kelly_ROI_%"])


def _self_check(tour: str) -> None:
    df = prepare(tour)
    rebuilt = settle_ew(df)
    ok = np.isclose(rebuilt, df["EW_Profit"], atol=0.01)
    rate = ok.mean()
    assert rate > 0.995, f"{tour}: EW settlement only reproduces {rate:.4f} of rows"
    print(f"  self-check OK ({tour}: settlement formula reproduces {rate*100:.2f}% "
          f"of {len(df):,} rows; {(~ok).sum()} known data errors)")


if __name__ == "__main__":
    tours = [t for t in TOURS if (RAW_DIR / f"{t}.xlsx").exists()]
    frames = {}
    for tour in tours:
        _self_check(tour)
        frames[tour] = screen(tour)

    out = BACKTESTS_DIR / "EW_Screen.xlsx"
    with pd.ExcelWriter(out) as xl:
        for tour, grid in frames.items():
            grid.to_excel(xl, sheet_name=tour, index=False)
    for tour, g in frames.items():
        print(f"\n--- {tour}: best ROI (n>=200) ---")
        print(g[g.N_Bets >= 200].nlargest(6, "ROI_%")[
            ["Filter_Type", "Filter_Value", "N_Bets", "Total_PnL",
             "ROI_%", "Sharpe", "Years_Pos"]].to_string(index=False))
    print(f"\nWrote {out}")
