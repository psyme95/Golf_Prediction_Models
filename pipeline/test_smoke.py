"""Smoke test: dead-heat logic, lay returns, and a tiny end-to-end train+backtest.

Run:  python test_smoke.py   (from pipeline/)
Uses synthetic data — no input files required. Takes ~1 minute.
"""

import numpy as np
import pandas as pd

import golfmodel.modeling as modeling
from golfmodel.backtest import (
    aggregate_results,
    apply_strategies,
    back_summary,
    backtest_events,
    dead_heat_info,
    export_results,
    run_back_grid,
    run_lay_grid,
    _kelly_back_stake,
    _kelly_lay_liability,
    _lay_return,
)
from golfmodel.config import (
    BACK_STAKE,
    COMMISSION,
    KELLY_BANKROLL,
    KELLY_FRACTION,
    KELLY_MAX_RISK,
)
from golfmodel.data import add_targets


# ===== dead_heat_info =====

def _event(posns):
    return pd.DataFrame({"playerID": range(len(posns)), "posn": posns})

# 3 players tied at 4 occupy places 4-6: two places (4th,5th) for three players
info = dead_heat_info(_event([1, 2, 3, 4, 4, 4, 7, 8]), cut=5)
assert info is not None and info[1] == 2 and info[2] == 3, info

# 2 players tied at 4 occupy 4-5: enough places, no dead heat
assert dead_heat_info(_event([1, 2, 3, 4, 4, 6]), cut=5) is None

# tie entirely below the cut boundary: no dead heat
assert dead_heat_info(_event([1, 1, 3, 4, 5, 6]), cut=5) is None

# 3 tied at exactly the cut with 1 place left → RF 1/3 case
info = dead_heat_info(_event([1, 2, 3, 4, 5, 5, 5, 8]), cut=5)
assert info is not None and info[1] == 1 and info[2] == 3, info

# no qualifiers at all
assert dead_heat_info(_event([6, 7, 8]), cut=5) is None

# solo player at cut with zero places (dirty data) must NOT trigger
assert dead_heat_info(_event([1, 1, 1, 1, 1, 5]), cut=5) is None
print("dead_heat_info: OK")


# ===== _lay_return =====

lay = np.array([4.0, 4.0])
liab = np.array([300.0, 300.0])
ones = np.ones(2)
r = _lay_return(lay, np.array([1.0, 0.0]), ones, ones, liab)
assert np.isclose(r[0], -300.0), r      # clean win for backer → layer pays liability
assert np.isclose(r[1], 100.0), r       # loss for backer → layer wins stake = liab/(odds-1)

# dead heat 1-of-3: layer pays a third of the effective payout
r_dh = _lay_return(np.array([4.0]), np.array([1.0]), np.array([1.0]), np.array([3.0]),
                   np.array([300.0]))
assert np.isclose(r_dh[0], ((1 - 4.0 / 3.0) / 3.0) * 300.0), r_dh
print("_lay_return: OK")


# ===== commission + Kelly via apply_strategies =====

event = pd.DataFrame({
    "playerID": [0, 1, 2], "posn": [1.0, 2.0, 3.0], "win": [1, 0, 0],
    "Lay_odds": [4.0, 6.0, 50.0],
    "Normalised_Model_Odds": [2.0, 10.0, 40.0],   # p0: back; p1: lay; p2: back
    "Probability": [0.5, 0.1, 0.02],
    "rating": [70.0, 65.0, 60.0],
})
out = apply_strategies(event.copy(), event, "Winner", "Lay_odds", "win")

# Per-row P&L is GROSS — commission is a per-market charge, applied on netting
assert bool(out.loc[0, "Back_Bet"])
assert np.isclose(out.loc[0, "Back_PnL"], BACK_STAKE * 3.0)
# Player 2 backs at 50 and loses: full stake lost
assert bool(out.loc[2, "Back_Bet"]) and np.isclose(out.loc[2, "Back_PnL"], -BACK_STAKE)
# Player 1 layed at 6.0, did not win: layer wins stake (=1000/5), gross
assert bool(out.loc[1, "Lay_Bet"])
assert np.isclose(out.loc[1, "Lay_PnL_FixedLiab"], 1000.0 / 5.0)
# Edge ratios
assert np.allclose(out["Edge_Raw"], [0.5 * 4, 0.1 * 6, 0.02 * 50])
assert np.allclose(out["Edge_Norm"], [4 / 2.0, 6 / 10.0, 50 / 40.0])
print("gross P&L + edge columns: OK")


# ===== market-level commission netting =====
# Betfair charges commission on NET market winnings, not per bet.
# 3 bets in one market: +5, +5, -5 → net 5 → 3% → 4.85  (not 4.70)
one_market = pd.DataFrame({
    "EventID": [1, 1, 1], "Back_Bet": [True] * 3,
    "Back_PnL": [5.0, 5.0, -5.0], "Back_PnL_Kelly": [0.0] * 3,
    "Back_Kelly_Stake": [0.0] * 3, "win": [1, 1, 0],
})
assert np.isclose(back_summary(one_market, "win")["Back_PnL"], 4.85), \
    back_summary(one_market, "win")

# A market that nets negative pays no commission at all
losing_market = one_market.copy()
losing_market["Back_PnL"] = [5.0, -5.0, -5.0]
assert np.isclose(back_summary(losing_market, "win")["Back_PnL"], -5.0)

# Two separate markets are charged independently (+5 net → 4.85; -5 → -5)
two_markets = pd.concat([one_market, losing_market.assign(EventID=2)], ignore_index=True)
assert np.isclose(back_summary(two_markets, "win")["Back_PnL"], 4.85 - 5.0)
print("per-market commission netting: OK")

# Kelly back stake: p=0.5, odds 4.0 → b=3*0.97, f*=(p*b-q)/b
b = 3.0 * (1 - COMMISSION)
f_star = (0.5 * b - 0.5) / b
expected_stake = min(KELLY_BANKROLL * KELLY_FRACTION * f_star,
                     KELLY_BANKROLL * KELLY_MAX_RISK)
assert np.isclose(_kelly_back_stake(np.array([0.5]), np.array([4.0]))[0], expected_stake)
assert np.isclose(out.loc[0, "Back_PnL_Kelly"], expected_stake * 3.0)
# Negative-edge Kelly → zero stake
assert _kelly_back_stake(np.array([0.1]), np.array([2.0]))[0] == 0.0
# Kelly lay liability: p=0.1, odds 6.0 → b'=(1-c)/5, f*=((1-p)b'-p)/b'
b2 = (1 - COMMISSION) / 5.0
f2 = (0.9 * b2 - 0.1) / b2
expected_liab = min(KELLY_BANKROLL * KELLY_FRACTION * f2, KELLY_BANKROLL * KELLY_MAX_RISK)
assert np.isclose(_kelly_lay_liability(np.array([0.1]), np.array([6.0]))[0], expected_liab)
assert np.isclose(out.loc[1, "Lay_PnL_Kelly"], expected_liab / 5.0)
print("kelly staking: OK")


# ===== end-to-end: train one market, backtest one year =====

rng = np.random.default_rng(0)
N_EVENTS, FIELD = 40, 60
rows = []
for e in range(N_EVENTS):
    skill = rng.normal(0, 1, FIELD)
    posn = (-skill + rng.normal(0, 1.2, FIELD)).argsort().argsort() + 1
    year = 2023 + (e >= 30)                     # 30 train events, 10 test events
    for p in range(FIELD):
        rows.append({
            "eventID": e, "playerID": p,
            "Date": pd.Timestamp(f"{year}-06-01") + pd.Timedelta(days=e),
            "surname": f"P{p}", "firstname": "X",
            "posn": float(posn[p]),
            "rating": 60 + 5 * skill[p] + rng.normal(0, 1),
            "current": skill[p] + rng.normal(0, 0.5),
            "field": rng.normal(0, 1),
            "Win_odds": float(np.clip(20 - 10 * skill[p], 1.5, 500)),
            "Lay_odds": float(np.clip(21 - 10 * skill[p], 1.5, 500)),
        })
df = add_targets(pd.DataFrame(rows))

modeling.N_CV_REPEATS = 1                       # keep the smoke test fast
import golfmodel.config as cfg_mod

train_df = df[df["Date"].dt.year == 2023]
test_df  = df[df["Date"].dt.year == 2024].copy()

from pathlib import Path
import tempfile
tmp = Path(tempfile.mkdtemp())
pkg = modeling.train_market("Winner", train_df, "TEST", use_meta_odds=True,
                            n_trials=2, models_dir=tmp)
assert pkg is not None and len(pkg["models"]) == 5
assert 0 < pkg["metrics"]["lgbm"]["roc_auc"] <= 1
# Residual meta: market coefficient learned and finite, logit form recorded
assert pkg["meta_odds_form"] == "logit"
assert np.isfinite(pkg["meta_market_coef"])

preds, summaries = backtest_events({"Winner": pkg}, test_df, "TEST", 2024)
assert preds and summaries
results = aggregate_results(preds, summaries, "TEST")
assert (results["all_predictions"]["Probability"] <= 1).all()
assert results["summary"]["AUC"].iloc[0] > 0.5   # skill signal must be learnable
assert "Back_PnL_Kelly" in results["summary"].columns
assert "Lay_ROI_Kelly" in results["summary"].columns

back_grid = run_back_grid(results["all_predictions"])
lay_grid  = run_lay_grid(results["all_predictions"])
# Grid structure: both edge bases, all sweep dimensions, consistency columns
assert set(back_grid["Edge_Basis"]) == {"Raw", "Norm"}
assert {"All", "Edge >=", "Odds >=", "Odds <", "Rating >=", "Rating <"} <= set(back_grid["Filter_Type"])
assert {"Years_Pos", "Worst_Year_PnL", "Kelly_PnL"} <= set(back_grid.columns)
assert {"FL_Years_Pos", "FS_Years_Pos"} <= set(lay_grid.columns)
out = export_results(results, tmp / "smoke_backtest.xlsx", back_grid, lay_grid)
assert out.exists()
print("end-to-end train + backtest + export (residual meta, grids): OK")

print("\nALL SMOKE TESTS PASSED")
