"""Smoke test: dead-heat logic, lay returns, and a tiny end-to-end train+backtest.

Run from the repo root:  python -m tests.test_smoke
Uses synthetic data, no input files required. Takes ~1 minute.
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
    BACK_SPREAD,
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

# Per-row P&L is GROSS, commission is a per-market charge, applied on netting.
# Backs are priced at a haircut off the lay quote (BACK_SPREAD); lays are not.
bp0, bp2 = 4.0 * (1 - BACK_SPREAD), 50.0 * (1 - BACK_SPREAD)
assert bool(out.loc[0, "Back_Bet"])
assert np.isclose(out.loc[0, "Back_PnL"], BACK_STAKE * (bp0 - 1))
# Player 2 backs at 50*(1-spread) and loses: full stake lost
assert bool(out.loc[2, "Back_Bet"]) and np.isclose(out.loc[2, "Back_PnL"], -BACK_STAKE)
# Player 1 layed at 6.0, did not win: layer wins stake (=1000/5), gross
assert bool(out.loc[1, "Lay_Bet"])
assert np.isclose(out.loc[1, "Lay_PnL_FixedLiab"], 1000.0 / 5.0)
# Lay edge ratios stay on the lay quote; back edges use the haircut price
assert np.allclose(out["Edge_Raw"], [0.5 * 4, 0.1 * 6, 0.02 * 50])
assert np.allclose(out["Edge_Norm"], [4 / 2.0, 6 / 10.0, 50 / 40.0])
assert np.allclose(out["Edge_Raw_Back"], [0.5 * bp0, 0.1 * 6 * (1 - BACK_SPREAD), 0.02 * bp2])
assert np.allclose(out["Edge_Norm_Back"], [bp0 / 2.0, 6 * (1 - BACK_SPREAD) / 10.0, bp2 / 40.0])
# The haircut must never turn a lay into a back: back price < lay price always
assert (out["_Back_Odds"] < out["Lay_odds"]).all()
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
stake_bp0 = _kelly_back_stake(np.array([0.5]), np.array([bp0]))[0]
assert np.isclose(out.loc[0, "Back_PnL_Kelly"], stake_bp0 * (bp0 - 1))
# Negative-edge Kelly → zero stake
assert _kelly_back_stake(np.array([0.1]), np.array([2.0]))[0] == 0.0
# Kelly lay liability: p=0.1, odds 6.0 → b'=(1-c)/5, f*=((1-p)b'-p)/b'
b2 = (1 - COMMISSION) / 5.0
f2 = (0.9 * b2 - 0.1) / b2
expected_liab = min(KELLY_BANKROLL * KELLY_FRACTION * f2, KELLY_BANKROLL * KELLY_MAX_RISK)
assert np.isclose(_kelly_lay_liability(np.array([0.1]), np.array([6.0]))[0], expected_liab)
assert np.isclose(out.loc[1, "Lay_PnL_Kelly"], expected_liab / 5.0)
print("kelly staking: OK")


# ===== normalisation when players are dropped =====
from golfmodel.modeling import market_share, normalise

# Implied 0.5/0.25/0.25 plus one unpriced player. The third player is dropped
# for missing features, so the other two hold 75% of the market between them.
_odds = pd.Series([2.0, 4.0, 4.0, np.nan])
_kept = pd.Series([True, True, False, False])
_share = market_share(_odds, _kept)
assert abs(_share - 0.75) < 1e-12
assert abs(normalise(np.array([0.4, 0.2]), 1, _share).sum() - 0.75) < 1e-12
assert abs(normalise(np.array([0.4, 0.2]), 1).sum() - 1.0) < 1e-12
print("normalisation with dropped players: OK")


# ===== training windows =====
from golfmodel.backtest import get_windows
from golfmodel.data import training_years_slice

_years = pd.DataFrame({"Date": pd.to_datetime([f"{y}-06-01" for y in range(2020, 2027)])})
# Both modes start testing in 2022 (two years in); only the training start differs.
assert get_windows(_years) == [(y - 2, y - 1, y) for y in range(2022, 2027)]
assert get_windows(_years, window="expanding") == [(2020, y - 1, y) for y in range(2022, 2027)]
assert set(training_years_slice(_years)["Date"].dt.year) == {2024, 2025}
assert set(training_years_slice(_years, "expanding")["Date"].dt.year) == set(range(2020, 2026))
print("rolling and expanding windows: OK")


# ===== dead-heat labels: weighted duplication == fractional cross-entropy =====
# This is the assertion the entire label change rests on. All five base models
# are classifiers that reject a continuous target; representing a fractional
# label f as (y=1, w=f) + (y=0, w=1-f) is exactly equivalent to cross-entropy
# against f, which is what lets the existing stack survive unchanged.
from golfmodel.modeling import expand_fractional
from golfmodel.data import add_frac_targets
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

_rng = np.random.default_rng(0)
_X = _rng.normal(size=(1500, 3))
_f = 1 / (1 + np.exp(-(_X @ [1.0, -0.5, 0.3])))          # fractional targets in (0,1)
_g = np.arange(1500) // 10

_X2, _y2, _w2, _g2 = expand_fractional(_X, _f, _g)
_lr = LogisticRegression(max_iter=5000).fit(_X2, _y2, sample_weight=_w2)

_Xc = np.c_[_X, np.ones(len(_X))]
def _nll(th):
    p = 1 / (1 + np.exp(-(_Xc @ th)))
    return -(_f * np.log(p + 1e-12) + (1 - _f) * np.log(1 - p + 1e-12)).sum()
_truth = minimize(_nll, np.zeros(4)).x
_fitted = np.r_[_lr.coef_[0], _lr.intercept_]
assert np.abs(_fitted - _truth).max() < 0.01, (_fitted, _truth)

# Round-trip: total weight is conserved, and both copies keep their group so
# grouped CV can never split one row across train and validation.
_Xr, _yr, _wr, _gr = expand_fractional(np.arange(8.).reshape(4, 2),
                                       np.array([0.0, 1.0, 0.5, 1/3]),
                                       np.array([10, 11, 12, 13]))
assert len(_yr) == 6, len(_yr)                    # 4 rows + 2 genuinely fractional
assert np.isclose(_wr.sum(), 4.0)
assert sorted(_gr[_gr == 12].tolist()) == [12, 12]
assert (_wr > 0).all()                            # zero-weight copies dropped
print("fractional labels: weighted duplication == cross-entropy: OK")


# ===== frac label recovery from settled profit =====
# frac = (profit/stake + 1)/odds. A 6-way tie for 6th in a top-7 market leaves
# 2 places for 6 players -> 1/3. Settlement errors are nulled, not clipped.
_dh = pd.DataFrame({
    "eventID": [1, 1, 1, 2],
    "posn":    [6.0, 6.0, 6.0, 1.0],
    "Top10_odds":   [4.0, 4.0, 4.0, 3.0],
    "Top10_Profit": [np.nan, np.nan, np.nan, 20.0],
})
# 3-way tie for 6th in top-10: 5 places left for 3 players -> frac 1.0 (capped)
_dh.loc[0:2, "Top10_Profit"] = 10.0 * (1.0 * 4.0 - 1)
_out = add_frac_targets(_dh)
assert np.allclose(_out["frac_top_10"], [1.0, 1.0, 1.0, 1.0]), _out["frac_top_10"].tolist()

# A top-5 finish settled as a total loss is a settlement error -> NaN, not 0.0
_bad = pd.DataFrame({"eventID": [1], "posn": [3.0],
                     "Top5_odds": [1001.0], "Top5_Profit": [-10.0]})
assert pd.isna(add_frac_targets(_bad)["frac_top_5"].iloc[0])
print("frac label recovery + settlement-error guard: OK")


# ===== end-to-end: train one market, backtest one year =====

rng = np.random.default_rng(0)
N_EVENTS, FIELD = 40, 60
rows = []
for e in range(N_EVENTS):
    skill = rng.normal(0, 1, FIELD)
    posn = (-skill + rng.normal(0, 1.2, FIELD)).argsort().argsort() + 1
    year = 2023 + (e >= 30)                     # 30 train events, 10 test events
    # Noisy market so the model and the market disagree both ways.
    odds = np.clip(20 - 10 * skill + rng.normal(0, 6, FIELD), 1.5, 500)
    for p in range(FIELD):
        rows.append({
            "eventID": e, "playerID": p,
            "Date": pd.Timestamp(f"{year}-06-01") + pd.Timedelta(days=e),
            "surname": f"P{p}", "firstname": "X",
            "posn": float(posn[p]),
            "rating": 60 + 5 * skill[p] + rng.normal(0, 1),
            "current": skill[p] + rng.normal(0, 0.5),
            "field": rng.normal(0, 1),
            "Win_odds": float(odds[p]),
            "Lay_odds": float(odds[p] + 1),
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
# Residual meta: market coefficient learned and finite
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

# ===== bayes: tie convention, coherence, posterior predictive =====

from golfmodel.bayes import (MARKET_CUTS, N_SKILL_BANDS, fit_model,
                             posterior_predictive)

_cuts = list(MARKET_CUTS.values())
_rng = np.random.default_rng(7)

# Synthetic field: score driven by a feature the model can see, with a
# deliberately skewed, skill-dependent error so the residual machinery is
# exercised rather than bypassed.
_np_, _ne = 80, 40
_rows = []
for _e in range(_ne):
    _pl = _rng.choice(_np_, 60, replace=False)
    _rating = 60 + _pl / 4.0
    _spread = 1.4 + 0.02 * (_np_ - _pl)              # better players steadier
    _err = _rng.gumbel(0, _spread / 1.6, len(_pl))   # right-skewed, like golf
    _sc = 72 - 0.08 * (_rating - 70) + _err
    _rows.append(pd.DataFrame({
        "Date": pd.Timestamp("2021-01-03") + pd.Timedelta(days=7 * _e),
        "eventID": 900 + _e, "playerID": _pl, "score": _sc,
        "posn": _sc.argsort().argsort() + 1, "rating": _rating,
    }))
_bt = pd.concat(_rows, ignore_index=True)

_pkg = fit_model(_bt, "features")
assert _pkg["sigma2"] > 0
assert len(_pkg["resid_pools"]) == N_SKILL_BANDS
assert all(len(pool) > 0 for pool in _pkg["resid_pools"]), "empty residual pool"
print("bayes fit_model: posterior + residual pools: OK")

_ev = _bt[_bt.eventID == 900]
_p = posterior_predictive(_ev, _pkg, _cuts, n_sims=20_000, seed=3)

# Coherence is the whole point: every cut is read off the same finishing order,
# so P(win) <= P(top5) <= P(top10) <= P(top20) cannot be violated.
assert (np.diff(_p, axis=1) >= -1e-12).all(), "simulated markets not monotone"
assert ((_p >= 0) & (_p <= 1)).all()
print("bayes posterior_predictive: monotone across markets: OK")

# Golf's tie convention: posn = 1 + players strictly better, so ties share the
# better position and MORE than `cut` players can satisfy posn <= cut. Measured
# in the real data: 22.5 players per event have posn <= 20. Continuous ranking
# would give exactly 20.0 and under-predict against the binary top_k targets.
_sums = _p.sum(axis=0)
# Exactly 1.0 by construction: a tie at the top is split as a playoff.
assert abs(_sums[0] - 1.0) < 1e-6, f"Winner sum {_sums[0]:.4f} must be 1.0"
assert _sums[3] > 20.5, f"Top20 sum {_sums[3]:.2f} shows no ties (continuous gives 20.0)"
assert _sums[3] < 26.0, f"Top20 sum {_sums[3]:.2f} implies far too many ties"
print(f"bayes tie convention: per-event sums "
      f"{_sums[0]:.2f}/{_sums[1]:.2f}/{_sums[2]:.2f}/{_sums[3]:.2f}: OK")

# A better-rated player must never be less likely to place.
_rank_corr = pd.Series(_p[:, 3]).corr(pd.Series(_ev["rating"].to_numpy()),
                                      method="spearman")
assert _rank_corr > 0.9, f"P(top20) not increasing in rating (rho={_rank_corr:.2f})"
print(f"bayes posterior_predictive: P(top20) rises with rating "
      f"(rho={_rank_corr:.2f}): OK")

# Monte-Carlo noise must be small enough that aggregate metrics are seed-stable.
_p2 = posterior_predictive(_ev, _pkg, _cuts, n_sims=20_000, seed=104)
assert np.abs(_p - _p2).mean() < 0.005, "simulation too noisy at N_SIMS"
print("bayes posterior_predictive: seed-stable: OK")

# Chunking is an implementation detail and must not change the answer.
import golfmodel.bayes as _bayes
_orig_chunk = _bayes.SIM_CHUNK
_bayes.SIM_CHUNK = 997
_p3 = posterior_predictive(_ev, _pkg, _cuts, n_sims=20_000, seed=3)
_bayes.SIM_CHUNK = _orig_chunk
assert np.abs(_p - _p3).mean() < 0.005, "chunk size changed the result"
print("bayes posterior_predictive: chunk-size invariant: OK")

# The residual pools must actually differ by skill band, or the measured
# heteroscedasticity (1.27-1.31x best-to-worst sd) is being thrown away.
_sds = [float(np.std(pool)) for pool in _pkg["resid_pools"]]
assert max(_sds) / min(_sds) > 1.1, "residual pools carry no spread difference"
print(f"bayes residual pools: spread ratio {max(_sds)/min(_sds):.2f}x: OK")

print("\nALL SMOKE TESTS PASSED")
