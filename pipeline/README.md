# Golf Prediction Pipeline

Clean rewrite of `Weekly_Modelling_Python` — one package, six modules, one CLI.
Runs on the system Python 3.13 (`py -3.13`); the repo `.venv` (3.14) is broken.

## Commands

Run from this folder (`pipeline/`):
cd D:\Golf\repo\pipeline

```
py -3.13 -m golfmodel preprocess   [--kind historical|weekly] [--tour PGA/Euro]   # raw → processed (default: both kinds, both tours)
py -3.13 -m golfmodel train        [--trials 75]  # seasonal 5-model ensemble + meta, per market
py -3.13 -m golfmodel predict                     # weekly prediction workbooks
py -3.13 -m golfmodel season       [--year 2026]  # backtest production bundle vs completed events
py -3.13 -m golfmodel walkforward  [--trials 30] [--start-year Y] [--min-test-year Y]
                                   [--force-retrain] [--parallel] [--tag NAME]
py -3.13 -m golfmodel paper        --date 23-06-2026
```

`--tag` segregates the walk-forward model cache and output workbook, for A/B
experiment runs (e.g. `--tag residual` vs `--tag noodds` after flipping
`use_meta_odds` in config.py).

All commands accept `--tour PGA|Euro` (default: both). `walkforward --parallel`
runs both tours as subprocesses, logging to `Output/Logs/`.

## Layout

```
golfmodel/config.py     paths, markets, features, per-tour settings (use_meta_odds etc.)
golfmodel/data.py       load raw, preprocess + feature engineering, lay-odds join
golfmodel/modeling.py   grouped CV, Optuna tuning, OOF stacking, meta-model, weekly predict
golfmodel/backtest.py   shared backtest core + season/walk-forward drivers
golfmodel/excel_out.py  shared Excel writer
Input/                  processed files          Output/                 models, predictions, backtests
test_smoke.py           synthetic-data smoke test: py -3.13 test_smoke.py
```

Raw inputs (`PGA.xlsx`, `Euro.xlsx`, `This_Week_*.csv`) live in `pipeline/Input/`
alongside the processed files (see `RAW_DIR` in config.py).

## Fixes vs the old pipeline

- Dead-heat reduction factors computed from the **full event field**, not the
  feature-filtered prediction rows; preprocessing no longer drops rows at all.
- Discrimination/calibration metrics use raw `Probability`; the normalised
  column (sums to market size) is for ranking and bet selection only.
- All CV (tuning, OOF, meta) is grouped by `eventID` — no same-event rows
  across folds. OOF metrics are slightly lower than the old pipeline's;
  that is the leakage being removed, not a regression.
- Walk-forward trains with the same `use_meta_odds` setting as production.
- Lay ROI = P&L / total liability everywhere (summaries and grids).
- Paper-testing totals read by header label, not hard-coded column positions.

Dropped: the Rd2 in-tournament layer and all R-pipeline compatibility shims.

## Betting layer (profit improvements, 2026-08-10)

- **Commission**: 3% (`COMMISSION`) deducted from every winning P&L component,
  in all summaries and grids.
- **Lay trigger in weekly predictions**: every market sheet has `Max_Lay_Odds`
  (= `LAY_EDGE_TRIGGER / Probability`) — lay on Betfair while the available lay
  odds are at or below this value; above it the edge is gone. The policy from
  the walk-forward evidence applies this to Top10/Top20 only.
- **Edge bases**: every prediction row carries `Edge_Raw`
  (`Probability × lay odds`) and `Edge_Norm` (`lay odds / Normalised_Model_Odds`).
  Headline bets remain the normalised zero-margin basis pending grid evidence;
  the strategy grids sweep both bases.
- **Strategy grids**: per market × edge basis, one filter at a time — edge
  thresholds (`EDGE_THRESHOLDS`), lay-odds floors/ceilings (`ODDS_GRID`, up to
  1000 for Winner), rating floors/ceilings. Every row reports `Years_Pos`
  (test years profitable, e.g. "4/6") and `Worst_Year_PnL` so strategies can be
  judged on consistency, not just aggregate P&L. Policy selection is human-led.
- **Kelly staking**: 1/10 Kelly from a fixed £1,000 notional bankroll, capped
  at 1% per bet (`KELLY_*` in config), computed from the raw calibrated
  probability alongside fixed stake/liability for comparison. Non-compounding
  so results stay order-independent.
- **Tuning objective**: Optuna minimises grouped-CV log-loss (was average
  precision) — calibration is what betting P&L depends on.
- **Residual modelling**: the meta-learner receives market implied log-odds as
  an anchor feature for both tours (`use_meta_odds=True`). The learned market
  coefficient is reported per market (`Meta_Market_Coef` in the training
  summary) — it shows how much the ensemble leans on market consensus vs
  deviates. Old bundles (raw implied-probability feature) still load and
  predict identically via the `meta_odds_form` bundle key.

## Validation status (2026-08-10)

- Weekly preprocess: exact value parity vs old pipeline; recovers players the
  old blanket dropna deleted (their NaNs are outside feature columns).
- Weekly predict with old production bundles: exact match, all 8 sheets, both
  tours (including the Euro odds-in-meta path).
- `paper`: output matches old script exactly for 23-06-2026.
- Smoke test (`test_smoke.py`): dead-heat edge cases, lay formula, end-to-end
  grouped-CV train → backtest → export on synthetic data.
- Profit-improvement layer (2026-08-10): commission math, Kelly formulas,
  edge columns, grid structure, and residual-meta training all covered by
  test_smoke.py; old-bundle predict parity re-verified after the changes.
- **Pending (needs `PGA.xlsx` / `Euro.xlsx` raw historical files):**
  historical preprocess comparison; fresh grouped-CV training metrics;
  `season` Winner P&L parity vs old `Current_Season_Backtest.py` (note:
  new P&L includes 3% commission — compare gross by setting COMMISSION=0);
  walk-forward A/B runs (`--tag residual` vs `--tag noodds`) and grid
  inspection for the edge/odds/rating strategy decision.
