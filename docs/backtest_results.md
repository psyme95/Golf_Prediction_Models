# Backtest results

This document summarises the walk-forward evaluation of the pipeline. The current results come from a run in October 2026, after the following fixes:

- normalisation when players are dropped for missing features
- LightGBM bagging
- feature scaling for the logistic base model
- removal of class weighting
- a log-odds meta-model

Where relevant, they are compared with the run immediately before the fixes. Both runs used identical data: the same 188,721 player-market predictions, with the same outcomes and prices on every row. Differences between them are therefore due to the code changes alone.

## 1. Validation design

Models were evaluated with rolling walk-forward windows. For each test year, the full pipeline (hyperparameter tuning, out-of-fold stacking and meta-model fitting) was run on the two preceding calendar years only, and the resulting models were used to price every event in the test year. Test years ran from 2022 to 2026, giving five out-of-sample years per tour. The 2026 test year is partial, as the data end on 16/08/2026.

All cross-validation within a training window was grouped by event, so that players from the same tournament never appeared in both training and validation folds. Players in the same event share the course, the weather and the field, and splitting them across folds overstated out-of-fold performance in an earlier version of the model.

Betting results were settled against Betfair exchange lay prices captured before each event, with dead heats settled according to exchange rules and 3% commission charged on the net P&L of each market. Results are expressed per £1 of lay stake. Sharpe ratios are the mean of per-event P&L divided by its standard deviation, and are not annualised.

## 2. Model performance

**Table 1.** Out-of-sample discrimination and calibration by tour and market (2022–2026), on the rows predicted by all three models. Before = ensemble before the fixes; After = ensemble after the fixes; Bayes = Bayesian field model on player features only. Lower log-loss and Brier scores are better.

| Tour | Market | AUC (Before / After / Bayes) | Log-loss (Before / After / Bayes) | Brier (Before / After / Bayes) |
|---|---|---|---|---|
| PGA | Winner | 0.813 / 0.818 / 0.809 | 0.0418 / 0.0410 / 0.0410 | 0.0080 / 0.0079 / 0.0079 |
| PGA | Top 5 | 0.775 / 0.780 / 0.770 | 0.1695 / 0.1682 / 0.1701 | 0.0431 / 0.0429 / 0.0432 |
| PGA | Top 10 | 0.762 / 0.762 / 0.753 | 0.2705 / 0.2699 / 0.2725 | 0.0770 / 0.0768 / 0.0772 |
| PGA | Top 20 | 0.743 / 0.743 / 0.736 | 0.4129 / 0.4126 / 0.4162 | 0.1299 / 0.1299 / 0.1308 |
| Euro | Winner | 0.830 / 0.833 / 0.816 | 0.0364 / 0.0361 / 0.0370 | 0.0070 / 0.0069 / 0.0069 |
| Euro | Top 5 | 0.780 / 0.780 / 0.770 | 0.1545 / 0.1545 / 0.1545 | 0.0384 / 0.0382 / 0.0384 |
| Euro | Top 10 | 0.764 / 0.764 / 0.756 | 0.2530 / 0.2525 / 0.2544 | 0.0707 / 0.0706 / 0.0710 |
| Euro | Top 20 | 0.748 / 0.748 / 0.740 | 0.3819 / 0.3815 / 0.3846 | 0.1180 / 0.1180 / 0.1190 |

The fixes left AUC unchanged or higher, and log-loss unchanged or lower, in all eight tour-market combinations. The largest gains were in the Winner and Top 5 markets, where positives are rarest and the removal of class weighting mattered most.

The clearest improvement was in calibration-in-the-large, the ratio of total predicted to total observed placings:

**Table 2.** Calibration-in-the-large (total predicted ÷ total observed). A value of 1 indicates no overall bias.

| | PGA Winner | PGA Top 5 | Euro Winner | Euro Top 5 |
|---|---|---|---|---|
| Before | 1.19 | 1.05 | 0.99 | 0.89 |
| After | 1.05 | 1.01 | 1.00 | 1.00 |

The longshot tail also improved. For PGA Top 20 players priced at 50 or longer, the observed placing rate was 1.04%. The ensemble predicted 1.90% before the fixes and 1.42% after.

The feature-only Bayesian model was 0.007–0.017 AUC behind the corrected ensemble. Before the fixes, the Bayesian model with market prices added had the best log-loss in seven of eight combinations. On the rows both models predicted, the corrected ensemble now has the lower log-loss in five of eight. The Bayesian model remains of interest because it reads every market off one simulated finishing order, which guarantees P(win) ≤ P(top 5) ≤ P(top 10) ≤ P(top 20). The per-market ensemble broke this ordering for 1.8% of PGA rows and 0.3% of European Tour rows before the fixes.

## 3. Betting results

### 3.1 Selected strategy

The strategy carried forward to paper trading lays every player in the Top 20 market whose normalised model odds exceed the available lay odds and whose lay odds are below 50, on both tours.

The odds cap matches the trigger already used for live paper trading. After the fixes it also has a material effect on the backtest. Without it, the corrected model lays far more extreme longshots than before:

- PGA: 295 lays at odds of 50 or more, against 20 before the fixes
- European Tour: 582, against 105

These lays raise the uncapped Sharpe ratio (PGA 0.565, European Tour 0.610) but lower the return on liability (1.19% and 1.04%). This is the pattern that ruled out the Winner market: many small wins against rare, very large losses.

**Table 3.** Top 20 lay strategy (lay odds below 50), net of commission, per £1 of lay stake (2022–2026).

| | PGA before | PGA after | Euro before | Euro after |
|---|---|---|---|---|
| Sharpe (per event) | 0.494 | 0.504 | 0.419 | 0.496 |
| Years profitable | 5/5 | 5/5 | 5/5 | 5/5 |
| Bets per event | 80.7 | 81.4 | 80.6 | 79.3 |
| Loss rate | 19.5% | 20.0% | 17.9% | 18.3% |
| Mean P&L per event | £7.90 | £8.74 | £7.44 | £8.67 |
| Worst event | −£40 | −£62 | −£40 | −£71 |
| Liability per event | £637 | £669 | £721 | £731 |
| Return on liability | 1.24% | 1.31% | 1.03% | 1.19% |

The fixes improved the return on capital on both tours, by a larger margin on the European Tour. The PGA Sharpe gain (+0.010) is within the variation previously observed between random seeds and should not be over-interpreted. The worst single event was worse after the fixes on both tours.

**Table 4.** Total P&L by test year after the fixes, per £1 of lay stake (Top 20, lay odds below 50).

| Test year | PGA | Euro |
|---|---|---|
| 2022 | +£443 | +£459 |
| 2023 | +£423 | +£268 |
| 2024 | +£386 | +£276 |
| 2025 | +£380 | +£342 |
| 2026 (to 16/08) | +£28 | +£94 |

### 3.2 Top 10 market

Before the fixes, the same rule in the Top 10 market was profitable in every year on both tours and was retained as an optional second tier. It no longer meets that standard.

PGA Top 10 deteriorated under the corrected model:

- Sharpe fell from 0.375 to 0.307 uncapped, and from 0.271 to 0.207 with lay odds below 50.
- With the cap, 2026 was loss-making under both models.

The European Tour Top 10 improved on return on liability (0.59% to 0.77% with the cap), but 2025 became loss-making. The Top 10 market is therefore no longer recommended on either tour.

### 3.3 Robustness

As the strategy was chosen after inspecting these results, its neighbourhood was checked to establish whether the result depended on a precise setting. The Top 20 grid contains 68 variants with at least 200 bets on both tours:

- 65 were profitable on both tours, compared with 60 of 66 before the fixes.
- 41 were profitable in all five years on both tours.

The variants include edge thresholds, odds bands, rating filters and limits on the number of bets per event.

The variants that failed were consistently those restricted to short prices (lay odds below 2, 3 or 10) or to the strongest players (rating of 60 or more). This is consistent with the favourite–longshot bias (Snowberg & Wolfers, 2010): short-priced players are priced fairly or generously, so there is little to gain from laying them.

### 3.4 Edge decay

**Table 5.** Return on liability (%) by test year, Top 20, lay odds below 50.

| | 2022 | 2023 | 2024 | 2025 | 2026 (to 16/08) |
|---|---|---|---|---|---|
| PGA before | 1.68 | 1.32 | 1.16 | 1.00 | 0.44 |
| PGA after | 1.53 | 1.19 | 1.33 | 1.50 | 0.34 |
| Euro before | 1.84 | 1.22 | 0.49 | 1.01 | 0.58 |
| Euro after | 1.76 | 0.90 | 0.97 | 1.26 | 0.98 |

Before the fixes, the PGA return declined in every year. The corrected model removes most of that decline in 2024–2025, which suggests that part of the apparent decay was model error rather than increasing market efficiency. PGA 2026 remains weak under both models (£1.21 per event), however, and with five yearly observations a genuine trend cannot be ruled out. The European Tour shows no clear trend under either model.

### 3.5 Calibration and profit

Before the fixes, the best-calibrated model was the Bayesian model with market prices added, and it was also the weakest bettor of the models tested. This suggested that anchoring a model to the market improves accuracy at the cost of the disagreement with the market that the lay strategy relies on.

The corrected ensemble weakens that conclusion. It is now as well calibrated as the market-anchored Bayesian model and has a higher return on liability, so better calibration and better betting performance are not inherently opposed. The betting results of the Bayesian models were produced before the normalisation fix and should be re-run before the two approaches are compared further.

## 4. Negative results

- **Back betting.** Back bets were originally priced at the exchange lay quote, which is better than any price that can actually be backed at. They were therefore repriced at a 5% spread below the lay quote, the same in both runs (`analysis/back_screen.py`). Before the fixes, none of 290 back strategies per tour remained profitable with at least 200 bets and four of five years positive. After the fixes, none did on the PGA Tour and four did on the European Tour. None did on both tours. The four European Tour survivors used unrelated filters and had Sharpe ratios of 0.045 or less. Backing every flagged player lost money in every market on both tours. A single price tick is already 2.4–4.6% of the price, so a 5% spread is optimistic.
- **Each-way.** Settlement was reproduced on 99.95% of rows, but no each-way strategy passed the consistency checks on both tours (`analysis/ew_screen.py`).
- **Top 5 lay.** The fixes brought this market closer to viability. Variants with at least four of five years positive on both tours rose from 7 to 28, but none was profitable in all five years on both tours.
- **Winner lay.** The only variants profitable in every year on both tours laid players at an average price of around 750. Their results depend on a handful of rare losses.

## 5. Errors found during evaluation

- **Leakage through tie-breaking.** Within-event bet ranking used `rank(method="first")`, which breaks ties by row order. The prediction rows were stored in finishing order, so tied players were ranked best finisher first. In the each-way screen, where 53.6% of rows tied, this produced an apparent ROI of 618%, which fell to 9.29% once ties were broken on player ID. The same error affected 2–3% of rows in the main grid.
- **Inflated normalised probabilities.** Players dropped for missing features were excluded before probabilities were normalised to the number of places, which inflated the probabilities of the remaining players by the share of the market held by those dropped. In a typical weekly PGA field this was around 3.5%, which is large relative to a return on liability of around 1%.
- **Inactive LightGBM bagging.** The `subsample` hyperparameter was tuned for both LightGBM models but had no effect, because LightGBM only applies bagging when `subsample_freq` is set.
- **Unstable meta-model scaling.** Once class weighting was removed, one base model produced near-constant out-of-fold predictions in a synthetic test. Standardising the meta-model inputs divided that model's later predictions by a very small standard deviation, pushing the final probabilities to 0 or 1. The meta-model now uses unstandardised log-odds.
- **Corrupt settlement data.** European Tour Top 40 settlement figures for 2020–2021 implied returns of +75% to +77%, against −19% to −22% in later years, with only 46 of 222 events priced. Top 40 was excluded from the exchange analysis.
- **Partially priced events.** De-vigged market probabilities were inflated by up to 4.6 times in events where only part of the field was priced. Events with less than 90% price coverage are now excluded from the market features.

## 6. Limitations

- **In-sample strategy selection.** The models were evaluated out of sample, but the betting rule, including the odds cap, was chosen from several hundred strategy evaluations on the same predictions. The robustness checks in Section 3.3 reduce, but do not remove, the risk of overfitting at this stage, and forward paper trading is the only fully out-of-sample test.
- **Execution.** Results assume that every bet is matched at the pre-event snapshot price, including outsiders in thin place markets. Partial fills would produce a different, untested strategy.
- **Tuning and stacking on the same data.** Hyperparameters were tuned on the same training window used to generate the out-of-fold predictions for the meta-model, which makes the out-of-fold scores slightly optimistic. The test years are unaffected.
- **Training window.** Only a rolling two-year training window has been tested. An expanding window may be more stable but has not been evaluated.
- **Short record.** Five test years, the last of them partial, provide limited evidence about trends.

## References

Snowberg, E. & Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *Journal of Political Economy*, 118(4), 723–746.
