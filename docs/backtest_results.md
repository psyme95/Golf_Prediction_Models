# Backtest results

This document summarises the walk-forward evaluation of the pipeline. The figures were produced in September 2026, before the modelling fixes made in October 2026 (normalisation with dropped players, LightGBM bagging, feature scaling for the logistic model, removal of class weighting and the log-odds meta-model). They will be updated once the walk-forward has been re-run, and the two sets compared.

## 1. Validation design

Models were evaluated with an expanding sequence of walk-forward windows. For each test year, the full pipeline (hyperparameter tuning, out-of-fold stacking and meta-model fitting) was run on the two preceding calendar years only, and the resulting models were used to price every event in the test year. Test years ran from 2022 to 2026, giving five out-of-sample years per tour. The 2026 test year is partial, as the data end on 28/06/2026.

All cross-validation within a training window was grouped by event, so that players from the same tournament never appeared in both training and validation folds. Players in the same event share the course, the weather and the field, and splitting them across folds overstated out-of-fold performance in an earlier version of the model.

Betting results were settled against Betfair exchange lay prices captured before each event, with dead heats settled according to exchange rules and 3% commission charged on the net P&L of each market. Sharpe ratios are the mean of per-event P&L divided by its standard deviation, and are not annualised.

## 2. Model performance

Table 1 compares the stacked ensemble with the Bayesian field model, with and without market prices as features, on held-out predictions pooled across all five test years.

**Table 1.** Out-of-sample discrimination and calibration by tour and market (2022–2026). Ens = stacked ensemble; Bayes (F) = Bayesian model on player features only; Bayes (M) = Bayesian model with de-vigged market prices added. Lower log-loss and Brier scores are better.

| Tour | Market | AUC (Ens / Bayes F / Bayes M) | Log-loss (Ens / Bayes F / Bayes M) | Brier (Ens / Bayes F / Bayes M) |
|---|---|---|---|---|
| PGA | Winner | 0.812 / 0.810 / 0.821 | 0.0418 / 0.0406 / 0.0402 | 0.0080 / 0.0078 / 0.0077 |
| PGA | Top 5 | 0.775 / 0.773 / 0.780 | 0.1694 / 0.1678 / 0.1664 | 0.0431 / 0.0425 / 0.0424 |
| PGA | Top 10 | 0.761 / 0.755 / 0.761 | 0.2705 / 0.2697 / 0.2676 | 0.0769 / 0.0762 / 0.0760 |
| PGA | Top 20 | 0.743 / 0.739 / 0.745 | 0.4126 / 0.4119 / 0.4090 | 0.1298 / 0.1292 / 0.1286 |
| Euro | Winner | 0.835 / 0.818 / 0.826 | 0.0362 / 0.0367 / 0.0363 | 0.0070 / 0.0069 / 0.0069 |
| Euro | Top 5 | 0.781 / 0.772 / 0.781 | 0.1546 / 0.1537 / 0.1523 | 0.0385 / 0.0382 / 0.0380 |
| Euro | Top 10 | 0.765 / 0.758 / 0.766 | 0.2530 / 0.2531 / 0.2508 | 0.0707 / 0.0705 / 0.0702 |
| Euro | Top 20 | 0.749 / 0.742 / 0.748 | 0.3819 / 0.3828 / 0.3798 | 0.1181 / 0.1183 / 0.1176 |

The Bayesian model with market prices was the best calibrated, with the lowest log-loss and Brier score in seven of eight tour-market combinations. The feature-only Bayesian model, a single regularised linear regression with a simulation, was within approximately 0.005 AUC and 0.001 log-loss of the five-model ensemble.

The Bayesian model was built because the ensemble trains each market independently, so nothing requires P(win) ≤ P(top 5) ≤ P(top 10) ≤ P(top 20). In the ensemble's walk-forward predictions, 1.8% of PGA rows and 0.3% of European Tour rows broke this ordering. The Bayesian model reads every market off one simulated finishing order, so the ordering holds by construction.

## 3. Betting results

### 3.1 Selected strategy

The strategy carried forward to paper trading lays every player in the Top 20 market whose normalised model odds exceed the available lay odds, on both tours, with no further filter.

**Table 2.** Top 20 lay strategy, net of commission, per £1 of lay stake (2022–2026).

| | PGA | Euro |
|---|---|---|
| Sharpe (per event) | 0.497 | 0.466 |
| Years profitable | 5/5 | 5/5 |
| Bets per event | 81 | 82 |
| Mean P&L per event | £7.98 | £8.25 |
| Return on liability | 1.23% | 1.08% |

The same rule in the Top 10 market was also profitable in every year (Sharpe 0.378 PGA, 0.320 Euro), but required roughly four times the liability for a lower Sharpe ratio.

**Table 3.** Total P&L by test year, per £1 of lay stake.

| Test year | PGA Top 20 | Euro Top 20 | PGA Top 10 | Euro Top 10 |
|---|---|---|---|---|
| 2022 | +£466 | +£361 | +£764 | +£143 |
| 2023 | +£433 | +£436 | +£867 | +£640 |
| 2024 | +£331 | +£169 | +£692 | +£61 |
| 2025 | +£234 | +£287 | +£177 | +£199 |
| 2026 (partial) | +£28 | +£92 | +£33 | +£223 |

### 3.2 Robustness

As the strategy was chosen after inspecting these results, its neighbourhood was checked to establish whether the result depended on a precise setting. Of 54 single-filter variants around the Top 20 rule (edge thresholds, odds bands and rating filters, each at several values), 48 were profitable on both tours and 29 were profitable in all five years on both tours. The variants that failed were consistently those restricted to favourites or short prices, which is consistent with the favourite–longshot bias (Snowberg & Wolfers, 2010): short-priced players are priced fairly or generously, so there is little to gain from laying them.

### 3.3 Edge decay

PGA Top 20 P&L per event fell in every year, from £11.08 in 2022 to £1.41 in 2026, and PGA Top 10 fell in every year from 2023 onwards, whereas the European Tour results varied without a clear trend. With five yearly observations it is not possible to distinguish increasing market efficiency, model degradation and chance, and the recent rate is likely to be a better guide to future returns than the five-year average.

The Bayesian model decayed faster than the ensemble in all four tour-market combinations. Although its five-year totals were competitive, its advantage was concentrated in 2022–2023, and in 2025–2026 the ensemble had the higher return on liability in three of four combinations.

### 3.4 Calibration and profit

The best-calibrated model (Bayesian, with market prices) was the worst bettor of the three on all four tour-market combinations tested. Anchoring the model to the market improves its accuracy but removes the disagreement with the market that the lay strategy relies on. Calibration was therefore not used as the sole criterion for choosing a model.

## 4. Negative results

- **Back betting.** Back bets were originally priced at the exchange lay quote, which is better than any price that can actually be backed at. Repriced at a 5% spread below the lay quote (`analysis/back_screen.py`), none of 292 back strategies remained profitable with at least 200 bets and four of five years positive on either tour. A single price tick is already 2.4–4.6% of the price, so 5% is an optimistic spread.
- **Each-way.** Settlement was reproduced on 99.95% of rows, but no each-way strategy passed the consistency checks on both tours (`analysis/ew_screen.py`).
- **Winner and Top 5 lay.** The profitable Winner variants laid extreme longshots with loss rates of 0.0–0.3%, so their results depended on a handful of rare losses. No Top 5 variant passed the consistency checks.

## 5. Errors found during evaluation

- **Leakage through tie-breaking.** Within-event bet ranking used `rank(method="first")`, which breaks ties by row order. The prediction rows were stored in finishing order, so tied players were ranked best finisher first. In the each-way screen, where 53.6% of rows tied, this produced an apparent ROI of 618%, which fell to 9.29% once ties were broken on player ID. The same error affected 2–3% of rows in the main grid.
- **Corrupt settlement data.** European Tour Top 40 settlement figures for 2020–2021 implied returns of +75% to +77%, against −19% to −22% in later years, with only 46 of 222 events priced. Top 40 was excluded from the exchange analysis.
- **Partially priced events.** De-vigged market probabilities were inflated by up to 4.6 times in events where only part of the field was priced. Events with less than 90% price coverage are now excluded from the market features.

## 6. Limitations

- **In-sample strategy selection.** The models were evaluated out of sample, but the betting rule was chosen from approximately 950 strategy evaluations on the same predictions. The robustness checks in Section 3.2 reduce, but do not remove, the risk of overfitting at this stage, and forward paper trading is the only fully out-of-sample test.
- **Execution.** Results assume that every bet is matched at the pre-event snapshot price, including outsiders in thin place markets. Partial fills would produce a different, untested strategy.
- **Tuning and stacking on the same data.** Hyperparameters were tuned on the same training window used to generate the out-of-fold predictions for the meta-model, which makes the out-of-fold scores slightly optimistic. The test years are unaffected.
- **Short record.** Five test years, the last of them partial, provide limited evidence about trends.

## References

Snowberg, E. & Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *Journal of Political Economy*, 118(4), 723–746.
