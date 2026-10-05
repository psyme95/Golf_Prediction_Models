# Backtest results

This document summarises the walk-forward evaluation of the pipeline. The results come from runs in October 2026, after the following fixes:

- normalisation when players are dropped for missing features
- LightGBM bagging
- feature scaling for the logistic base model
- removal of class weighting
- a log-odds meta-model

Where relevant, they are compared with the run immediately before the fixes. Both runs used identical data: the same 188,721 player-market predictions, with the same outcomes and prices on every row. Differences between them are therefore due to the code changes alone.

Six model configurations were evaluated:

- the stacked ensemble and two variants of the Bayesian field model (one using player features only, one with de-vigged market prices added)
- each with a rolling and an expanding training window

The rolling-window ensemble is the configuration carried forward.

## 1. Validation design

For each test year, the full pipeline (hyperparameter tuning, out-of-fold stacking and meta-model fitting) was run on earlier years only, and the resulting models were used to price every event in the test year. Two training windows were compared:

- **Rolling:** the two calendar years before the test year.
- **Expanding:** every year from the start of the data (2020) to the year before the test year.

Test years ran from 2022 to 2026 in both cases, giving five out-of-sample years per tour. The 2026 test year is partial, as the data end on 16/08/2026. In 2022 both windows cover 2020–2021, and the two produced identical predictions, which confirms that later differences arise from the window alone.

All cross-validation within a training window was grouped by event, so that players from the same tournament never appeared in both training and validation folds. Players in the same event share the course, the weather and the field, and splitting them across folds overstated out-of-fold performance in an earlier version of the model.

Betting results were settled against Betfair exchange lay prices captured before each event, with dead heats settled according to exchange rules and 3% commission charged on the net P&L of each market. Results are expressed per £1 of lay stake. Sharpe ratios are the mean of per-event P&L divided by its standard deviation, and are not annualised.

Differences between configurations were assessed with a paired bootstrap over events (2,000 resamples), which gives a 95% interval for the difference in Sharpe ratio and in return on liability.

## 2. Model performance

### 2.1 Effect of the fixes

**Table 1.** Out-of-sample discrimination and calibration of the rolling-window ensemble before and after the fixes (2022–2026). Lower log-loss and Brier scores are better.

| Tour | Market | AUC (before / after) | Log-loss (before / after) | Brier (before / after) |
|---|---|---|---|---|
| PGA | Winner | 0.813 / 0.818 | 0.0418 / 0.0410 | 0.0080 / 0.0079 |
| PGA | Top 5 | 0.775 / 0.780 | 0.1695 / 0.1682 | 0.0431 / 0.0429 |
| PGA | Top 10 | 0.762 / 0.762 | 0.2705 / 0.2699 | 0.0770 / 0.0768 |
| PGA | Top 20 | 0.743 / 0.743 | 0.4129 / 0.4126 | 0.1299 / 0.1299 |
| Euro | Winner | 0.830 / 0.833 | 0.0364 / 0.0361 | 0.0070 / 0.0069 |
| Euro | Top 5 | 0.780 / 0.780 | 0.1545 / 0.1545 | 0.0384 / 0.0382 |
| Euro | Top 10 | 0.764 / 0.764 | 0.2530 / 0.2525 | 0.0707 / 0.0706 |
| Euro | Top 20 | 0.748 / 0.748 | 0.3819 / 0.3815 | 0.1180 / 0.1180 |

The fixes left AUC unchanged or higher, and log-loss unchanged or lower, in all eight tour-market combinations. The largest gains were in the Winner and Top 5 markets, where positives are rarest and the removal of class weighting mattered most.

The clearest improvement was in calibration-in-the-large, the ratio of total predicted to total observed placings:

**Table 2.** Calibration-in-the-large (total predicted ÷ total observed). A value of 1 indicates no overall bias.

| | PGA Winner | PGA Top 5 | Euro Winner | Euro Top 5 |
|---|---|---|---|---|
| Before | 1.19 | 1.05 | 0.99 | 0.89 |
| After | 1.05 | 1.01 | 1.00 | 1.00 |

The longshot tail also improved. For PGA Top 20 players priced at 50 or longer, the observed placing rate was 1.04%. The ensemble predicted 1.90% before the fixes and 1.42% after.

### 2.2 Comparison of models

**Table 3.** Out-of-sample AUC and log-loss of the three models with a rolling window (2022–2026). Ens = stacked ensemble; Bayes (F) = Bayesian model on player features; Bayes (M) = Bayesian model with market prices added.

| Tour | Market | AUC (Ens / Bayes F / Bayes M) | Log-loss (Ens / Bayes F / Bayes M) |
|---|---|---|---|
| PGA | Winner | 0.818 / 0.809 / 0.820 | 0.0410 / 0.0411 / 0.0407 |
| PGA | Top 5 | 0.780 / 0.770 / 0.777 | 0.1682 / 0.1701 / 0.1688 |
| PGA | Top 10 | 0.762 / 0.753 / 0.759 | 0.2699 / 0.2725 / 0.2706 |
| PGA | Top 20 | 0.743 / 0.736 / 0.742 | 0.4126 / 0.4162 / 0.4135 |
| Euro | Winner | 0.833 / 0.816 / 0.820 | 0.0361 / 0.0370 / 0.0367 |
| Euro | Top 5 | 0.780 / 0.770 / 0.779 | 0.1545 / 0.1545 / 0.1531 |
| Euro | Top 10 | 0.764 / 0.756 / 0.764 | 0.2525 / 0.2544 / 0.2523 |
| Euro | Top 20 | 0.748 / 0.740 / 0.746 | 0.3815 / 0.3846 / 0.3818 |

The ensemble had the highest AUC in seven of eight combinations. The feature-only Bayesian model was 0.007–0.017 AUC behind it. The Bayesian model with market prices was within 0.004 AUC of the ensemble except in the Euro Winner market, and had the lower log-loss in three of eight combinations, so the two are effectively equal on calibration.

The Bayesian models read every market off one simulated finishing order, so P(win) ≤ P(top 5) ≤ P(top 10) ≤ P(top 20) always holds. The ensemble models each market separately and broke this ordering for 1.2% of PGA and 0.7% of European Tour player-events after the fixes (1.8% and 0.3% before).

The expanding window changed model quality very little. For the ensemble it lowered log-loss slightly in six of eight combinations over 2023–2026, by at most 0.0008, and raised Winner AUC by 0.013 (PGA) and 0.005 (European Tour). For the Bayesian models the effect was similarly small.

## 3. Betting results

### 3.1 Selected strategy

The strategy carried forward to paper trading uses the rolling-window ensemble. It lays every player in the Top 20 market whose normalised model odds exceed the available lay odds and whose lay odds are below 50, on both tours.

The odds cap matches the trigger already used for live paper trading. After the fixes it also has a material effect on the backtest. Without it, the corrected ensemble lays far more extreme longshots than before:

- PGA: 295 lays at odds of 50 or more, against 20 before the fixes
- European Tour: 582, against 105

These lays raise the uncapped Sharpe ratio (PGA 0.565, European Tour 0.610) but lower the return on liability (1.19% and 1.04%). This is the pattern that ruled out the Winner market: many small wins against rare, very large losses. The Bayesian model with market prices illustrates the risk directly. Without the cap, its worst PGA event lost £410–£413 per £1 of stake (depending on the training window), against £59 for the ensemble. With a rolling window, its uncapped return on liability was lower than the ensemble's by a margin whose bootstrap interval excluded zero, the only such case in the comparison.

**Table 4.** Top 20 lay strategy (lay odds below 50), rolling-window ensemble, net of commission, per £1 of lay stake (2022–2026).

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

**Table 5.** Total P&L by test year after the fixes, per £1 of lay stake (Top 20, lay odds below 50, rolling-window ensemble).

| Test year | PGA | Euro |
|---|---|---|
| 2022 | +£443 | +£459 |
| 2023 | +£423 | +£268 |
| 2024 | +£386 | +£276 |
| 2025 | +£380 | +£342 |
| 2026 (to 16/08) | +£28 | +£94 |

### 3.2 Comparison of models and training windows

**Table 6.** Top 20 lay strategy (lay odds below 50) for all six configurations, per £1 of lay stake (2022–2026). Return on liability is abbreviated to RoL.

| Model | Window | PGA Sharpe | PGA RoL | PGA 2026 RoL | Euro Sharpe | Euro RoL | Euro 2026 RoL | Years profitable (PGA, Euro) |
|---|---|---|---|---|---|---|---|---|
| Ensemble | Rolling | 0.504 | 1.31% | +0.34% | 0.496 | 1.19% | +0.98% | 5/5, 5/5 |
| Ensemble | Expanding | 0.472 | 1.30% | +0.21% | 0.538 | 1.32% | +0.24% | 5/5, 5/5 |
| Bayes (F) | Rolling | 0.448 | 1.30% | −0.50% | 0.513 | 1.44% | +0.76% | 4/5, 5/5 |
| Bayes (F) | Expanding | 0.416 | 1.24% | −0.11% | 0.504 | 1.47% | +0.75% | 4/5, 5/5 |
| Bayes (M) | Rolling | 0.486 | 1.20% | 0.00% | 0.402 | 1.11% | −0.47% | 5/5, 4/5 |
| Bayes (M) | Expanding | 0.541 | 1.34% | +0.42% | 0.461 | 1.30% | −0.09% | 5/5, 4/5 |

None of the five alternatives differed from the rolling-window ensemble by more than chance. For every configuration and tour, the 95% bootstrap intervals for the differences in both Sharpe ratio and return on liability included zero. On the PGA Tour, for example, the interval for the difference in return on liability ranged from −0.30 to +0.36 percentage points for the strongest alternative.

No configuration was best on both tours. The Bayesian model with market prices and an expanding window performed best on the PGA Tour but lost money in 2026 on the European Tour. The feature-only Bayesian model had the highest return on liability on the European Tour but lost money in 2026 on the PGA Tour. Choosing a different configuration for each tour from these results would be a form of selection bias.

The two ensemble configurations were the only ones profitable in every test year on both tours, including 2026. The expanding window gave the ensemble a higher return on liability on the European Tour, almost entirely from 2023 (1.75% against 0.90%). It was weaker in 2026 on both tours, and had deeper drawdowns on both. The rolling-window ensemble was therefore retained. It is also the configuration supported by the weekly training and prediction commands.

The earlier conclusion that anchoring a model to the market improves its accuracy at the cost of betting performance did not hold up. The market-anchored Bayesian model was the weakest bettor with a rolling window but the strongest on the PGA Tour with an expanding window, so the effect depends on the configuration rather than on market anchoring as such.

### 3.3 Top 10 market

Before the fixes, the same rule in the Top 10 market was profitable in every year on both tours and was retained as an optional second tier. It no longer meets that standard.

PGA Top 10 deteriorated under the corrected ensemble:

- Sharpe fell from 0.375 to 0.307 uncapped, and from 0.271 to 0.207 with lay odds below 50.
- With the cap, 2026 was loss-making under both versions of the model.
- It was weak in every configuration tested (Sharpe 0.13–0.27 with the cap).

The European Tour Top 10 was stronger under the Bayesian models (Sharpe 0.32–0.39 with the cap, profitable in all five years), but it does not reach the standard of the Top 20 rule. The Top 10 market is therefore not recommended on either tour.

### 3.4 Robustness

As the strategy was chosen after inspecting these results, its neighbourhood was checked to establish whether the result depended on a precise setting. The Top 20 grid contains 68 variants with at least 200 bets on both tours:

- 65 were profitable on both tours, compared with 60 of 66 before the fixes.
- 41 were profitable in all five years on both tours (50 with an expanding window).

The variants include edge thresholds, odds bands, rating filters and limits on the number of bets per event.

The variants that failed were consistently those restricted to short prices (lay odds below 2, 3 or 10) or to the strongest players (rating of 60 or more). This is consistent with the favourite–longshot bias (Snowberg & Wolfers, 2010): short-priced players are priced fairly or generously, so there is little to gain from laying them.

### 3.5 Edge decay

**Table 7.** Return on liability (%) by test year, Top 20, lay odds below 50, rolling-window ensemble.

| | 2022 | 2023 | 2024 | 2025 | 2026 (to 16/08) |
|---|---|---|---|---|---|
| PGA before | 1.68 | 1.32 | 1.16 | 1.00 | 0.44 |
| PGA after | 1.53 | 1.19 | 1.33 | 1.50 | 0.34 |
| Euro before | 1.84 | 1.22 | 0.49 | 1.01 | 0.58 |
| Euro after | 1.76 | 0.90 | 0.97 | 1.26 | 0.98 |

Before the fixes, the PGA return declined in every year. The corrected ensemble removes most of that decline in 2024–2025, which suggests that part of the apparent decay was model error rather than increasing market efficiency.

PGA 2026 nevertheless remains weak. Across the six configurations its return on liability ranged from −0.50% to +0.42%, so the weakness does not depend on the choice of model or training window. With five yearly observations, a genuine decline in the PGA edge cannot be ruled out. The European Tour shows no clear trend.

## 4. Negative results

- **Back betting.** Back bets were originally priced at the exchange lay quote, which is better than any price that can actually be backed at. They were therefore repriced at a 5% spread below the lay quote in every run (`analysis/back_screen.py`). The screen required positive P&L, at least 200 bets and four of five years positive, from 290 strategies per tour.
  - Before the fixes, no strategy passed.
  - With the corrected rolling-window ensemble, none passed on the PGA Tour and four did on the European Tour.
  - With an expanding window, 3 passed on the PGA Tour and 13 on the European Tour.

  No strategy passed on both tours in any configuration. The European Tour survivors were mostly rating floors (backing players rated 55–75) in each market, with Sharpe ratios up to 0.15. This is a consistent pattern, but it is limited to one tour and depends on an optimistic spread assumption: a single price tick is already 2.4–4.6% of the price. Backing every flagged player lost money in every market on both tours.
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

- **In-sample strategy selection.** The models were evaluated out of sample, but the betting rule (including the odds cap), the model and the training window were all chosen after inspecting results on the same predictions. The robustness checks in Section 3.4, and the decision to keep the default configuration where alternatives were not clearly better, reduce but do not remove the risk of overfitting. Forward paper trading is the only fully out-of-sample test.
- **Execution.** Results assume that every bet is matched at the pre-event snapshot price, including outsiders in thin place markets. Partial fills would produce a different, untested strategy.
- **Tuning and stacking on the same data.** Hyperparameters were tuned on the same training window used to generate the out-of-fold predictions for the meta-model, which makes the out-of-fold scores slightly optimistic. The test years are unaffected.
- **Monte Carlo noise.** The Bayesian models are simulated, and earlier checks showed their lay Sharpe ratios varying in the second decimal place between random seeds. Small differences between configurations in Table 6 should not be over-interpreted.
- **Short record.** Five test years, the last of them partial, provide limited evidence about trends.

## References

Snowberg, E. & Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? *Journal of Political Economy*, 118(4), 723–746.
