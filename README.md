# Golf Prediction Models

Probability models for professional golf finishing-position markets (Winner, Top 5, Top 10 and Top 20) on the PGA Tour and the European (DP World) Tour, evaluated with a walk-forward backtest against Betfair exchange prices.

The underlying problem is one of pricing: estimating the probability of an outcome for each player, and comparing it with the price a market is offering. Much of the work has therefore gone into calibration, into validation that does not leak information between training and test data, and into checking whether apparent edges survive realistic prices and settlement rules.

The project is in paper trading. No real money has been staked.

## Repository layout

```
golfmodel/        Python package and command-line interface
  config.py       paths, markets, feature lists and betting settings
  data.py         loading, preprocessing and feature engineering
  modeling.py     grouped cross-validation, tuning, stacking and weekly predictions
  bayes.py        Bayesian field model (alternative to the ensemble)
  backtest.py     settlement, summaries, strategy grids and the walk-forward driver
analysis/         one-off screens of back and each-way betting
tests/            smoke test on synthetic data
docs/             backtest results write-up
r_legacy/         the original R version of the model
```

## Data

Each row is one player in one tournament. The features include player ratings and recent form, course and location history, strokes-gained statistics, field size and strength, and bookmaker and exchange prices for each market. Historical rows also carry the finishing position and the settled result of each bet. The data cover 2020 to mid-2026 from Tour-Tips.com using a Master Data Subscription and are not included in this repository.

Player strength only matters relative to the rest of that week's field, so most features are also expressed relative to the event: z-scores, percentiles, and differences from the field mean, median and best.

## Method

### Stacked ensemble

Each market is modelled as a binary outcome (e.g. finishing in the top 20) with five base learners: elastic-net logistic regression, random forest, LightGBM, XGBoost and LightGBM with DART boosting. Hyperparameters were tuned with Optuna (Akiba et al., 2019) to minimise cross-validated log-loss, as betting returns depend on the probabilities being well calibrated rather than on ranking alone.

Out-of-fold predictions from five repeats of five-fold cross-validation were then used to fit a logistic regression meta-model on the base models' log-odds. For both tours the meta-model also receives the market's implied log-odds, so that it models where the base models should depart from the market. Its coefficient on the market term shows how heavily the final prediction leans on the market price.

All cross-validation is grouped by event. Players in the same tournament share a course, weather and field, and allowing them into both training and validation folds produced optimistic out-of-fold scores in an earlier version.

### Normalisation

Probabilities are normalised within each event to sum to the number of places paid (1 for Winner, 20 for Top 20). Players dropped for missing features keep their share of the market-implied probability, so the remaining players are normalised to the corresponding fraction of the total rather than the full number of places.

### Dead-heat labels

Ties across a market cut are settled as dead heats, so a player tied for 20th with five others is paid a fraction of the stake rather than in full. With `--labels frac`, the model is trained on the settled fraction instead of the binary label. As the base learners are classifiers, each partially paid row is represented as two weighted rows, one positive with weight *f* and one negative with weight 1 − *f*, which reproduces the cross-entropy against the fractional target exactly.

### Bayesian field model

As an alternative (`--model bayes`), a Bayesian linear regression predicts each player's score relative to the field, and Monte Carlo simulation of the whole field produces every market from the same finishing order. This guarantees P(win) ≤ P(top 5) ≤ P(top 10) ≤ P(top 20), which the per-market ensemble does not. Residuals are resampled within bands of predicted skill to keep their heteroscedasticity and skew, and simulated scores are rounded to golf's scoring granularity so that ties occur at a realistic rate. It runs in minutes rather than hours, as there is nothing to tune, but is currently available in the walk-forward backtest only.

### Backtesting

The walk-forward backtest trains on a rolling window of the two calendar years before each test year and prices every event in the test year, from 2022 to 2026. With `--window expanding`, each window instead trains on every year from the start of the data, and the same option applies to the seasonal model trained by `train`. Bets are settled with dead-heat rules and 3% commission on the net P&L of each market. Strategy grids then sweep edge thresholds, odds bands and rating filters one at a time, reporting the number of profitable years and the worst year alongside total P&L and Sharpe ratio.

## Results

Full results, including model comparisons, robustness checks and negative results, are in [docs/backtest_results.md](docs/backtest_results.md). The figures below are from the walk-forward runs of October 2026, with data to 16/08/2026, using the rolling-window ensemble.

Laying every player in the Top 20 market whose model odds exceed the Betfair lay odds, at lay odds below 50, was profitable in all five test years on both tours:

| | PGA | European Tour |
|---|---|---|
| Sharpe ratio (per event) | 0.504 | 0.496 |
| Years profitable | 5/5 | 5/5 |
| Return on liability | 1.31% | 1.19% |

The main findings were:

- Fixing errors in normalisation, class weighting and the meta-model improved calibration in every market, and raised the return on liability of the Top 20 strategy on both tours.
- Better calibration in the longshot tail made the model lay many more extreme outsiders. Without an odds cap, this raised the Sharpe ratio but lowered the return on capital, so the cap is part of the rule.
- Neither an expanding training window nor the Bayesian field model, with or without market prices, beat the rolling-window ensemble by more than chance. No alternative was best on both tours, and only the ensemble was profitable in every year on both tours.
- Much of the apparent decline in the PGA edge was model error. PGA 2026 remains weak under every model and window tested, so a genuine decline cannot be ruled out.
- No back betting or each-way strategy survived realistic prices.
- A tie-breaking error that leaked finishing order into bet ranking was found by its implausible result (618% ROI) and fixed.

The main limitation is that the betting rule, model and training window were chosen after inspecting the same out-of-sample predictions, so forward paper trading is the only fully independent test.

## Running

Requires Python 3.13.

```
pip install -r requirements.txt
python -m tests.test_smoke          # synthetic data, runs in about a minute
```

With the raw files (`PGA.xlsx`, `Euro.xlsx`, `This_Week_PGA.csv`, `This_Week_Euro.csv`) in `Input/`:

```
python -m golfmodel preprocess      # raw files -> processed features
python -m golfmodel train           # seasonal models for each tour and market
python -m golfmodel predict         # weekly prediction workbooks
python -m golfmodel walkforward     # walk-forward backtest
```

The alternatives in the results were run with:

```
python -m golfmodel walkforward --window expanding
python -m golfmodel walkforward --model bayes --prior features   # or --prior market
```

`--parallel` runs both tours at once, with logs in `Output/Logs/`.

Each command takes `--tour PGA` or `--tour Euro`, and `python -m golfmodel <command> -h` lists the remaining options. Outputs are written to `Output/`.

## Background

The model was first built in R with biomod2, an ensemble package from species distribution modelling (see [r_legacy/](r_legacy/)). It was rewritten in Python to fix leakage in the cross-validation and the calibration, and to make the walk-forward backtest practical to run.

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: a next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.
