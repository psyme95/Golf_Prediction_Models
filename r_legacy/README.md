# R version (legacy)

The first version of the model, written in R before it was rewritten in Python (see the main README). It is kept for reference and is no longer maintained.

## Method

Each market (Winner, Top 5, Top 10, Top 20) was modelled as a binary outcome with an ensemble of GAM, random forest, neural network, GBM and XGBoost models, built with biomod2 (Thuiller et al., 2009). biomod2 is a species distribution modelling package; I used it because it was the ensemble framework I already knew from ecological modelling. Finishing inside the market cut took the place of presence, finishing outside it took the place of absence, and dummy coordinates were supplied because biomod2 expects spatial data. Ensemble members were weighted by TSS, and the ensemble score was then calibrated per tour and market with either Platt scaling or a binomial GLM on the score and the market odds.

A second, in-tournament layer (scripts 4 and 5) re-priced players after round 2.

## Running

Open `r_legacy.Rproj` so paths resolve to this folder, put the raw files in `Input/`, and run the scripts in order:

1. `1_data_preprocessing.R` builds the processed historical and weekly files
2. `2_seasonal_model_training.R` trains the seasonal ensemble and calibration models
3. `3_weekly_predictions.R` writes the weekly prediction workbooks
4. `4_rd2_model_training.R` trains the round 2 layer
5. `5_rd2_predictions.R` applies it to the current event

## Known issues

These are the main reasons for the rewrite, and all are addressed in the Python pipeline:

- Cross-validation splits were random by row, so players from the same tournament appeared in both training and validation folds.
- Ensemble weights used TSS from the full-data (`allRun`) models, and the calibration models were fitted on in-sample ensemble scores. Both are optimistic, and the second makes the calibrated probabilities overconfident.
- `complete.cases()` was applied to the whole data frame, which removed any player with a missing value in any column, including columns the models never used. This shrank event fields and biased the field-relative features.

## References

Thuiller, W., Lafourcade, B., Engler, R. & Araújo, M.B. (2009). BIOMOD – a platform for ensemble forecasting of species distributions. *Ecography*, 32, 369–373.
