# Golf Prediction Models

Ensemble ML pipeline that generates player win/place probabilities for PGA Tour and European Tour tournaments. Supports pre-tournament predictions across four betting markets (Winner, Top5, Top10, Top20) and an in-tournament Round 2 model.

---

## Repository structure

```
Golf_Prediction_Models/
├── Weekly_Modelling/            # R pipeline (original)
│   └── Input/                   # Raw data shared with Python pipeline
├── Weekly_Modelling_Python/     # Python pipeline (mirrors R)
│   ├── Input/                   # Python-processed data
│   ├── Output/
│   │   ├── Models/              # Trained .pkl packages
│   │   └── Predictions/         # Weekly prediction Excel files
│   └── Script/
│       ├── config.py
│       ├── 1_weekly_data_preprocessing.py
│       ├── 2_weekly_model_predictions.py
│       ├── 3_weekly_rd2_model_predictions.py
│       ├── seasonal_model_training.py
│       ├── seasonal_rd2_model_training.py
│       ├── walk_forward_backtest.py
└── Testing Scripts/
```

---

## Python pipeline

### Weekly workflow (run in order)

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `1_weekly_data_preprocessing.py` | Reads raw Excel/CSV from `Weekly_Modelling/Input/`, cleans and writes processed files to `Weekly_Modelling_Python/Input/` |
| 2 | `2_weekly_model_predictions.py` | Loads trained model packages, runs ensemble predictions for all four markets, writes ranked Excel output |
| 3 | `3_weekly_rd2_model_predictions.py` | After Round 2, loads the Rd2 model and generates updated Win probabilities using in-tournament position/lead features |

### Seasonal training (once per season)

| Script | What it does |
|--------|--------------|
| `seasonal_model_training.py` | Trains one ensemble per market (Winner / Top5 / Top10 / Top20). Optuna-tunes five base models, generates OOF predictions via RepeatedStratifiedKFold, then fits a LogisticRegression meta-model that incorporates implied market odds as a calibration signal. Saves `.pkl` packages to `Output/Models/`. |
| `seasonal_rd2_model_training.py` | Same approach for the Rd2 model, trained on Round 2 in-tournament features (position, lead, average position, pre-tournament model score). Meta-model is trained on base model OOF scores only — no implied odds. |

### Backtesting

| Script | What it does |
|--------|--------------|
| `walk_forward_backtest.py` | Rolling 2-year train → 1-year test windows across all available data. Trains from scratch per window (warm-starts between adjacent windows). P&L uses Betfair Lay odds with dead-heat adjustment for place markets. |

---

## Model architecture

### Base models (five per market)
- **LogisticRegression** (ElasticNet) — replaces GAM
- **RandomForestClassifier**
- **LGBMClassifier** — replaces GBM
- **XGBClassifier**
- **LGBMClassifier (DART boosting)** — replaces ANN

All models use `class_weight="balanced"` / `scale_pos_weight` to handle the strong class imbalance in golf (e.g. ~0.8% win rate).

### Features
Base models use pure player-skill and course-fit signals only — odds are deliberately excluded from all base model feature sets. Strokes Gained categories, historical finishing rates, course/location history, and field-context metrics are the primary signals. See `config.py → BASE_MODEL_VARS` for the full list.

### Meta-model
A `LogisticRegression` is trained on the OOF predictions from the five base models. For standard markets it also receives implied probability (1/odds) as an additional input, combining player-skill scores with market consensus in a single calibration step. Rd2 models omit the odds feature.

### Hyperparameter tuning
Optuna tunes each base model independently per market per season. Best parameters are saved as `.pkl` files and used as warm-starts the following season to reduce tuning time.

---

## Configuration (`config.py`)

Key settings:

| Constant | Default | Description |
|----------|---------|-------------|
| `SEASON_SUFFIX` | `"S26"` | Label appended to saved model filenames |
| `TRAINING_YEARS` | `2` | Rolling window of complete seasons used for training |
| `OPTUNA_TRIALS` | `75` | Optuna trials per base model per market |
| `N_CV_SPLITS` | `5` | Folds for RepeatedStratifiedKFold |
| `N_CV_REPEATS` | `5` | Repeats → 25 OOF fits per model |
| `RANDOM_SEED` | `42` | Global reproducibility seed |

Tours and file paths are defined in `TOUR_CONFIG` and `TOUR_PREPROCESSING`.

---

## Running the pipeline

```bash
cd Weekly_Modelling_Python/Script

# One-off: train models for the current season
python seasonal_model_training.py
python seasonal_rd2_model_training.py

# Weekly: preprocess → predict → (after Rd2) update
python 1_weekly_data_preprocessing.py
python 2_weekly_model_predictions.py
python 3_weekly_rd2_model_predictions.py

# Backtesting
python walk_forward_backtest.py
```

Scripts can also be run from any working directory — `config.py` resolves paths relative to its own location, including Jupyter/IPython environments.
