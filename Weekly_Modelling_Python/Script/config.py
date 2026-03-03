"""
Golf Prediction Models - Python Pipeline Configuration
Mirrors Weekly_Modelling R pipeline structure.

Path conventions:
  SHARED_INPUT_DIR  - raw data files shared with R pipeline (read-only)
  INPUT_DIR         - Python-processed files (Weekly_Modelling_Python/Input/)
  MODELS_DIR        - trained model .pkl files
  PREDICTIONS_DIR   - weekly prediction Excel files
"""

from pathlib import Path

# ===== PATHS =====
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent                                        # Weekly_Modelling_Python/
SHARED_INPUT_DIR = BASE_DIR.parent / "Weekly_Modelling" / "Input"  # Raw data shared with R
INPUT_DIR = BASE_DIR / "Input"                                      # Python-processed data
MODELS_DIR = BASE_DIR / "Output" / "Models"
PREDICTIONS_DIR = BASE_DIR / "Output" / "Predictions"

# ===== SEASON =====
SEASON_SUFFIX = "S26"
TRAINING_YEARS = 2   # number of complete calendar years to train on

# ===== REPRODUCIBILITY =====
RANDOM_SEED = 42

# ===== OPTUNA =====
OPTUNA_TRIALS = 75   # per model per market; warm-starts from saved best params
N_CV_SPLITS = 5      # folds for RepeatedStratifiedKFold OOF
N_CV_REPEATS = 5     # repeats  → 25 OOF fits per model per market

# ===== BETTING MARKETS =====
BETTING_MARKETS = {
    "Winner": {
        "target_col": "win",
        "odds_col": "Win_odds",
        "model_odds_cols": ["Win_odds", "Top5_odds"],
        "market_size": 1,
    },
    "Top5": {
        "target_col": "top_5",
        "odds_col": "Top5_odds",
        "model_odds_cols": ["Top5_odds", "Top10_odds"],
        "market_size": 5,
    },
    "Top10": {
        "target_col": "top_10",
        "odds_col": "Top10_odds",
        "model_odds_cols": ["Top10_odds", "Top20_odds"],
        "market_size": 10,
    },
    "Top20": {
        "target_col": "top_20",
        "odds_col": "Top20_odds",
        "model_odds_cols": ["Top20_odds"],
        "market_size": 20,
    },
}

# ===== BASE MODEL VARIABLES =====
# Mirrors Seasonal_Model_Training.R base_model_vars
BASE_MODEL_VARS = [
    "rating_vs_field_best",
    "rating",
    "rating_vs_field_worst",
    "yr3_All",
    "Top5_rank",
    "compat2",
    "sgp_field_zscore",
    "sgtee_field_zscore",
    "course",
    "sgatg_vs_field_median",
    "sgapp_vs_field_median",
    "Starts_Not10",
    "current",
    "location",
    "field",
]

# ===== RD2 MODEL VARIABLES =====
RD2_MODEL_VARS = [
    "Rd2Pos",
    "Rd2Lead",
    "AvPosn",
    "Top5",
    "GLM_Odds_Probability_Median",
    "Model_Score_Median",
]

# ===== COLUMNS TO DROP FROM HISTORICAL DATA =====
# Present in raw Excel but not needed for modelling
HISTORICAL_ONLY_COLS = [
    "Lay_odds", "Top40_odds", "EW_Profit", "Top5_Profit",
    "Top10_Profit", "Top20_Profit", "Top40_Profit",
    "Lay_top5", "Lay_top10", "Lay_top20",
    "Rd2Pos", "Rd2Lead", "Betfair_rd2",
]

# ===== TOUR CONFIGURATION =====
TOUR_CONFIG = {
    "PGA": {
        "name": "PGA Tour",
        "historical_file": INPUT_DIR / "PGA_Processed.xlsx",
        "weekly_file": INPUT_DIR / "This_Week_PGA_Processed.xlsx",
        "rd2_predictions_file": SHARED_INPUT_DIR / "Full_PGA_Historical_Predictions.xlsx",
        "rd2_raw_file": SHARED_INPUT_DIR / "PGA.xlsx",
        "rd2_weekly_file": SHARED_INPUT_DIR / "This_Week_Rd2_PGA.csv",
    },
    "Euro": {
        "name": "European Tour",
        "historical_file": INPUT_DIR / "Euro_Processed.xlsx",
        "weekly_file": INPUT_DIR / "This_Week_Euro_Processed.xlsx",
        "rd2_predictions_file": SHARED_INPUT_DIR / "Full_Euro_Historical_Predictions.xlsx",
        "rd2_raw_file": SHARED_INPUT_DIR / "Euro.xlsx",
        "rd2_weekly_file": SHARED_INPUT_DIR / "This_Week_Rd2_Euro.csv",
    },
}

# ===== PREPROCESSING TOUR CONFIGURATION =====
TOUR_PREPROCESSING = {
    "pga": {
        "name": "PGA Tour",
        "historical_input": SHARED_INPUT_DIR / "PGA.xlsx",
        "weekly_input": SHARED_INPUT_DIR / "This_Week_PGA.csv",
        "historical_output": INPUT_DIR / "PGA_Processed.xlsx",
        "weekly_output": INPUT_DIR / "This_Week_PGA_Processed.xlsx",
    },
    "euro": {
        "name": "European Tour",
        "historical_input": SHARED_INPUT_DIR / "Euro.xlsx",
        "weekly_input": SHARED_INPUT_DIR / "This_Week_Euro.csv",
        "historical_output": INPUT_DIR / "Euro_Processed.xlsx",
        "weekly_output": INPUT_DIR / "This_Week_Euro_Processed.xlsx",
    },
}
