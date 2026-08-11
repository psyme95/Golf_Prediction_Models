"""Golf prediction pipeline configuration — paths, markets, features, tours."""

from pathlib import Path

# ===== PATHS =====
PIPELINE_DIR = Path(__file__).resolve().parent.parent          # pipeline/
REPO_DIR     = PIPELINE_DIR.parent                             # D:/Golf/Repo

RAW_DIR         = PIPELINE_DIR / "Input"
INPUT_DIR       = PIPELINE_DIR / "Input"                       # processed files
MODELS_DIR      = PIPELINE_DIR / "Output" / "Models"
PREDICTIONS_DIR = PIPELINE_DIR / "Output" / "Predictions"
BACKTESTS_DIR   = PIPELINE_DIR / "Output" / "Backtests"
LOGS_DIR        = PIPELINE_DIR / "Output" / "Logs"

# ===== SEASON / TRAINING =====
SEASON_SUFFIX  = "S26"
TRAINING_YEARS = 2      # complete calendar years of training data
RANDOM_SEED    = 42

# ===== OPTUNA / CV =====
OPTUNA_TRIALS = 75      # production trials per model per market
N_CV_SPLITS   = 5
N_CV_REPEATS  = 5       # repeated grouped CV → up to 25 OOF fits per model

# ===== BETTING MARKETS =====
MARKETS = {
    "Winner": {"target_col": "win",    "odds_col": "Win_odds",   "market_size": 1},
    "Top5":   {"target_col": "top_5",  "odds_col": "Top5_odds",  "market_size": 5},
    "Top10":  {"target_col": "top_10", "odds_col": "Top10_odds", "market_size": 10},
    "Top20":  {"target_col": "top_20", "odds_col": "Top20_odds", "market_size": 20},
}

# Betfair lay odds column per market (pre-event snapshots; joined from raw file).
LAY_ODDS_COLS = {
    "Winner": "Lay_odds",
    "Top5":   "Lay_top5",
    "Top10":  "Lay_top10",
    "Top20":  "Lay_top20",
}

# Cut position per place market for dead-heat calculation (Winner: none).
PLACE_MARKET_CUTS = {"Top5": 5, "Top10": 10, "Top20": 20}

# ===== BETTING STAKES =====
BACK_STAKE = 10.0                       # £ per back bet
LAY_FIXED_LIABILITY = {"Winner": 1000.0, "Top5": 200.0, "Top10": 100.0, "Top20": 50.0}
LAY_FIXED_STAKE     = {"Winner": 1.0,    "Top5": 5.0,   "Top10": 10.0,  "Top20": 20.0}

# ===== COMMISSION =====
COMMISSION = 0.03    # Betfair commission, applied to all winning bets

# ===== KELLY STAKING =====
# Extremely conservative fractional Kelly, compared against fixed staking.
# Non-compounding: stakes sized from a fixed notional bankroll so P&L stays
# comparable across strategies and independent of bet ordering.
KELLY_FRACTION = 0.10     # 1/10 Kelly
KELLY_BANKROLL = 1000.0   # £ notional bankroll
KELLY_MAX_RISK = 0.01     # cap stake/liability at 1% of bankroll per bet

# ===== STRATEGY GRID =====
GRID_RATING_THRESHOLDS = [55, 60, 65, 70, 75]   # floor and ceiling sweeps

# Minimum-edge sweep: model probability must exceed threshold × implied probability.
EDGE_THRESHOLDS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]

# Lay trigger for weekly predictions: lay while available lay odds <= this
# threshold / model probability. 0.87 chosen from walk-forward grid evidence
# (raw-edge sweep, consistent 4-5/5 years across both tours on place markets).
LAY_EDGE_TRIGGER = 0.87

# Per-market lay-odds bands (floor and ceiling sweeps). Winner lay odds run
# into the hundreds (up to ~1000); place markets are much shorter.
ODDS_GRID = {
    "Winner": [5, 10, 20, 50, 100, 200, 500, 1000],
    "Top5":   [2, 5, 10, 20, 50, 100, 200],
    "Top10":  [2, 3, 5, 10, 20, 50, 100],
    "Top20":  [1.5, 2, 3, 5, 10, 20, 50],
}

# ===== BASE MODEL FEATURES =====
# Pure player-skill / course-fit signals. Odds are never base-model features;
# whether implied odds enter the meta-learner is per-tour (use_meta_odds).
BASE_MODEL_VARS = [
    "rating_vs_field_best",
    "rating",
    "yr3_All",
    "X_1yr",
    "X_6m",
    "lastweek",
    "current",
    "Top5_rank",
    "Starts_Not10",
    "compat",
    "compat2",
    "course",
    "course_top5",
    "course_top20",
    "location",
    "location_top5",
    "location_top20",
    "field",
    "field_strength",
    "field_depth",
    "sgtee_field_zscore",
    "sgt2g_field_zscore",
    "sgapp_field_zscore",
    "sgatg_vs_field_median",
    "sgp_field_zscore",
    "sg_ball_striking_field_zscore",
    "sg_short_game_field_zscore",
]

# Raw columns excluded from processed feature files (betting results / odds
# artefacts, not model features). Lay odds are re-joined at backtest time.
DROP_COLS = [
    "Lay_odds", "Top40_odds", "EW_Profit", "Top5_Profit",
    "Top10_Profit", "Top20_Profit", "Top40_Profit",
    "Lay_top5", "Lay_top10", "Lay_top20",
    "Rd2Pos", "Rd2Lead", "Betfair_rd2",
]

# ===== TOURS =====
TOURS = {
    "PGA": {
        "name": "PGA Tour",
        "raw_historical":      RAW_DIR / "PGA.xlsx",
        "raw_weekly":          RAW_DIR / "This_Week_PGA.csv",
        "processed_historical": INPUT_DIR / "PGA_Processed.xlsx",
        "processed_weekly":     INPUT_DIR / "This_Week_PGA_Processed.xlsx",
        "use_meta_odds": True,   # residual modelling: market anchor in the meta-learner
    },
    "Euro": {
        "name": "European Tour",
        "raw_historical":      RAW_DIR / "Euro.xlsx",
        "raw_weekly":          RAW_DIR / "This_Week_Euro.csv",
        "processed_historical": INPUT_DIR / "Euro_Processed.xlsx",
        "processed_weekly":     INPUT_DIR / "This_Week_Euro_Processed.xlsx",
        "use_meta_odds": True,
    },
}