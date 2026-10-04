"""Paths, markets, features and per-tour settings."""

from pathlib import Path

# ===== PATHS =====
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR         = ROOT_DIR / "Input"
INPUT_DIR       = ROOT_DIR / "Input"                       # processed files
OUTPUT_DIR      = ROOT_DIR / "Output"
MODELS_DIR      = OUTPUT_DIR / "Models"
PREDICTIONS_DIR = OUTPUT_DIR / "Predictions"
BACKTESTS_DIR   = OUTPUT_DIR / "Backtests"
PAPER_DIR       = OUTPUT_DIR / "Paper_Testing"
LOGS_DIR        = OUTPUT_DIR / "Logs"

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

# The data only has exchange lay prices, so back bets are priced at a spread
# below the lay quote. A single tick is already 2.4-4.6% of the price (see
# analysis/back_screen.py), so 0.05 is an optimistic assumption. It is also flat
# across markets and prices, whereas thin longshot markets are wider; this
# should be estimated from paired back/lay quotes once they are collected.
BACK_SPREAD = 0.05
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

# Top-K-within-event sweep: keep only the K strongest-edge bets in each event,
# ranked against that event's own field. Unlike a global edge threshold (which
# concentrates bets into a handful of events), this holds bets-per-event fixed,
# so within-event diversification is preserved while capital per event falls.
GRID_TOPK = [5, 10, 20, 30, 40, 60]

# Lay trigger: lay while the available lay odds sit below Normalised_Model_Odds.

# Per-market lay-odds bands (floor and ceiling sweeps). Winner lay odds run
# into the hundreds (up to ~1000); place markets are much shorter.
ODDS_GRID = {
    "Winner": [5, 10, 20, 50, 100, 200, 500, 1000],
    "Top5":   [2, 5, 10, 20, 50, 100, 200],
    "Top10":  [2, 3, 5, 10, 20, 50, 100],
    "Top20":  [1.5, 2, 3, 5, 10, 20, 50],
}

# ===== BACK-SHAPED TUNING OBJECTIVE =====
# Most rows are longshots, so plain log-loss spends its effort there, while back
# bets are only ever struck at short prices. objective="shortprice" weights each
# row by (1/odds)^BACK_OBJ_POWER, normalised to mean 1.
BACK_OBJ_POWER = 0.5

# ===== CROSS-MARKET FEATURES =====
# The bookmakers' markets are not always consistent with each other (e.g. the
# Top 10 price can predict top-20 finishes better than the Top 20 price does),
# so de-vigged prices from every market are offered as features.
#
# Top40 is excluded because the weekly files rarely have Top40 prices, so the
# feature would exist in training but not at prediction time.
CROSS_MARKET_ODDS = {"win": "Win_odds", "top5": "Top5_odds",
                     "top10": "Top10_odds", "top20": "Top20_odds"}
CROSS_MARKET_SIZE = {"win": 1, "top5": 5, "top10": 10, "top20": 20}

# An event priced for only part of its field would have every priced player's
# de-vigged probability inflated by 1/coverage, so such events are nulled.
MKT_MIN_COVERAGE = 0.9

# Market-implied shape of a player's finish distribution. Ratios of adjacent
# markets show whether a player is priced as "high ceiling" or "steady".
CROSS_MARKET_VARS = [
    "mkt_win_logit", "mkt_top5_logit", "mkt_top10_logit", "mkt_top20_logit",
    "mkt_win_vs_top5", "mkt_top5_vs_top10", "mkt_top10_vs_top20",
    "mkt_top10_logit_field_zscore", "mkt_top20_logit_field_zscore",
]

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
    "Lay_odds", "Lay_top5", "Lay_top10", "Lay_top20",
    "Rd2Pos", "Rd2Lead", "Betfair_rd2",
]
# The *_Profit settlement columns are kept, as they are the only record of the
# dead-heat-adjusted settled fraction (see FRAC_LABEL_COLS).

# Dead-heat-adjusted labels. A bet returns stake*frac*(odds-1) and loses
# stake*(1-frac), so profit = stake*(frac*odds - 1)  ->  frac = (profit/stake + 1)/odds.
# frac is 1 for a clean win, 0 for a loss, and e.g. 2/6 for a 6-way tie straddling
# the cut. The binary `posn <= k` label counts every tied player as a full winner.
# Maps market -> (binary target, profit column, odds column, fractional target).
FRAC_STAKE = 10.0     # the *_Profit columns are settled on a £10 stake
FRAC_LABEL_COLS = {
    "Top5":  ("top_5",  "Top5_Profit",  "Top5_odds",  "frac_top_5"),
    "Top10": ("top_10", "Top10_Profit", "Top10_odds", "frac_top_10"),
    "Top20": ("top_20", "Top20_Profit", "Top20_odds", "frac_top_20"),
    "Top40": ("top_40", "Top40_Profit", "Top40_odds", "frac_top_40"),
}
# The Winner market keeps the binary `win` label: ties for first go to a playoff,
# so there is no partial payout to recover.

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