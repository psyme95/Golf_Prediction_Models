"""
Backtesting Script
Evaluates the trained model on out-of-sample events for a given year.

The processed historical data contains both training-era events (2024-2025)
and current-year events (2026). The model was trained on 2024-2025, so 2026
events are a clean out-of-sample test.

For each event the script:
  1. Extracts the field (all players in that event from the processed data)
  2. Applies the trained model exactly as the weekly prediction script does
  3. Compares predictions against actual outcomes across three betting strategies

Betting strategies
------------------
Back only
  Bet when Normalised_Model_Odds < Market_Odds (model thinks player underpriced).
  Fixed stake: £10. P&L = (market_odds - 1) × 10 on win, -10 on loss.
  ROI = total_P&L / (n_bets × 10)

Lay only
  Bet when Normalised_Model_Odds > Market_Odds (model thinks player overpriced).
  Lay odds = 1.2 × market_back_odds.
  Liability per bet = £1000 / market_size:
    Winner £1000 | Top5 £200 | Top10 £100 | Top20 £50
  Maximum possible loss per event per market = £1000 (at most market_size
  players can finish in-the-money, each costing exactly liability_per_bet).
  Stake = liability / (lay_odds - 1).
  P&L = +stake if player does NOT finish in market, -liability if they do.
  ROI = total_P&L / total_liability_fronted

Combination
  Both strategies applied simultaneously. Reports absolute £ P&L only.

Model quality metrics: AUC, Average Precision, TSS, Log-loss, Brier score

Output Excel (one per tour):
  Summary          — model metrics + all three strategy results by market
  Event_Results    — per-event P&L with running cumulative columns
  All_Predictions  — every player prediction + outcome + per-bet P&L
  Calibration      — binned predicted vs actual rates for calibration plots

Run: python backtest.py
     python backtest.py --year 2025
     python backtest.py --tour PGA
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASE_MODEL_VARS,
    BETTING_MARKETS,
    MODELS_DIR,
    PREDICTIONS_DIR,
    SEASON_SUFFIX,
    TOUR_CONFIG,
    TRAINING_YEARS,
)
from seasonal_model_training import tss_optimal

warnings.filterwarnings("ignore")


# ===== BETTING CONSTANTS =====
BACK_STAKE          = 10.0     # £ per back bet
LAY_TOTAL_LIABILITY = 1000.0   # £ max possible loss per market per event
LAY_ODDS_MULTIPLIER = 1.2      # exchange spread on lay side


# ===== ARGS =====

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest golf prediction models")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to backtest (default: auto-detect from data)")
    parser.add_argument("--tour", type=str, default=None,
                        help="Tour to backtest: PGA or Euro (default: both)")
    return parser.parse_args()


def get_backtest_year(df: pd.DataFrame) -> int:
    max_year = int(df["Date"].dt.year.max())
    if (df["Date"].dt.year == max_year).sum() > 0:
        return max_year
    return max_year - 1


# ===== PREDICTION (mirrors 2_weekly_model_predictions.py) =====

def predict_event(event_df: pd.DataFrame, market_name: str,
                  market_pkg: dict) -> pd.DataFrame | None:
    model_vars = market_pkg["model_vars"]
    odds_col   = market_pkg["odds_col"]

    available = [v for v in model_vars if v in event_df.columns]
    df = event_df[available + [odds_col]].copy()
    for col in available + [odds_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    if len(df) == 0:
        return None

    X    = df[available].values.astype(float)
    odds = df[odds_col].values.astype(float)

    model_preds = np.column_stack([
        m.predict_proba(X)[:, 1] for m in market_pkg["models"].values()
    ])
    raw_score = model_preds.mean(axis=1)

    implied_odds  = 1.0 / np.clip(odds, 1e-8, None)
    meta_X        = np.column_stack([model_preds, implied_odds])
    meta_X_scaled = market_pkg["meta_scaler"].transform(meta_X)
    proba         = market_pkg["meta_model"].predict_proba(meta_X_scaled)[:, 1]

    market_size = market_pkg["market_size"]
    prob_sum    = proba.sum()
    norm_prob   = (proba / prob_sum) * market_size if prob_sum > 0 else proba

    result = event_df.loc[df.index].copy()
    result["Model_Score"]            = raw_score.round(5)
    result["Probability"]            = proba.round(6)
    result["Normalised_Probability"] = norm_prob.round(6)
    result["Normalised_Model_Odds"]  = (1.0 / norm_prob.clip(1e-8)).round(2)
    return result


# ===== BETTING STRATEGIES =====

def apply_back_strategy(df: pd.DataFrame, odds_col: str,
                        target_col: str) -> pd.DataFrame:
    """
    Back when Normalised_Model_Odds < Market_Odds.
    Fixed stake BACK_STAKE per bet.
    """
    df = df.copy()
    mkt    = df[odds_col].to_numpy(dtype=float)
    actual = df[target_col].to_numpy(dtype=float)
    model  = df["Normalised_Model_Odds"].to_numpy(dtype=float)
    is_back = model < mkt
    is_win  = actual == 1
    df["Back_Bet"] = is_back
    df["Back_PnL"] = np.where(
        is_back & is_win,  (mkt - 1) * BACK_STAKE,
        np.where(is_back, -BACK_STAKE, 0.0)
    )
    return df


def apply_lay_strategy(df: pd.DataFrame, odds_col: str,
                       target_col: str, market_size: int) -> pd.DataFrame:
    """
    Lay when Normalised_Model_Odds > Market_Odds.
    Lay odds = LAY_ODDS_MULTIPLIER × market_back_odds.
    Liability per bet = LAY_TOTAL_LIABILITY / market_size.
    Stake = liability / (lay_odds - 1).
    P&L = +stake if player misses market, -liability if player hits market.

    Maximum total loss per event = market_size × liability_per_bet
                                  = LAY_TOTAL_LIABILITY (£1000).
    """
    df = df.copy()
    mkt    = df[odds_col].to_numpy(dtype=float)
    actual = df[target_col].to_numpy(dtype=float)
    model  = df["Normalised_Model_Odds"].to_numpy(dtype=float)
    liability_per_bet = LAY_TOTAL_LIABILITY / market_size
    lay_odds  = LAY_ODDS_MULTIPLIER * mkt
    lay_stake = liability_per_bet / np.maximum(lay_odds - 1, 1e-8)
    is_lay = model > mkt
    is_win = actual == 1
    df["Lay_Bet"]       = is_lay
    df["Lay_Liability"] = np.where(is_lay, liability_per_bet, 0.0)
    df["Lay_Stake"]     = np.round(np.where(is_lay, lay_stake, 0.0), 4)
    df["Lay_PnL"]       = np.where(
        is_lay & ~is_win, np.where(is_lay, lay_stake, 0.0),
        np.where(is_lay & is_win, -liability_per_bet, 0.0)
    )
    return df


# ===== STRATEGY SUMMARY HELPERS =====

def back_summary(df: pd.DataFrame, target_col: str) -> dict:
    bets = df[df["Back_Bet"]]
    if len(bets) == 0:
        return {"Back_NBets": 0, "Back_NWon": 0, "Back_HitRate": np.nan,
                "Back_PnL": 0.0, "Back_ROI": np.nan}
    n_bets = len(bets)
    n_won  = int(bets[target_col].sum())
    pnl    = float(bets["Back_PnL"].sum())
    return {
        "Back_NBets":    n_bets,
        "Back_NWon":     n_won,
        "Back_HitRate":  round(n_won / n_bets, 4),
        "Back_PnL":      round(pnl, 2),
        "Back_ROI":      round(pnl / (n_bets * BACK_STAKE), 4),
    }


def lay_summary(df: pd.DataFrame, target_col: str) -> dict:
    bets = df[df["Lay_Bet"]]
    if len(bets) == 0:
        return {"Lay_NBets": 0, "Lay_NLost": 0, "Lay_PnL": 0.0,
                "Lay_ROI_On_Liability": np.nan}
    n_bets        = len(bets)
    n_lost        = int(bets[target_col].sum())   # player hit market = lay loss
    pnl           = float(bets["Lay_PnL"].sum())
    total_liab    = float(bets["Lay_Liability"].sum())
    return {
        "Lay_NBets":           n_bets,
        "Lay_NLost":           n_lost,   # number of lay bets that lost
        "Lay_PnL":             round(pnl, 2),
        "Lay_ROI_On_Liability": round(pnl / total_liab, 4) if total_liab > 0 else np.nan,
    }


# ===== MODEL QUALITY METRICS =====

def compute_discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if y_true.sum() == 0:
        return {"AUC": np.nan, "Avg_Precision": np.nan, "TSS": np.nan,
                "Log_Loss": np.nan, "Brier": np.nan}
    y_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "AUC":           round(roc_auc_score(y_true, y_prob), 4),
        "Avg_Precision": round(average_precision_score(y_true, y_prob), 4),
        "TSS":           round(tss_optimal(y_true, y_prob), 4),
        "Log_Loss":      round(log_loss(y_true, y_clip), 5),
        "Brier":         round(brier_score_loss(y_true, y_prob), 5),
    }


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "Bin_Low":        round(lo, 2),
            "Bin_High":       round(hi, 2),
            "N":              int(mask.sum()),
            "Mean_Predicted": round(float(y_prob[mask].mean()), 4),
            "Actual_Rate":    round(float(y_true[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


# ===== MAIN BACKTEST LOOP =====

def run_backtest(tour_key: str, tour_info: dict, backtest_year: int):
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {tour_info['name']}  |  Year: {backtest_year}")
    print(f"{'='*60}")

    hist_path  = tour_info["historical_file"]
    model_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"

    if not hist_path.exists():
        print(f"  Historical file not found: {hist_path}"); return None
    if not model_path.exists():
        print(f"  Model file not found: {model_path}"); return None

    df      = pd.read_excel(hist_path)
    df["Date"] = pd.to_datetime(df["Date"])
    package = joblib.load(model_path)

    backtest_df = df[df["Date"].dt.year == backtest_year].copy()
    if len(backtest_df) == 0:
        print(f"  No data found for {backtest_year}"); return None

    events = backtest_df["eventID"].unique()
    print(f"  Events: {len(events)}  |  Players: {len(backtest_df):,}")

    all_predictions = []
    event_summaries = []

    for event_id in sorted(events):
        event_df   = backtest_df[backtest_df["eventID"] == event_id].copy()
        event_date = event_df["Date"].iloc[0].strftime("%Y-%m-%d")

        for market_name, market_pkg in package["markets"].items():
            market_config = BETTING_MARKETS[market_name]
            target_col    = market_config["target_col"]
            odds_col      = market_config["odds_col"]
            market_size   = market_config["market_size"]

            if target_col not in event_df.columns:
                continue

            preds = predict_event(event_df, market_name, market_pkg)
            if preds is None or len(preds) == 0:
                continue

            preds["Actual"] = preds[target_col].astype(int)

            # Apply both strategies
            preds = apply_back_strategy(preds, odds_col, target_col)
            preds = apply_lay_strategy( preds, odds_col, target_col, market_size)

            # Combination P&L
            preds["Combo_PnL"] = preds["Back_PnL"] + preds["Lay_PnL"]

            # Per-event summary
            bs = back_summary(preds, target_col)
            ls = lay_summary( preds, target_col)
            event_summaries.append({
                "Tour":      tour_key,
                "EventID":   event_id,
                "Date":      event_date,
                "Market":    market_name,
                "FieldSize": len(preds),
                "Positives": int(preds["Actual"].sum()),
                **bs,
                **ls,
                "Combo_PnL": round(float(preds["Combo_PnL"].sum()), 2),
            })

            preds["Tour"]    = tour_key
            preds["EventID"] = event_id
            preds["Date"]    = event_date
            preds["Market"]  = market_name
            all_predictions.append(preds)

    if not all_predictions:
        print("  No predictions generated."); return None

    pred_df      = pd.concat(all_predictions, ignore_index=True)
    event_sum_df = pd.DataFrame(event_summaries).sort_values(["Market", "Date", "EventID"])

    # Cumulative P&L per market, sorted by date
    for market_name in event_sum_df["Market"].unique():
        mask = event_sum_df["Market"] == market_name
        for col, cum_col in [("Back_PnL",  "Back_Cumulative_PnL"),
                              ("Lay_PnL",   "Lay_Cumulative_PnL"),
                              ("Combo_PnL", "Combo_Cumulative_PnL")]:
            event_sum_df.loc[mask, cum_col] = (
                event_sum_df.loc[mask, col].cumsum().round(2)
            )

    # ===== AGGREGATE METRICS =====
    summary_rows = []
    calib_sheets = {}

    for market_name, market_config in BETTING_MARKETS.items():
        target_col  = market_config["target_col"]
        odds_col    = market_config["odds_col"]
        market_size = market_config["market_size"]

        mdf = pred_df[pred_df["Market"] == market_name]
        if len(mdf) == 0 or target_col not in mdf.columns:
            continue

        y_true = mdf["Actual"].values
        y_prob = mdf["Normalised_Probability"].values

        disc  = compute_discrimination(y_true, y_prob)
        bs    = back_summary(mdf, target_col)
        ls    = lay_summary( mdf, target_col)
        combo = round(float(mdf["Combo_PnL"].sum()), 2)
        calib_sheets[market_name] = calibration_bins(y_true, y_prob)

        n_events   = int(mdf["EventID"].nunique())
        prevalence = round(float(y_true.mean()), 4)
        liability  = LAY_TOTAL_LIABILITY / market_size

        print(f"\n  {market_name} ({n_events} events, prevalence={prevalence:.1%}, "
              f"lay liability/bet=£{liability:.0f})")
        print(f"    Model:  AUC={disc['AUC']:.4f}  AP={disc['Avg_Precision']:.4f}  "
              f"TSS={disc['TSS']:.4f}  log_loss={disc['Log_Loss']:.4f}")
        print(f"    Back:   {bs['Back_NBets']} bets  won={bs['Back_NWon']}  "
              f"P&L=£{bs['Back_PnL']:.2f}  ROI={bs['Back_ROI']:.1%}")
        print(f"    Lay:    {ls['Lay_NBets']} bets  lost={ls['Lay_NLost']}  "
              f"P&L=£{ls['Lay_PnL']:.2f}  ROI={ls['Lay_ROI_On_Liability']:.1%}")
        print(f"    Combo:  P&L=£{combo:.2f}")

        summary_rows.append({
            "Tour":        tour_key,
            "Year":        backtest_year,
            "Market":      market_name,
            "N_Events":    n_events,
            "N_Players":   len(mdf),
            "Prevalence":  prevalence,
            "AP_Baseline": prevalence,
            **disc,
            **bs,
            **ls,
            "Combo_PnL":   combo,
        })

    return {
        "summary":         pd.DataFrame(summary_rows),
        "event_results":   event_sum_df,
        "all_predictions": pred_df,
        "calibration":     calib_sheets,
    }


# ===== EXPORT =====

def export_results(results: dict, tour_key: str, backtest_year: int):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{tour_key}_Backtest_{backtest_year}.xlsx"

    # All_Predictions column order
    id_cols   = [c for c in ["Date", "EventID", "Market", "surname", "firstname",
                              "posn", "rating"] if c in results["all_predictions"].columns]
    odds_cols = [c for c in ["Win_odds", "Top5_odds", "Top10_odds", "Top20_odds"]
                 if c in results["all_predictions"].columns]
    pred_cols = ["Model_Score", "Probability", "Normalised_Probability",
                 "Normalised_Model_Odds", "Actual"]
    bet_cols  = ["Back_Bet", "Back_PnL",
                 "Lay_Bet", "Lay_Stake", "Lay_Liability", "Lay_PnL",
                 "Combo_PnL"]
    all_cols  = id_cols + odds_cols + pred_cols + bet_cols
    export_df = results["all_predictions"][
        [c for c in all_cols if c in results["all_predictions"].columns]
    ]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results["summary"].to_excel(      writer, sheet_name="Summary",         index=False)
        results["event_results"].to_excel(writer, sheet_name="Event_Results",   index=False)
        export_df.to_excel(               writer, sheet_name="All_Predictions", index=False)
        for market_name, calib_df in results["calibration"].items():
            calib_df.to_excel(writer, sheet_name=f"Calib_{market_name}"[:31], index=False)

    print(f"\n  Results saved: {out_path.name}")
    return out_path


# ===== MAIN =====

def main():
    args = parse_args()
    print("=== GOLF MODEL BACKTESTING ===")
    print(f"Season: {SEASON_SUFFIX}  |  Training years: {TRAINING_YEARS}")
    print(f"Back stake: £{BACK_STAKE}  |  Lay total liability: £{LAY_TOTAL_LIABILITY}  "
          f"|  Lay odds multiplier: {LAY_ODDS_MULTIPLIER}×")

    tours_to_run = {k: v for k, v in TOUR_CONFIG.items()
                    if args.tour is None or k == args.tour}

    for tour_key, tour_info in tours_to_run.items():
        try:
            if args.year:
                backtest_year = args.year
            else:
                hist_path = tour_info["historical_file"]
                if not hist_path.exists():
                    print(f"\nHistorical file not found: {hist_path}"); continue
                df_tmp = pd.read_excel(hist_path, usecols=["Date"])
                df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])
                backtest_year = get_backtest_year(df_tmp)

            print(f"\nBacktest year: {backtest_year}")
            results = run_backtest(tour_key, tour_info, backtest_year)
            if results:
                export_results(results, tour_key, backtest_year)

        except Exception as e:
            print(f"\nError backtesting {tour_key}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== BACKTESTING COMPLETE ===")


if __name__ == "__main__":
    main()
