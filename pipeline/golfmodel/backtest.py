"""Shared backtest core + static (current-season) and walk-forward drivers.

Fixes baked in relative to the old pipeline:
  - Dead-heat counts come from the FULL event field (every row of the event,
    before any feature filtering), not from the filtered prediction rows.
  - Discrimination/calibration metrics use raw Probability only; the
    normalised probability (sums to market_size per event) is for ranking
    and bet selection, never for calibration metrics.
  - Lay ROI = P&L / total liability everywhere (summaries and grids).
  - use_meta_odds flows from TOURS config into walk-forward training, so the
    backtest evaluates the same architecture as production.

Profit-improvement layer:
  - 3% commission (COMMISSION) deducted from every winning P&L component.
  - Dual edge basis per row (raw calibrated probability vs field-normalised)
    with strategy grids sweeping edge thresholds, odds bands, and rating
    filters on both bases, including year-by-year consistency columns.
  - Conservative fractional-Kelly staking computed alongside fixed staking.
  - Headline bet flags remain the normalised zero-margin basis until the
    grid evidence supports switching (policy decision is human-led).
"""

import shutil
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

from .config import (
    BACK_STAKE,
    BACKTESTS_DIR,
    COMMISSION,
    EDGE_THRESHOLDS,
    GRID_RATING_THRESHOLDS,
    KELLY_BANKROLL,
    KELLY_FRACTION,
    KELLY_MAX_RISK,
    LAY_FIXED_LIABILITY,
    LAY_FIXED_STAKE,
    LAY_ODDS_COLS,
    MARKETS,
    MODELS_DIR,
    ODDS_GRID,
    PLACE_MARKET_CUTS,
    SEASON_SUFFIX,
    TOURS,
    TRAINING_YEARS,
)
from .data import join_lay_odds, load_processed_historical
from .excel_out import write_workbook
from .modeling import ensemble_predict, load_bundle, train_market, tss_optimal

WF_MODELS_DIR = BACKTESTS_DIR / "WalkForward_Models"
MIN_POSITIVES = 10   # skip a market window when training positives are fewer


# ===== DEAD HEAT =====

def dead_heat_info(event_full: pd.DataFrame, cut: int):
    """Dead-heat details for one market from the FULL event field.

    Returns (tied_player_ids, places_available, players_tied) when a genuine
    dead heat straddles the cut, else None. Only the group tied at the last
    qualifying position can be short of places.
    """
    posn = pd.to_numeric(event_full["posn"], errors="coerce")
    qual = posn[posn <= cut]
    if qual.empty:
        return None
    tied_posn    = qual.max()
    filled_above = int((posn < tied_posn).sum())
    available    = max(cut - filled_above, 0)
    tied_mask    = posn == tied_posn
    players_tied = int(tied_mask.sum())
    if players_tied > 1 and available < players_tied:
        return set(event_full.loc[tied_mask, "playerID"]), available, players_tied
    return None


def dead_heat_arrays(preds: pd.DataFrame, event_full: pd.DataFrame, market_name: str):
    """Per-row (rf, exp_winners, act_winners) aligned to preds, from the full field."""
    n   = len(preds)
    rf  = np.ones(n)
    exp = np.ones(n)
    act = np.ones(n)

    cut = PLACE_MARKET_CUTS.get(market_name)
    if cut is None or "posn" not in event_full.columns:
        return rf, exp, act

    info = dead_heat_info(event_full, cut)
    if info is None:
        return rf, exp, act

    tied_ids, available, players_tied = info
    mask = preds["playerID"].isin(tied_ids).to_numpy()
    rf[mask]  = np.clip(available / players_tied, 0.0, 1.0)
    exp[mask] = float(available)
    act[mask] = float(players_tied)
    return rf, exp, act


# ===== PREDICTION =====

def predict_event(event_full: pd.DataFrame, market_pkg: dict) -> pd.DataFrame | None:
    """Apply one market's ensemble to one event. Rows missing model inputs or
    market odds are excluded from predictions (but stay in the full field)."""
    model_vars = [v for v in market_pkg["model_vars"] if v in event_full.columns]
    odds_col   = market_pkg["odds_col"]
    if odds_col not in event_full.columns:
        return None

    df = event_full.copy()
    for col in model_vars + [odds_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=model_vars + [odds_col])
    if df.empty:
        return None

    X = df[model_vars].values.astype(float)
    odds_values = df[odds_col].values.astype(float) if market_pkg.get("meta_uses_odds") else None
    proba, raw_score = ensemble_predict(market_pkg, X, odds_values=odds_values)

    market_size = market_pkg["market_size"]
    prob_sum    = proba.sum()
    norm_prob   = (proba / prob_sum) * market_size if prob_sum > 0 else proba

    result = df.copy()
    result["Model_Score"]            = raw_score.round(5)
    result["Probability"]            = proba.round(6)
    result["Normalised_Probability"] = norm_prob.round(6)
    result["Normalised_Model_Odds"]  = (1.0 / np.clip(norm_prob, 1e-8, None)).round(2)
    return result


# ===== BETTING STRATEGIES =====

def _lay_return(lay, actual, exp_winners, act_winners, liability):
    """Dead-heat-aware lay return, gross of commission. Win (bad for layer):
    scaled loss; loss (good for layer): +stake. exp==act==1 → -liability/+stake."""
    stake      = liability / np.clip(lay - 1, 1e-8, None)
    win_return = ((1 - (exp_winners / np.clip(act_winners, 1e-8, None)) * lay)
                  / np.clip(lay - 1, 1e-8, None)) * liability
    return np.where(actual == 1, win_return, stake)


def _net(pnl: np.ndarray) -> np.ndarray:
    """Betfair commission: winning bets pay COMMISSION on the winnings."""
    return np.where(pnl > 0, pnl * (1 - COMMISSION), pnl)


def _kelly_back_stake(p: np.ndarray, lay: np.ndarray) -> np.ndarray:
    """Fractional-Kelly back stake from raw model probability, net of commission."""
    b  = np.clip((lay - 1) * (1 - COMMISSION), 1e-8, None)
    f  = (p * b - (1 - p)) / b
    return np.clip(KELLY_BANKROLL * KELLY_FRACTION * np.clip(f, 0, None),
                   0, KELLY_BANKROLL * KELLY_MAX_RISK)


def _kelly_lay_liability(p: np.ndarray, lay: np.ndarray) -> np.ndarray:
    """Fractional-Kelly lay liability: laying = backing the complement at
    effective odds, so b' = (1-c)/(odds-1) and the 'win' probability is 1-p."""
    b  = np.clip((1 - COMMISSION) / np.clip(lay - 1, 1e-8, None), 1e-8, None)
    f  = ((1 - p) * b - p) / b
    return np.clip(KELLY_BANKROLL * KELLY_FRACTION * np.clip(f, 0, None),
                   0, KELLY_BANKROLL * KELLY_MAX_RISK)


def apply_strategies(preds: pd.DataFrame, event_full: pd.DataFrame,
                     market_name: str, lay_odds_col: str,
                     target_col: str) -> pd.DataFrame:
    """Betting P&L for one event × market.

    Edge ratios (model vs market) are computed on two bases:
      Edge_Raw  = Probability × lay_odds          (raw calibrated probability)
      Edge_Norm = lay_odds / Normalised_Model_Odds (field-normalised)
    >1 → model thinks underpriced (back); <1 → overpriced (lay).

    Headline Back_Bet/Lay_Bet remain the normalised basis at zero margin for
    continuity; per-row potential P&L columns (prefix "_") let the strategy
    grids re-select bets under any basis/threshold/filter without recomputing.

    All winning P&L components are net of COMMISSION. Kelly stakes always use
    the raw probability — Kelly sizing needs a real probability, and the
    normalised column is not one.
    """
    df     = preds.copy()
    lay    = df[lay_odds_col].to_numpy(dtype=float)
    actual = df[target_col].to_numpy(dtype=float)
    model  = df["Normalised_Model_Odds"].to_numpy(dtype=float)
    p      = df["Probability"].to_numpy(dtype=float)

    rf, exp_w, act_w = dead_heat_arrays(df, event_full, market_name)
    df["DeadHeat_RF"] = rf
    df["Edge_Raw"]    = p * lay
    df["Edge_Norm"]   = lay / np.clip(model, 1e-8, None)

    # --- Potential P&L (as if every row were bet) ---
    back_pot = _net(np.where(actual == 1, BACK_STAKE * (rf * lay - 1), -BACK_STAKE))

    fl       = np.full(len(df), LAY_FIXED_LIABILITY[market_name])
    fs_liab  = LAY_FIXED_STAKE[market_name] * np.clip(lay - 1, 1e-8, None)
    lay_fl_pot = _net(_lay_return(lay, actual, exp_w, act_w, fl))
    lay_fs_pot = _net(_lay_return(lay, actual, exp_w, act_w, fs_liab))

    kb_stake = _kelly_back_stake(p, lay)
    kl_liab  = _kelly_lay_liability(p, lay)
    back_kelly_pot = _net(np.where(actual == 1, kb_stake * (rf * lay - 1), -kb_stake))
    lay_kelly_pot  = _net(_lay_return(lay, actual, exp_w, act_w, kl_liab))

    df["_Back_PnL_Pot"]       = back_pot
    df["_Back_Kelly_PnL_Pot"] = back_kelly_pot
    df["_Back_Kelly_Stake"]   = kb_stake
    df["_Lay_PnL_FL_Pot"]     = lay_fl_pot
    df["_Lay_PnL_FS_Pot"]     = lay_fs_pot
    df["_Lay_FS_Liab"]        = fs_liab
    df["_Lay_Kelly_PnL_Pot"]  = lay_kelly_pot
    df["_Lay_Kelly_Liab"]     = kl_liab

    # --- Headline bets: normalised basis, zero margin (NaN odds → no bet) ---
    is_back = model < lay
    df["Back_Bet"]         = is_back
    df["Back_PnL"]         = np.where(is_back, back_pot, 0.0)
    df["Back_Kelly_Stake"] = np.where(is_back, kb_stake, 0.0)
    df["Back_PnL_Kelly"]   = np.where(is_back, back_kelly_pot, 0.0)

    is_lay = model > lay
    df["Lay_Bet"]                  = is_lay
    df["Lay_PnL_FixedLiab"]        = np.where(is_lay, lay_fl_pot, 0.0)
    df["Lay_PnL_FixedStake"]       = np.where(is_lay, lay_fs_pot, 0.0)
    df["Lay_Liability_FixedStake"] = np.where(is_lay, fs_liab, 0.0)
    df["Lay_Kelly_Liability"]      = np.where(is_lay, kl_liab, 0.0)
    df["Lay_PnL_Kelly"]            = np.where(is_lay, lay_kelly_pot, 0.0)
    return df


# ===== SUMMARIES =====

def back_summary(df: pd.DataFrame, target_col: str) -> dict:
    bets = df[df["Back_Bet"]]
    if bets.empty:
        return {"Back_NBets": 0, "Back_NWon": 0, "Back_HitRate": np.nan,
                "Back_PnL": 0.0, "Back_ROI": np.nan,
                "Back_PnL_Kelly": 0.0, "Back_ROI_Kelly": np.nan}
    pnl       = float(bets["Back_PnL"].sum())
    pnl_k     = float(bets["Back_PnL_Kelly"].sum())
    staked_k  = float(bets["Back_Kelly_Stake"].sum())
    return {
        "Back_NBets":     len(bets),
        "Back_NWon":      int(bets[target_col].sum()),
        "Back_HitRate":   round(bets[target_col].mean(), 4),
        "Back_PnL":       round(pnl, 2),
        "Back_ROI":       round(pnl / (len(bets) * BACK_STAKE), 4),
        "Back_PnL_Kelly": round(pnl_k, 2),
        "Back_ROI_Kelly": round(pnl_k / staked_k, 4) if staked_k > 0 else np.nan,
    }


def lay_summary(df: pd.DataFrame, target_col: str, market_name: str) -> dict:
    """Lay ROI = P&L / total liability for every staking scheme."""
    bets = df[df["Lay_Bet"]]
    if bets.empty:
        return {"Lay_NBets": 0, "Lay_NLost": 0, "Lay_HitRate": np.nan,
                "Lay_PnL_FixedLiab": 0.0, "Lay_ROI_FixedLiab": np.nan,
                "Lay_PnL_FixedStake": 0.0, "Lay_ROI_FixedStake": np.nan,
                "Lay_PnL_Kelly": 0.0, "Lay_ROI_Kelly": np.nan}
    n_bets  = len(bets)
    n_lost  = int(bets[target_col].sum())          # player placed → layer lost
    pnl_fl  = float(bets["Lay_PnL_FixedLiab"].sum())
    pnl_fs  = float(bets["Lay_PnL_FixedStake"].sum())
    pnl_k   = float(bets["Lay_PnL_Kelly"].sum())
    liab_fl = n_bets * LAY_FIXED_LIABILITY[market_name]
    liab_fs = float(bets["Lay_Liability_FixedStake"].sum())
    liab_k  = float(bets["Lay_Kelly_Liability"].sum())
    return {
        "Lay_NBets":           n_bets,
        "Lay_NLost":           n_lost,
        "Lay_HitRate":         round((n_bets - n_lost) / n_bets, 4),
        "Lay_PnL_FixedLiab":   round(pnl_fl, 2),
        "Lay_ROI_FixedLiab":   round(pnl_fl / liab_fl, 4) if liab_fl > 0 else np.nan,
        "Lay_PnL_FixedStake":  round(pnl_fs, 2),
        "Lay_ROI_FixedStake":  round(pnl_fs / liab_fs, 4) if liab_fs > 0 else np.nan,
        "Lay_PnL_Kelly":       round(pnl_k, 2),
        "Lay_ROI_Kelly":       round(pnl_k / liab_k, 4) if liab_k > 0 else np.nan,
    }


# ===== MODEL QUALITY =====

def discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Metrics on RAW probabilities (never the normalised column)."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"AUC": np.nan, "Avg_Precision": np.nan, "TSS": np.nan,
                "Log_Loss": np.nan, "Brier": np.nan}
    y_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "AUC":           round(roc_auc_score(y_true, y_prob), 4),
        "Avg_Precision": round(average_precision_score(y_true, y_prob), 4),
        "TSS":           round(tss_optimal(y_true, y_prob), 4),
        "Log_Loss":      round(log_loss(y_true, y_clip), 5),
        "Brier":         round(brier_score_loss(y_true, y_clip), 5),
    }


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & ((y_prob < hi) | (hi == 1.0))
        if mask.sum() == 0:
            continue
        rows.append({
            "Bin_Low": round(lo, 2), "Bin_High": round(hi, 2), "N": int(mask.sum()),
            "Mean_Predicted": round(float(y_prob[mask].mean()), 4),
            "Actual_Rate":    round(float(y_true[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


# ===== EVENT LOOP =====

def backtest_events(package: dict, test_df: pd.DataFrame, tour_key: str,
                    test_year: int):
    """Predict + bet every event for every trained market.
    Returns (list of prediction frames, list of per-event summary dicts)."""
    all_preds, event_summaries = [], []
    events = sorted(test_df["eventID"].unique())
    print(f"  Test {test_year}: {len(events)} events | {len(test_df):,} rows")

    for event_id in events:
        event_full = test_df[test_df["eventID"] == event_id]
        event_date = event_full["Date"].iloc[0].strftime("%Y-%m-%d")

        for market_name, market_pkg in package.items():
            target_col   = MARKETS[market_name]["target_col"]
            lay_odds_col = LAY_ODDS_COLS[market_name]
            if target_col not in event_full.columns or lay_odds_col not in event_full.columns:
                continue
            if event_full[lay_odds_col].isna().all():
                continue

            preds = predict_event(event_full, market_pkg)
            if preds is None or preds.empty:
                continue

            preds["Actual"] = preds[target_col].astype(int)
            preds = apply_strategies(preds, event_full, market_name,
                                     lay_odds_col, target_col)

            event_summaries.append({
                "Tour": tour_key, "Test_Year": test_year, "EventID": event_id,
                "Date": event_date, "Market": market_name,
                "FieldSize": len(preds), "Positives": int(preds["Actual"].sum()),
                **back_summary(preds, target_col),
                **lay_summary(preds, target_col, market_name),
            })

            preds["Tour"], preds["Test_Year"] = tour_key, test_year
            preds["EventID"], preds["Date"], preds["Market"] = event_id, event_date, market_name
            all_preds.append(preds)

    return all_preds, event_summaries


# ===== AGGREGATION =====

def aggregate_results(all_preds: list, event_summaries: list, tour_key: str) -> dict | None:
    if not all_preds:
        return None
    pred_df  = pd.concat(all_preds, ignore_index=True)
    event_df = (pd.DataFrame(event_summaries)
                .sort_values(["Market", "Test_Year", "Date", "EventID"])
                .reset_index(drop=True))

    for mkt in event_df["Market"].unique():
        mask = event_df["Market"] == mkt
        for col, cum in [("Back_PnL", "Back_Cumulative_PnL"),
                         ("Lay_PnL_FixedLiab", "Lay_Cumulative_PnL_FixedLiab"),
                         ("Lay_PnL_FixedStake", "Lay_Cumulative_PnL_FixedStake")]:
            event_df.loc[mask, cum] = event_df.loc[mask, col].cumsum().round(2).to_numpy()

    def _rows(group_iter, extra_key):
        rows = []
        for keys, mdf in group_iter:
            market_name = keys[-1] if isinstance(keys, tuple) else keys
            target_col = MARKETS[market_name]["target_col"]
            y_true = mdf["Actual"].values
            y_prob = mdf["Probability"].values          # raw, not normalised
            row = {"Tour": tour_key}
            if extra_key:
                row[extra_key] = keys[0]
            rows.append({
                **row, "Market": market_name,
                "N_Events": int(mdf["EventID"].nunique()), "N_Players": len(mdf),
                "Prevalence": round(float(y_true.mean()), 4),
                **discrimination(y_true, y_prob),
                **back_summary(mdf, target_col),
                **lay_summary(mdf, target_col, market_name),
            })
        return pd.DataFrame(rows)

    season_df  = _rows(pred_df.groupby(["Test_Year", "Market"], sort=True), "Test_Year")
    summary_df = _rows(pred_df.groupby("Market", sort=False), None)

    calib = {mkt: calibration_bins(mdf["Actual"].values, mdf["Probability"].values)
             for mkt, mdf in pred_df.groupby("Market", sort=False)}

    return {"summary": summary_df, "season_summary": season_df,
            "event_results": event_df, "all_predictions": pred_df,
            "calibration": calib}


# ===== STRATEGY GRIDS =====
# Each grid row is one candidate strategy: market × edge basis × one filter
# applied on top of that basis's zero-margin baseline. P&L is recomputed from
# the per-row potential columns, so bets differ between bases and filters.
# Year-by-year consistency columns support the (human-led) policy decision.

def _grid_metrics(band_df: pd.DataFrame, pnl_col: str, total_staked: float,
                  kelly_pnl_col: str, kelly_denom_col: str) -> dict:
    pnl_vals  = band_df[pnl_col].to_numpy(dtype=float)
    total_pnl = pnl_vals.sum()
    event_pnl = band_df.groupby("EventID")[pnl_col].sum()
    epnl_std  = event_pnl.std()
    cum       = event_pnl.cumsum().values

    year_pnl   = band_df.groupby("Test_Year")[pnl_col].sum()
    years_pos  = f"{int((year_pnl > 0).sum())}/{len(year_pnl)}"
    worst_year = float(year_pnl.min())

    kelly_pnl   = float(band_df[kelly_pnl_col].sum())
    kelly_denom = float(band_df[kelly_denom_col].sum())

    return {
        "N_Bets":         len(band_df),
        "N_Won":          int(band_df["Actual"].sum()),
        "Strike_Rate_%":  round(band_df["Actual"].mean() * 100, 2),
        "Total_PnL":      round(total_pnl, 2),
        "ROI_%":          round(total_pnl / total_staked * 100, 2) if total_staked > 0 else np.nan,
        "Kelly_PnL":      round(kelly_pnl, 2),
        "Kelly_ROI_%":    round(kelly_pnl / kelly_denom * 100, 2) if kelly_denom > 0 else np.nan,
        "Avg_Lay_Odds":   round(float(band_df["_Lay_Odds"].mean()), 2),
        "Sharpe":         round(event_pnl.mean() / epnl_std, 3) if epnl_std > 0 else 0.0,
        "Max_Drawdown":   round(float((cum - np.maximum.accumulate(cum)).min()), 2),
        "Years_Pos":      years_pos,
        "Worst_Year_PnL": round(worst_year, 2),
    }


def _with_lay_odds(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["_Lay_Odds"] = np.nan
    for market_name, lay_col in LAY_ODDS_COLS.items():
        if lay_col in df.columns:
            m = df["Market"] == market_name
            df.loc[m, "_Lay_Odds"] = df.loc[m, lay_col]
    return df


def _sweeps(base: pd.DataFrame, market_name: str, edge_col: str, side: str):
    """Yield (filter_type, filter_value, band_df) for one basis baseline.

    One dimension at a time: All, edge thresholds, odds floors/ceilings,
    rating floors/ceilings. Back edge tightens upward (Edge >= t); lay edge
    tightens downward (Edge <= 1/t, i.e. market at least t× the model price).
    """
    yield "All", "-", base
    for t in EDGE_THRESHOLDS:
        if t == 1.0:
            continue                          # 1.0 is the baseline itself
        if side == "back":
            yield "Edge >=", t, base[base[edge_col] >= t]
        else:
            yield "Edge <=", round(1 / t, 3), base[base[edge_col] <= 1 / t]
    for t in ODDS_GRID[market_name]:
        yield "Odds >=", t, base[base["_Lay_Odds"] >= t]
    for t in ODDS_GRID[market_name]:
        yield "Odds <", t, base[base["_Lay_Odds"] < t]
    if "rating" in base.columns:
        for t in GRID_RATING_THRESHOLDS:
            yield "Rating >=", t, base[base["rating"] >= t]
        for t in GRID_RATING_THRESHOLDS:
            yield "Rating <", t, base[base["rating"] < t]


_EDGE_BASES = [("Raw", "Edge_Raw"), ("Norm", "Edge_Norm")]


def run_back_grid(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lay_odds(pred_df)
    rows = []
    for market_name in MARKETS:
        mdf = df[df["Market"] == market_name]
        if mdf.empty:
            continue
        for basis, edge_col in _EDGE_BASES:
            base = mdf[mdf[edge_col] > 1.0]
            for ftype, fval, band in _sweeps(base, market_name, edge_col, "back"):
                if band.empty:
                    continue
                rows.append({
                    "Market": market_name, "Edge_Basis": basis,
                    "Filter_Type": ftype, "Filter_Value": fval,
                    **_grid_metrics(band, "_Back_PnL_Pot", len(band) * BACK_STAKE,
                                    "_Back_Kelly_PnL_Pot", "_Back_Kelly_Stake"),
                })
    return pd.DataFrame(rows)


def run_lay_grid(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = _with_lay_odds(pred_df)
    rows = []
    for market_name in MARKETS:
        mdf = df[df["Market"] == market_name]
        if mdf.empty:
            continue
        fl = LAY_FIXED_LIABILITY[market_name]
        for basis, edge_col in _EDGE_BASES:
            base = mdf[mdf[edge_col] < 1.0]
            for ftype, fval, band in _sweeps(base, market_name, edge_col, "lay"):
                if band.empty:
                    continue
                m_fl = _grid_metrics(band, "_Lay_PnL_FL_Pot", len(band) * fl,
                                     "_Lay_Kelly_PnL_Pot", "_Lay_Kelly_Liab")
                m_fs = _grid_metrics(band, "_Lay_PnL_FS_Pot",
                                     float(band["_Lay_FS_Liab"].sum()),
                                     "_Lay_Kelly_PnL_Pot", "_Lay_Kelly_Liab")
                rows.append({
                    "Market": market_name, "Edge_Basis": basis,
                    "Filter_Type": ftype, "Filter_Value": fval,
                    "N_Bets": m_fl["N_Bets"], "N_Lost": m_fl["N_Won"],
                    "Strike_Rate_%": round(100 - m_fl["Strike_Rate_%"], 2),
                    "Avg_Lay_Odds": m_fl["Avg_Lay_Odds"],
                    "FL_Liability": fl, "FL_Total_PnL": m_fl["Total_PnL"],
                    "FL_ROI_%": m_fl["ROI_%"], "FL_Sharpe": m_fl["Sharpe"],
                    "FL_Max_Drawdown": m_fl["Max_Drawdown"],
                    "FL_Years_Pos": m_fl["Years_Pos"],
                    "FL_Worst_Year_PnL": m_fl["Worst_Year_PnL"],
                    "FS_Stake": LAY_FIXED_STAKE[market_name],
                    "FS_Total_PnL": m_fs["Total_PnL"], "FS_ROI_%": m_fs["ROI_%"],
                    "FS_Sharpe": m_fs["Sharpe"], "FS_Max_Drawdown": m_fs["Max_Drawdown"],
                    "FS_Years_Pos": m_fs["Years_Pos"],
                    "FS_Worst_Year_PnL": m_fs["Worst_Year_PnL"],
                    "Kelly_PnL": m_fl["Kelly_PnL"], "Kelly_ROI_%": m_fl["Kelly_ROI_%"],
                })
    return pd.DataFrame(rows)


# ===== EXPORT =====

def export_results(results: dict, out_path: Path,
                   back_grid: pd.DataFrame = None, lay_grid: pd.DataFrame = None) -> Path:
    pred_df  = results["all_predictions"]
    id_cols  = [c for c in ["Test_Year", "Date", "EventID", "Market", "playerID",
                            "surname", "firstname", "posn", "rating"] if c in pred_df.columns]
    lay_cols = [c for c in LAY_ODDS_COLS.values() if c in pred_df.columns]
    keep = id_cols + lay_cols + [
        "Model_Score", "Probability", "Normalised_Probability", "Normalised_Model_Odds",
        "Edge_Raw", "Edge_Norm", "Actual", "Back_Bet", "DeadHeat_RF", "Back_PnL",
        "Back_Kelly_Stake", "Back_PnL_Kelly",
        "Lay_Bet", "Lay_PnL_FixedLiab", "Lay_PnL_FixedStake",
        "Lay_Kelly_Liability", "Lay_PnL_Kelly",
    ]

    sheets = {
        "Summary":         results["summary"],
        "Event_Results":   results["event_results"],
        "All_Predictions": pred_df[[c for c in keep if c in pred_df.columns]],
    }
    if "season_summary" in results and len(results["season_summary"]):
        sheets["Season_Summary"] = results["season_summary"]
    for mkt, cdf in results.get("calibration", {}).items():
        if len(cdf):
            sheets[f"Calib_{mkt}"] = cdf
    if back_grid is not None and len(back_grid):
        sheets["Back_Strategy_Grid"] = back_grid
    if lay_grid is not None and len(lay_grid):
        sheets["Lay_Strategy_Grid"] = lay_grid

    write_workbook(out_path, sheets)
    print(f"\n  Saved: {out_path}")
    return out_path


# ===== STATIC DRIVER (current-season backtest) =====

def run_static(tour_key: str, year: int, no_grid: bool = False):
    """Evaluate the production bundle against completed events in `year`."""
    cfg = TOURS[tour_key]
    bundle_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"
    print(f"\n=== SEASONAL BACKTEST: {cfg['name']} {year} ===")

    if not bundle_path.exists():
        print(f"  Bundle not found: {bundle_path}"); return None
    if not cfg["processed_historical"].exists():
        print(f"  Processed historical file not found: {cfg['processed_historical']}\n"
              f"  Place {cfg['raw_historical'].name} in {cfg['raw_historical'].parent} "
              f"and run: python -m golfmodel preprocess"); return None
    package = load_bundle(bundle_path)
    print(f"  Markets: {list(package)}")

    df = load_processed_historical(tour_key)
    df = join_lay_odds(df, cfg["raw_historical"])
    year_df = df[df["Date"].dt.year == year].copy()
    if year_df.empty:
        print(f"  No data for {year}"); return None

    # Completed events only: at least one recorded finish position
    completed = year_df.groupby("eventID")["posn"].transform(lambda s: s.notna().any())
    year_df = year_df[completed]
    if year_df.empty:
        print(f"  No completed events for {year}"); return None

    all_preds, event_summaries = backtest_events(package, year_df, tour_key, year)
    results = aggregate_results(all_preds, event_summaries, tour_key)
    if results is None:
        print("  No results."); return None

    back_grid = lay_grid = None
    if not no_grid:
        back_grid = run_back_grid(results["all_predictions"])
        lay_grid  = run_lay_grid(results["all_predictions"])

    out_path = BACKTESTS_DIR / f"{tour_key}_Seasonal_Backtest_{year}.xlsx"
    export_results(results, out_path, back_grid, lay_grid)
    return results


# ===== WALK-FORWARD DRIVER =====

def get_windows(df: pd.DataFrame, min_test_year: int = None) -> list:
    """(train_start, train_end, test_year) tuples with full training coverage."""
    years = sorted(df["Date"].dt.year.dropna().unique())
    windows = []
    for test_year in years:
        train_start = test_year - TRAINING_YEARS
        if train_start < years[0]:
            continue
        if min_test_year and test_year < min_test_year:
            continue
        windows.append((int(train_start), int(test_year - 1), int(test_year)))
    return windows


def run_walkforward(tour_key: str, n_trials: int, start_year: int = None,
                    min_test_year: int = None, force_retrain: bool = False,
                    tag: str = None):
    """tag segregates cache dir and output file for A/B experiment runs
    (e.g. --tag residual vs --tag noodds) without clobbering each other."""
    cfg = TOURS[tour_key]
    wf_models_dir = WF_MODELS_DIR / tag if tag else WF_MODELS_DIR
    out_name = (f"{tour_key}_WalkForward_Backtest_{tag}.xlsx" if tag
                else f"{tour_key}_WalkForward_Backtest.xlsx")
    print(f"\n=== WALK-FORWARD: {cfg['name']}{f' [{tag}]' if tag else ''} ===")

    if not cfg["processed_historical"].exists():
        print(f"  Processed historical file not found: {cfg['processed_historical']}\n"
              f"  Place {cfg['raw_historical'].name} in {cfg['raw_historical'].parent} "
              f"and run: python -m golfmodel preprocess"); return None
    df = load_processed_historical(tour_key)
    if start_year:
        df = df[df["Date"].dt.year >= start_year].copy()
    df = join_lay_odds(df, cfg["raw_historical"])
    print(f"  {len(df):,} rows | years "
          f"{int(df['Date'].dt.year.min())}–{int(df['Date'].dt.year.max())}")

    windows = get_windows(df, min_test_year)
    if not windows:
        print("  No valid windows."); return None
    print(f"  Plan: {len(windows)} windows")

    all_preds, event_summaries = [], []
    prev_window_dir = None

    for train_start, train_end, test_year in windows:
        print(f"\n  --- Train {train_start}–{train_end} → Test {test_year} ---")
        window_dir  = wf_models_dir / f"{tour_key}_{test_year}"
        bundle_path = window_dir / "market_bundle.pkl"

        if bundle_path.exists() and not force_retrain:
            print("  Loading cached bundle...")
            package = joblib.load(bundle_path)
        else:
            window_dir.mkdir(parents=True, exist_ok=True)
            if prev_window_dir and prev_window_dir.exists():   # warm-start params
                for f in prev_window_dir.glob("*_best_params.pkl"):
                    if not (window_dir / f.name).exists():
                        shutil.copy(f, window_dir / f.name)

            train_df = df[(df["Date"].dt.year >= train_start) &
                          (df["Date"].dt.year <= train_end)]
            package = {}
            for market_name, market in MARKETS.items():
                if market["target_col"] in train_df.columns:
                    n_pos = int(train_df[market["target_col"]].sum())
                    if n_pos < MIN_POSITIVES:
                        print(f"    {market_name}: skipped ({n_pos} positives)")
                        continue
                result = train_market(market_name, train_df, tour_key,
                                      cfg["use_meta_odds"], n_trials, window_dir)
                if result is not None:
                    package[market_name] = result
            if package:
                joblib.dump(package, bundle_path)

        prev_window_dir = window_dir
        if not package:
            print(f"  No markets trained — skipping {test_year}")
            continue

        test_df = df[df["Date"].dt.year == test_year].copy()
        if test_df.empty:
            print(f"  No test data for {test_year}")
            continue
        preds, summaries = backtest_events(package, test_df, tour_key, test_year)
        all_preds.extend(preds)
        event_summaries.extend(summaries)

    results = aggregate_results(all_preds, event_summaries, tour_key)
    if results is None:
        print("  No results to aggregate."); return None

    print(f"\n  === AGGREGATE ({len(windows)} windows) ===")
    for _, row in results["summary"].iterrows():
        print(f"  {row['Market']:<8} AUC={row['AUC']:.4f} AP={row['Avg_Precision']:.4f} "
              f"Back P&L=£{row['Back_PnL']:.2f} ROI={row['Back_ROI']:.1%}")

    back_grid = run_back_grid(results["all_predictions"])
    lay_grid  = run_lay_grid(results["all_predictions"])
    export_results(results, BACKTESTS_DIR / out_name, back_grid, lay_grid)
    return results
