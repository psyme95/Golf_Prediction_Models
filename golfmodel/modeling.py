"""Model training: grouped CV, Optuna tuning, OOF stacking, meta-calibration.

All cross-validation is grouped by eventID (StratifiedGroupKFold) so rows from
the same tournament never appear in both train and validation folds, the
single fix for the leakage that inflated the old pipeline's OOF metrics.

Bundle schema is kept compatible with the old pipeline's joblib bundles:
per-market dict with keys models / meta_model / meta_scaler / meta_uses_odds /
model_vars / odds_col / market_size, wrapped in {"markets": {...}}.
"""

from datetime import datetime
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from .config import (
    BACK_OBJ_POWER,
    BASE_MODEL_VARS,
    CROSS_MARKET_VARS,
    FRAC_LABEL_COLS,
    MARKETS,
    MODELS_DIR,
    N_CV_REPEATS,
    N_CV_SPLITS,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    SEASON_SUFFIX,
    TOURS,
)
from .data import load_processed_historical, training_years_slice

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = ["logistic", "rf", "lgbm", "xgb", "lgbm_dart"]

_RF_MAX_FEATURES = {"sqrt": "sqrt", "log2": "log2", "frac03": 0.3, "frac05": 0.5}


# ===== GROUPED CV =====

def grouped_cv_splits(X, y, groups, n_repeats: int, seed: int = RANDOM_SEED) -> list:
    """Repeated StratifiedGroupKFold splits (sklearn has no repeated variant).
    Degenerate folds (a class missing from either side) are skipped."""
    splits = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=seed + r)
        for tr, va in cv.split(X, y, groups):
            if y[tr].sum() == 0 or y[va].sum() == 0:
                continue
            splits.append((tr, va))
    return splits


# ===== METRICS =====

def tss_optimal(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Max TSS (TPR - FPR) over all decision thresholds, O(n log n)."""
    y_true = np.asarray(y_true)
    order  = np.argsort(-y_prob, kind="stable")
    y      = y_true[order]
    n_pos  = y.sum()
    n_neg  = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(1 - y) / n_neg
    probs = np.asarray(y_prob)[order]
    last_of_tie = np.r_[probs[1:] != probs[:-1], True]   # evaluate once per distinct threshold
    return float((tpr - fpr)[last_of_tie].max())


# ===== MODEL REGISTRY =====

def _suggest_params(trial, model_key: str, n_pos: int) -> dict:
    leaf_max = min(200, max(5, n_pos * 2))
    if model_key == "logistic":
        return {
            "C":        trial.suggest_float("C", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_key == "rf":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
            "max_features":     trial.suggest_categorical("max_features", list(_RF_MAX_FEATURES)),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, leaf_max),
            "max_depth":        trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
        }
    if model_key == "lgbm":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, leaf_max),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
    if model_key == "xgb":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
    if model_key == "lgbm_dart":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, leaf_max),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "drop_rate":         trial.suggest_float("drop_rate", 0.05, 0.3),
        }
    raise ValueError(f"Unknown model key: {model_key}")


def price_weights(odds: np.ndarray) -> np.ndarray:
    """Sample weights that tilt the tuning objective toward backable prices.

    Log-loss otherwise optimises calibration where the mass is, the longshot
    tail, but a back bet is only ever struck at short prices. Weight rises with
    implied probability and is normalised to mean 1 so losses stay comparable
    across markets and settings.
    """
    implied = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.0 + 1e-6, None)
    w = np.power(np.clip(implied, 1e-6, 1.0), BACK_OBJ_POWER)
    return w / w.mean()


def expand_fractional(X, frac, groups):
    """Represent a fractional target as weighted binary rows.

    Cross-entropy against a fractional label f is exactly reproduced by fitting a
    binary classifier on two rows, (x, y=1, w=f) and (x, y=0, w=1-f). That keeps
    the whole five-model classifier stack, the grouped CV, predict_proba()[:,1],
    the meta LogisticRegression and every metric working unchanged, the
    alternative is swapping all five base models for regressors, and
    LogisticRegression has no fractional-target mode at all.

    Only genuinely fractional rows are duplicated (1.3-5.6% of the data), so this
    costs a few percent more rows, not 2x. Both copies keep the same group, so
    grouped CV never splits a row across train and validation.
    """
    frac   = np.asarray(frac, dtype=float)
    groups = np.asarray(groups)
    n      = len(frac)

    # Every row emits both a positive and a negative copy; zero-weight copies are
    # then dropped. f==0 keeps only the negative, f==1 only the positive, and a
    # genuine dead heat keeps both.
    X2      = np.vstack([X, X])
    y2      = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    w2      = np.concatenate([frac, 1.0 - frac])
    groups2 = np.concatenate([groups, groups])

    keep = w2 > 0
    return X2[keep], y2[keep], w2[keep], groups2[keep]


def make_model(model_key: str, params: dict, spw: float, seed: int = RANDOM_SEED,
               weighted: bool = False):
    """Instantiate a model from tuned params plus fixed per-family kwargs."""
    p = dict(params)
    # class_weight / scale_pos_weight MULTIPLY with sample_weight, which would
    # double-count the dead-heat weighting, so they are dropped when weights
    # carry the class balance themselves.
    cw  = None if weighted else "balanced"
    spw = 1.0 if weighted else spw
    if model_key == "logistic":
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  class_weight=cw, max_iter=2000,
                                  random_state=seed, **p)
    if model_key == "rf":
        p["max_features"] = _RF_MAX_FEATURES.get(p.get("max_features", "sqrt"), "sqrt")
        return RandomForestClassifier(class_weight=cw, n_jobs=-1,
                                      random_state=seed, **p)
    if model_key == "lgbm":
        return lgb.LGBMClassifier(class_weight=cw, random_state=seed,
                                  verbose=-1, **p)
    if model_key == "xgb":
        return xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=seed, verbosity=0, **p)
    if model_key == "lgbm_dart":
        return lgb.LGBMClassifier(boosting_type="dart", class_weight=cw,
                                  random_state=seed, verbose=-1, **p)
    raise ValueError(f"Unknown model key: {model_key}")


# ===== TUNING =====

def tune(model_key: str, X, y, groups, n_trials: int, spw: float,
         warm_params: dict = None, seed: int = RANDOM_SEED, w=None,
         weighted: bool = False) -> dict:
    """Optuna search minimising grouped-CV log-loss.

    Log-loss targets probability calibration directly, betting P&L depends on
    being right about the probability at the price, not on ranking (which is
    what the previous average-precision objective optimised).

    Scored by a manual fold loop rather than cross_val_score, because sklearn's
    "neg_log_loss" string scorer has no hook for per-row sample weights, which
    dead-heat labels require on both the fit and the score."""
    # `weighted` means the weights carry the class balance (dead-heat labels), so
    # class_weight="balanced" must be dropped or it double-counts. A price tilt
    # alone does not carry class balance, so it leaves class_weight in place.
    w = np.ones(len(y), dtype=float) if w is None else np.asarray(w, dtype=float)
    n_pos = int(round(float(w[y == 1].sum())))
    cv_splits = grouped_cv_splits(X, y, groups, n_repeats=1, seed=seed)

    def fold_loss(params, tr, va):
        model = make_model(model_key, params, spw, seed, weighted=weighted)
        model.fit(X[tr], y[tr], sample_weight=w[tr])
        p = np.clip(model.predict_proba(X[va])[:, 1], 1e-15, 1 - 1e-15)
        return log_loss(y[va], p, sample_weight=w[va], labels=[0, 1])

    def objective(trial):
        params = _suggest_params(trial, model_key, n_pos)
        losses = Parallel(n_jobs=-1)(
            delayed(fold_loss)(params, tr, va) for tr, va in cv_splits)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm_params:
        try:
            study.enqueue_trial(warm_params)
        except Exception:
            pass
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ===== OOF + META =====

def generate_oof(model_params: dict, X, y, groups, spw: float,
                 seed: int = RANDOM_SEED, w=None, weighted: bool = False):
    """Out-of-fold predictions for all models over repeated grouped CV.
    Returns (oof_matrix, covered_mask); rows never validated are masked out."""
    splits = grouped_cv_splits(X, y, groups, n_repeats=N_CV_REPEATS, seed=seed)
    n, m = len(y), len(model_params)
    oof_sum   = np.zeros((n, m))
    oof_count = np.zeros(n, dtype=int)   # identical across models (same splits)

    for tr, va in splits:
        for j, (name, params) in enumerate(model_params.items()):
            model = make_model(name, params, spw, seed, weighted=weighted)
            model.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
            oof_sum[va, j] += model.predict_proba(X[va])[:, 1]
        oof_count[va] += 1

    covered = oof_count > 0
    oof = oof_sum[covered] / oof_count[covered, None]
    return oof, covered


def market_logit(odds_values: np.ndarray) -> np.ndarray:
    """Implied log-odds from decimal odds, the residual-modelling market anchor.
    The meta-learner sees logit(1/odds) so its coefficient measures how much
    the ensemble leans on market consensus vs deviates from it."""
    implied = np.clip(1.0 / np.clip(odds_values, 1.0 + 1e-6, None), 1e-6, 1 - 1e-6)
    return np.log(implied / (1 - implied))


def fit_meta_model(oof_matrix, y, market_feature=None, seed: int = RANDOM_SEED, w=None):
    """LogisticRegression on OOF base-model scores (+ optional market log-odds)."""
    meta_X = oof_matrix if market_feature is None else np.column_stack([oof_matrix, market_feature])
    scaler = StandardScaler()
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    meta.fit(scaler.fit_transform(meta_X), y, sample_weight=w)
    return meta, scaler


def market_share(odds: pd.Series, kept: pd.Series) -> float:
    """Share of an event's market-implied probability held by the kept rows.

    Players dropped for missing features are still in the event and still hold
    their share of the market, so the remaining players should be normalised to
    market_size * share rather than the full market_size.
    """
    odds = pd.to_numeric(odds, errors="coerce")
    implied = 1.0 / odds.where(odds > 1.0)
    total = implied.sum()
    return float(implied[kept].sum() / total) if total > 0 else 1.0


def normalise(proba: np.ndarray, market_size: float, share: float = 1.0) -> np.ndarray:
    """Rescale probabilities to sum to market_size * share within an event."""
    total = proba.sum()
    return proba / total * market_size * share if total > 0 else proba


def ensemble_predict(market_pkg: dict, X: np.ndarray, odds_values=None):
    """Base models → meta-model → calibrated probability.

    Bundle compat: old bundles (no 'meta_odds_form' key) trained on raw implied
    probability; new bundles ('meta_odds_form' == 'logit') on market log-odds.
    The feature fed at predict time must match what the meta was trained on."""
    model_preds = np.column_stack([
        m.predict_proba(X)[:, 1] for m in market_pkg["models"].values()
    ])
    raw_score = model_preds.mean(axis=1)

    if market_pkg.get("meta_uses_odds"):
        if odds_values is None:
            raise ValueError("Market meta-model requires odds_values")
        if market_pkg.get("meta_odds_form") == "logit":
            market_feat = market_logit(np.asarray(odds_values, dtype=float))
        else:
            market_feat = 1.0 / np.clip(odds_values, 1e-8, None)
        meta_input = np.column_stack([model_preds, market_feat])
    else:
        meta_input = model_preds

    proba = market_pkg["meta_model"].predict_proba(
        market_pkg["meta_scaler"].transform(meta_input)
    )[:, 1]
    return proba, raw_score


# ===== MARKET TRAINING =====

def train_market(market_name: str, train_df: pd.DataFrame, tour_key: str,
                 use_meta_odds: bool, n_trials: int, models_dir: Path,
                 seed: int = RANDOM_SEED, labels: str = "naive",
                 features: str = "base", objective: str = "logloss") -> dict | None:
    """Tune, stack, and calibrate one market. Returns the market package.

    labels="naive" trains on the binary `posn <= k` target (the lay baseline,
    unchanged). labels="frac" trains on the dead-heat-adjusted settled fraction,
    represented as weighted binary rows via expand_fractional, the naive label
    overstates the base rate by 13-19% relative, which inflates back edge.

    features="base" is the lay baseline's pure skill/course-fit set. "cross" adds
    de-vigged cross-market prices; "odds" uses ONLY those, which is the ablation
    arm that separates cross-market arbitrage from genuine golf forecasting.
    """
    market = MARKETS[market_name]
    target_col, odds_col = market["target_col"], market["odds_col"]

    # Winner has no single-odds profit column, so it keeps the binary label.
    frac_col = FRAC_LABEL_COLS.get(market_name, (None,) * 4)[3]
    use_frac = labels == "frac" and frac_col is not None and frac_col in train_df.columns
    if labels == "frac" and not use_frac:
        print(f"    {market_name}: no fractional label available, using binary target")
    label_col = frac_col if use_frac else target_col

    wanted = {"base":  BASE_MODEL_VARS,
              "cross": BASE_MODEL_VARS + CROSS_MARKET_VARS,
              "odds":  CROSS_MARKET_VARS}[features]        # "odds" = ablation arm
    available_vars = [v for v in wanted if v in train_df.columns]
    missing = [v for v in wanted if v not in train_df.columns]
    if missing:
        print(f"    Warning: missing vars (skipped): {missing}")

    df = train_df.copy()
    if "posn" in df.columns:                      # no recorded result → not trainable
        df = df[df["posn"].notna()]
    subset = available_vars + [label_col] + ([odds_col] if use_meta_odds else [])
    cols = [c for c in subset if c in df.columns]
    # Coerce stray non-numeric cells to NaN so they drop out as missing rows
    # instead of crashing .astype(float) mid-run (mirrors predict_event).
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols)

    X      = df[available_vars].values.astype(float)
    groups = df["eventID"].values
    n_rows = len(df)

    if use_frac:
        # Weighted binary rows are exactly cross-entropy against the fraction.
        # market_feat below must be indexed by the expanded row order, so keep
        # the odds aligned through the same expansion.
        odds_raw = df[odds_col].values.astype(float) if use_meta_odds else np.zeros(n_rows)
        Xo, y, w, groups = expand_fractional(
            np.column_stack([X, odds_raw]), df[label_col].values, groups)
        X, odds_expanded = Xo[:, :-1], Xo[:, -1]
        n_dup = len(y) - n_rows
    else:
        y, w, odds_expanded, n_dup = df[label_col].values.astype(int), None, None, 0

    if objective == "shortprice":
        if not use_meta_odds:
            raise ValueError("objective='shortprice' needs the market odds column")
        pw = price_weights(odds_expanded if use_frac else df[odds_col].values.astype(float))
        w  = pw if w is None else w * pw

    n_pos = float(w[y == 1].sum()) if use_frac else int(y.sum())
    n_neg = (float(w[y == 0].sum()) if use_frac else len(y) - n_pos)
    if n_pos == 0:
        print(f"    {market_name}: no positives, skipped")
        return None
    spw = n_neg / n_pos
    dup_note = f" | +{n_dup} dead-heat rows" if n_dup else ""
    print(f"    {market_name}: {n_rows:,} rows | {n_pos:.1f} positives "
          f"({100 * n_pos / n_rows:.1f}%) | {df['eventID'].nunique()} events"
          f"{dup_note} | labels={'frac' if use_frac else 'naive'}")

    model_params = {}
    for name in MODEL_NAMES:
        warm_path = models_dir / f"{tour_key}_{market_name}_{name}_best_params.pkl"
        warm = joblib.load(warm_path) if warm_path.exists() else None
        model_params[name] = tune(name, X, y, groups, n_trials, spw, warm, seed,
                                  w=w, weighted=use_frac)

    models_dir.mkdir(parents=True, exist_ok=True)
    for name, params in model_params.items():
        joblib.dump(params, models_dir / f"{tour_key}_{market_name}_{name}_best_params.pkl")

    print(f"    Generating OOF ({N_CV_SPLITS}-fold x {N_CV_REPEATS} grouped repeats)...")
    oof, covered = generate_oof(model_params, X, y, groups, spw, seed, w=w,
                                weighted=use_frac)
    y_cov = y[covered]
    w_cov = w[covered] if w is not None else None
    if (~covered).sum():
        print(f"    {(~covered).sum()} rows never validated, excluded from meta fit")

    metrics = {}
    for j, name in enumerate(model_params):
        oof_j = np.clip(oof[:, j], 1e-15, 1 - 1e-15)
        metrics[name] = {
            "log_loss":      log_loss(y_cov, oof_j, sample_weight=w_cov, labels=[0, 1]),
            "roc_auc":       roc_auc_score(y_cov, oof_j, sample_weight=w_cov),
            "avg_precision": average_precision_score(y_cov, oof_j, sample_weight=w_cov),
            "tss":           tss_optimal(y_cov, oof_j),
        }
        m = metrics[name]
        print(f"      {name:10s} log_loss={m['log_loss']:.4f} AUC={m['roc_auc']:.4f} "
              f"AP={m['avg_precision']:.4f} TSS={m['tss']:.4f}")

    if use_meta_odds:
        odds_for_meta = odds_expanded if use_frac else df[odds_col].values.astype(float)
        market_feat = market_logit(odds_for_meta[covered])
    else:
        market_feat = None
    meta_model, meta_scaler = fit_meta_model(oof, y_cov, market_feat, seed, w=w_cov)
    meta_market_coef = float(meta_model.coef_[0, -1]) if use_meta_odds else None
    if use_meta_odds:
        print(f"    Meta market coefficient (scaled): {meta_market_coef:.3f}")

    final_models = {}
    for name, params in model_params.items():
        model = make_model(name, params, spw, seed, weighted=use_frac)
        model.fit(X, y, sample_weight=w)
        final_models[name] = model

    return {
        "models":         final_models,
        "model_names":    list(model_params),
        "meta_model":     meta_model,
        "meta_scaler":    meta_scaler,
        "meta_uses_odds": use_meta_odds,
        "meta_odds_form": "logit" if use_meta_odds else None,
        "meta_market_coef": meta_market_coef,
        "model_vars":     available_vars,
        "odds_col":       odds_col,
        "market_size":    market["market_size"],
        "features":       features,
        "objective":      objective,
        "metrics":        metrics,
        "n_samples":      n_rows,
        "n_positives":    n_pos,
        "labels":         labels,
        "trained_at":     datetime.now(),
    }


# ===== TOUR TRAINING =====

def train_tour(tour_key: str, n_trials: int = OPTUNA_TRIALS,
               models_dir: Path = MODELS_DIR,
               train_df: pd.DataFrame = None,
               labels: str = "naive", features: str = "base",
               objective: str = "logloss") -> dict | None:
    """Train all markets for one tour and save the bundle + summary.
    train_df override is used by the walk-forward (window slices)."""
    cfg = TOURS[tour_key]
    print(f"\n=== TRAINING: {cfg['name']} ===")

    if train_df is None:
        if not cfg["processed_historical"].exists():
            print(f"  Processed file not found: {cfg['processed_historical']}")
            return None
        train_df = training_years_slice(load_processed_historical(tour_key))

    markets = {}
    for market_name in MARKETS:
        result = train_market(market_name, train_df, tour_key,
                              cfg["use_meta_odds"], n_trials, models_dir,
                              labels=labels, features=features,
                              objective=objective)
        if result is not None:
            markets[market_name] = result

    if not markets:
        return None

    package = {
        "markets":    markets,
        "tour_key":   tour_key,
        "season":     SEASON_SUFFIX,
        "trained_at": datetime.now(),
    }
    bundle_path = models_dir / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"
    joblib.dump(package, bundle_path)
    print(f"  Bundle saved: {bundle_path}")

    rows = [
        {"Tour": tour_key, "Market": mkt, "Model": name,
         "Log_Loss": round(m["log_loss"], 5), "ROC_AUC": round(m["roc_auc"], 4),
         "Avg_Precision": round(m["avg_precision"], 4), "TSS": round(m["tss"], 4),
         "N_Samples": res["n_samples"], "N_Positives": res["n_positives"],
         "Meta_Market_Coef": (round(res["meta_market_coef"], 4)
                              if res.get("meta_market_coef") is not None else None)}
        for mkt, res in markets.items() for name, m in res["metrics"].items()
    ]
    summary_path = models_dir / f"{tour_key}_Training_Summary_{SEASON_SUFFIX}.xlsx"
    pd.DataFrame(rows).to_excel(summary_path, index=False)
    print(f"  Summary saved: {summary_path.name}")
    return package


def predict_weekly(tour_key: str) -> Path | None:
    """Generate the weekly prediction workbook from the production bundle."""
    cfg = TOURS[tour_key]
    bundle_path = MODELS_DIR / f"{tour_key}_Trained_Models_{SEASON_SUFFIX}.pkl"
    weekly_path = cfg["processed_weekly"]
    if not bundle_path.exists():
        print(f"  Bundle not found: {bundle_path}"); return None
    if not weekly_path.exists():
        print(f"  Weekly file not found: {weekly_path}"); return None

    package = load_bundle(bundle_path)
    newdat  = pd.read_excel(weekly_path)
    print(f"\n=== PREDICTING: {cfg['name']} ({len(newdat)} players) ===")

    from .config import PREDICTIONS_DIR
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{tour_key}_Predictions_{datetime.now():%d-%m-%Y}.xlsx"

    try:
        writer_cm = pd.ExcelWriter(out_path, engine="openpyxl")
    except PermissionError:
        print(f"  ERROR: {out_path.name} is open in Excel, close it and rerun.")
        return None

    with writer_cm as writer:
        for market_name, pkg in package.items():
            model_vars = [v for v in pkg["model_vars"] if v in newdat.columns]
            odds_col   = pkg["odds_col"]
            if odds_col not in newdat.columns:
                print(f"  {market_name}: odds column '{odds_col}' missing, skipped")
                continue

            kept = newdat[model_vars + [odds_col]].notna().all(axis=1)
            share = market_share(newdat[odds_col], kept)
            df = newdat[kept]
            if df.empty:
                print(f"  {market_name}: no complete rows, skipped")
                continue

            X = df[model_vars].values.astype(float)
            odds_values = df[odds_col].values.astype(float) if pkg.get("meta_uses_odds") else None
            proba, raw_score = ensemble_predict(pkg, X, odds_values=odds_values)

            norm_prob = normalise(proba, pkg["market_size"], share)

            out = pd.DataFrame({
                "Surname":                df.get("surname", df.get("Surname")),
                "Firstname":              df.get("firstname", df.get("Firstname")),
                "Rating":                 df.get("rating"),
                "Market_Odds":            df[odds_col],
                "Model_Score":            np.round(raw_score, 5),
                "Probability":            np.round(proba, 6),
                "Normalised_Probability": np.round(norm_prob, 6),
                # Lay trigger: lay while the available lay odds are below this.
                "Normalised_Model_Odds":  np.round(1.0 / np.clip(norm_prob, 1e-8, None), 2),
            }).sort_values("Market_Odds")
            out.to_excel(writer, sheet_name=f"{market_name}_Market", index=False)
            print(f"  {market_name}: {len(out)} players | "
                  f"prob {proba.min():.4f}–{proba.max():.4f}")

    print(f"  Saved: {out_path.name}")
    return out_path


def load_bundle(path: Path) -> dict:
    """Load a bundle and normalise to {market_name: market_pkg}.
    Accepts both wrapped ({'markets': {...}}) and flat formats."""
    bundle = joblib.load(path)
    if "markets" in bundle and isinstance(bundle["markets"], dict):
        bundle = bundle["markets"]
    known = {k: v for k, v in bundle.items() if k in MARKETS}
    if not known:
        raise ValueError(f"No recognised markets in bundle {path.name}: {list(bundle)}")
    return known
