"""Data loading, preprocessing, and feature engineering.

Preprocessing keeps every row, no blanket dropna. NaN filtering happens only
at model-input time (dropna on the columns a model actually needs), so the
processed historical file retains the full field of every event. That full
field is what dead-heat and field-relative calculations must see.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CROSS_MARKET_ODDS, CROSS_MARKET_SIZE, DROP_COLS,
                     FRAC_LABEL_COLS, FRAC_STAKE, LAY_ODDS_COLS,
                     MKT_MIN_COVERAGE, TOURS, TRAINING_YEARS)

# Raw columns that may arrive as strings but must be numeric.
_NUMERIC_COERCE = ["yr3_All", "rating", "current", "X_1yr", "X_6m", "posn"]

# Columns whose presence marks a historical (completed-events) file.
_HISTORICAL_COLS = {"eventID", "posn", "Date", "playerID"}


def load_raw(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)

    # Excel/R convention: columns like `_1yr` arrive underscore-prefixed.
    df.columns = [f"X{c}" if c.startswith("_") else c for c in df.columns]

    for col in _NUMERIC_COERCE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def is_historical(df: pd.DataFrame) -> bool:
    return _HISTORICAL_COLS.issubset(df.columns)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Finish-position targets. NaN posn (WD/no result) → 0 for all targets;
    such rows are excluded from training via the posn filter in train_market."""
    df = df.copy()
    for name, cut in [("top_40", 40), ("top_20", 20), ("top_10", 10), ("top_5", 5)]:
        df[name] = (df["posn"] <= cut).astype(int)
    df["win"] = (df["posn"] == 1).astype(int)
    return add_frac_targets(df)


# Max disagreement between the settled fraction and the dead-heat-implied one
# before a row is treated as a settlement error rather than a real dead heat.
_FRAC_TOL = 0.02


def add_frac_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Dead-heat-adjusted labels recovered from the settled profit columns.

    profit = stake * (frac*odds - 1)  ->  frac = (profit/stake + 1) / odds

    A naive `posn <= k` label counts every member of a tie straddling the cut as
    a full winner, overstating the base rate by 13-19% relative. The recovered
    fraction is what actually settled.

    Rows are nulled (not clipped) where the recovered fraction disagrees with the
    dead-heat-implied fraction by more than _FRAC_TOL. ~31 rows across both tours
    are genuine settlement errors (e.g. a top-5 finish at 1001 settled as a total
    loss); training on them would teach the model a lie. Missing odds or profit
    also yield NaN, filtered at model-input time.
    """
    if "posn" not in df.columns:
        return df
    df = df.copy()
    posn   = pd.to_numeric(df["posn"], errors="coerce")
    n_tied = df.groupby(["eventID", "posn"])["posn"].transform("size")         if "eventID" in df.columns else pd.Series(1, index=df.index)

    for _, (_, profit_col, odds_col, frac_col) in FRAC_LABEL_COLS.items():
        if profit_col not in df.columns or odds_col not in df.columns:
            continue
        cut    = int("".join(c for c in odds_col if c.isdigit()))
        profit = pd.to_numeric(df[profit_col], errors="coerce")
        odds   = pd.to_numeric(df[odds_col],   errors="coerce").replace(0, np.nan)

        frac    = ((profit / FRAC_STAKE) + 1.0) / odds
        implied = ((cut - posn + 1) / n_tied).clip(0.0, 1.0)
        bad     = (frac - implied).abs() > _FRAC_TOL
        df[frac_col] = frac.clip(0.0, 1.0).mask(bad)
    return df


def _field_relatives(df: pd.DataFrame, col: str, group) -> pd.DataFrame:
    """Field-relative stats for one column within each event group.
    Group stats use the full field (NaN-skipping), not a filtered subset."""
    g = df.groupby(group)[col]
    field_mean = g.transform("mean")
    field_std  = g.transform("std").clip(lower=1e-8)

    df[f"{col}_vs_field_mean"]    = df[col] - field_mean
    df[f"{col}_vs_field_median"]  = df[col] - g.transform("median")
    df[f"{col}_vs_field_best"]    = df[col] - g.transform("max")
    df[f"{col}_field_zscore"]     = (df[col] - field_mean) / field_std
    df[f"{col}_field_percentile"] = g.transform(lambda x: x.rank(method="average", pct=True))
    return df


def add_market_probs(df: pd.DataFrame, group) -> pd.DataFrame:
    """Overround-free market probabilities and cross-market shape features.

    Each market's implied probabilities are normalised within the event to sum to
    that market's size (the expected number of players finishing top-N), which
    strips the bookmaker's margin. Ratios between adjacent markets then describe
    the market-implied SHAPE of a player's finish distribution, and, where the
    books contradict themselves, the mispricing.

    Events priced for less than MKT_MIN_COVERAGE of the field are nulled rather
    than normalised: a partially-priced event inflates every survivor by
    1/coverage, because groupby.transform("sum") skips NaN.
    """
    for name, odds_col in CROSS_MARKET_ODDS.items():
        if odds_col not in df.columns:
            continue
        odds = pd.to_numeric(df[odds_col], errors="coerce").replace(0, np.nan)
        raw  = 1.0 / odds.where(odds > 1.0)
        cov  = raw.notna().groupby(group).transform("mean")
        tot  = raw.groupby(group).transform("sum")

        p = (raw / tot * CROSS_MARKET_SIZE[name]).clip(1e-6, 0.999)
        p = p.mask(cov < MKT_MIN_COVERAGE)
        df[f"mkt_{name}"]       = p
        df[f"mkt_{name}_logit"] = np.log(p / (1 - p))

    # Adjacent-market log-odds ratios: the shape of the implied distribution.
    for a, b in [("win", "top5"), ("top5", "top10"), ("top10", "top20")]:
        la, lb = f"mkt_{a}_logit", f"mkt_{b}_logit"
        if la in df.columns and lb in df.columns:
            df[f"mkt_{a}_vs_{b}"] = df[la] - df[lb]

    for name in ("top10", "top20"):
        col = f"mkt_{name}_logit"
        if col in df.columns:
            df = _field_relatives(df, col, group)
    return df


def add_event_features(df: pd.DataFrame, historical: bool) -> pd.DataFrame:
    """Event-relative rating and strokes-gained features.
    Weekly files are a single field, so the whole frame is one group."""
    df = df.copy()
    group = df["eventID"] if historical else pd.Series(0, index=df.index)

    if "rating" in df.columns:
        df = _field_relatives(df, "rating", group)
        df["rating_vs_field_worst"] = df["rating"] - df.groupby(group)["rating"].transform("min")
        df["field_size"]     = df.groupby(group)["rating"].transform("count")
        df["field_strength"] = df.groupby(group)["rating"].transform("mean")
        df["field_depth"]    = df.groupby(group)["rating"].transform("std").clip(lower=1e-8)

    sg_cols = [c for c in ["sgtee", "sgt2g", "sgapp", "sgatg", "sgp"] if c in df.columns]
    if "sgtee" in df.columns and "sgapp" in df.columns:
        df["sg_ball_striking"] = df["sgtee"] + df["sgapp"]
        sg_cols.append("sg_ball_striking")
    if "sgatg" in df.columns and "sgp" in df.columns:
        df["sg_short_game"] = df["sgatg"] + df["sgp"]
        sg_cols.append("sg_short_game")

    for col in sg_cols:
        df = _field_relatives(df, col, group)

    return add_market_probs(df, group)


def preprocess(tour_key: str, kind: str) -> pd.DataFrame | None:
    """Process one raw file ('historical' or 'weekly') and write the result.
    Returns the processed frame, or None if the raw file is missing."""
    cfg = TOURS[tour_key]
    in_path  = cfg[f"raw_{kind}"]
    out_path = cfg[f"processed_{kind}"]

    if not in_path.exists():
        print(f"  {tour_key} {kind}: raw file not found, skipping ({in_path})")
        return None

    df = load_raw(in_path)
    historical = is_historical(df)
    print(f"  {tour_key} {kind}: {len(df):,} rows, {df.shape[1]} cols")

    drop = [c for c in DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    if historical:
        df = add_targets(df)
    df = add_event_features(df, historical)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"  {tour_key} {kind}: saved {len(df):,} rows, {df.shape[1]} cols → {out_path.name}")
    return df


def load_processed_historical(tour_key: str) -> pd.DataFrame:
    """Load the processed historical file with parsed dates."""
    path = TOURS[tour_key]["processed_historical"]
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.dropna(subset=["Date"])


def join_lay_odds(df: pd.DataFrame, raw_file: Path) -> pd.DataFrame:
    """Left-join Betfair lay odds from the raw file on eventID + playerID.
    Zero odds (invalid) are treated as missing."""
    join_on  = ["eventID", "playerID"]
    lay_cols = list(LAY_ODDS_COLS.values())

    if not raw_file.exists():
        print(f"  WARNING: raw file not found for lay odds join: {raw_file}")
        return df

    usecols = join_on + lay_cols
    raw = pd.read_excel(raw_file, usecols=lambda c: c in usecols)
    present = [c for c in lay_cols if c in raw.columns]
    if not present:
        print(f"  WARNING: no lay odds columns in {raw_file.name}")
        return df

    raw = (raw[join_on + present]
           .dropna(subset=join_on)
           .drop_duplicates(subset=join_on, keep="first"))
    raw[present] = raw[present].replace(0, np.nan)

    df = df.drop(columns=[c for c in present if c in df.columns])
    df = df.merge(raw, on=join_on, how="left")

    n = df[present[0]].notna().sum()
    print(f"  Lay odds joined: {n:,}/{len(df):,} rows ({100 * n / len(df):.1f}%)")
    return df


def training_years_slice(df: pd.DataFrame, window: str = "rolling") -> pd.DataFrame:
    """Complete calendar years to train on, excluding the in-progress year.
    rolling: the last TRAINING_YEARS years. expanding: every year in the data."""
    max_year   = int(df["Date"].dt.year.max())
    start_year = (int(df["Date"].dt.year.min()) if window == "expanding"
                  else max_year - TRAINING_YEARS)
    end_year   = max_year - 1
    subset = df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)]
    print(f"  Training window: {start_year}–{end_year} ({len(subset):,} of {len(df):,} rows)")
    return subset
