"""
1. Weekly Data Preprocessing
Mirrors: Weekly_Modelling/Script/1. Weekly Data Preprocessing.R

Reads raw data from the shared R input directory and writes processed files
to Weekly_Modelling_Python/Input/ so both pipelines are independently runnable.

Run order: this script first, then seasonal_model_training.py or
           2_weekly_model_predictions.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    HISTORICAL_ONLY_COLS,
    INPUT_DIR,
    TOUR_PREPROCESSING,
)


# ===== DATA LOADING =====

def load_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Rename columns starting with '_' (R convention: `_Top5` → stored as _Top5)
    df.columns = [f"X{c}" if c.startswith("_") else c for c in df.columns]

    # Coerce known numeric columns that may arrive as strings
    for col in ["yr3_All", "rating", "current", "X_1yr", "X_6m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def detect_data_type(df: pd.DataFrame) -> str:
    required = {"eventID", "posn", "Date", "playerID"}
    return "historical" if required.issubset(df.columns) else "weekly"


# ===== TARGET VARIABLES (historical only) =====

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["top_40"] = (df["posn"] <= 40).astype(int)
    df["top_20"] = (df["posn"] <= 20).astype(int)
    df["top_10"] = (df["posn"] <= 10).astype(int)
    df["top_5"]  = (df["posn"] <= 5).astype(int)
    df["win"]    = (df["posn"] == 1).astype(int)
    print("  Created target variables (win / top_5 / top_10 / top_20 / top_40)")
    return df


# ===== EVENT-RELATIVE FEATURES =====

def _event_relative_rating(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Within-event rating features using vectorised groupby."""
    g = df.groupby(group_col)["rating"]

    field_mean = g.transform("mean")
    field_std  = g.transform("std").clip(lower=1e-8)

    df["rating_vs_field_mean"]    = df["rating"] - field_mean
    df["rating_vs_field_median"]  = df["rating"] - g.transform("median")
    df["rating_vs_field_best"]    = df["rating"] - g.transform("max")
    df["rating_vs_field_worst"]   = df["rating"] - g.transform("min")
    df["rating_field_zscore"]     = (df["rating"] - field_mean) / field_std
    df["rating_field_percentile"] = g.transform(
        lambda x: x.rank(method="average") / len(x)
    )
    df["field_size"]     = g.transform("count")
    df["field_strength"] = field_mean
    df["field_depth"]    = field_std
    return df


def create_event_relative_features(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    if "rating" not in df.columns:
        return df

    df = df.copy()
    if data_type == "weekly":
        df["_tmp_event"] = "__weekly__"
        df = _event_relative_rating(df, "_tmp_event")
        df = df.drop(columns=["_tmp_event"])
    else:
        df = _event_relative_rating(df, "eventID")

    return df


# ===== STROKES GAINED FEATURES =====

def _sg_relatives(df: pd.DataFrame, col: str, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[col]
    field_mean = g.transform("mean")
    field_std  = g.transform("std").clip(lower=1e-8)

    df[f"{col}_vs_field_mean"]    = df[col] - field_mean
    df[f"{col}_vs_field_median"]  = df[col] - g.transform("median")
    df[f"{col}_vs_field_best"]    = df[col] - g.transform("max")
    df[f"{col}_field_zscore"]     = (df[col] - field_mean) / field_std
    df[f"{col}_field_percentile"] = g.transform(
        lambda x: x.rank(method="average") / len(x)
    )
    return df


def create_sg_features(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    df = df.copy()

    sg_base = [c for c in ["sgtee", "sgt2g", "sgapp", "sgatg", "sgp"] if c in df.columns]

    if "sgtee" in df.columns and "sgapp" in df.columns:
        df["sg_ball_striking"] = df["sgtee"] + df["sgapp"]
    if "sgatg" in df.columns and "sgp" in df.columns:
        df["sg_short_game"] = df["sgatg"] + df["sgp"]

    all_sg = sg_base + [c for c in ["sg_ball_striking", "sg_short_game"] if c in df.columns]

    if not all_sg:
        return df

    if data_type == "weekly":
        df["_tmp_event"] = "__weekly__"
        for col in all_sg:
            df = _sg_relatives(df, col, "_tmp_event")
        df = df.drop(columns=["_tmp_event"])
    else:
        for col in all_sg:
            df = _sg_relatives(df, col, "eventID")

    return df


# ===== MAIN PREPROCESSING FUNCTION =====

def preprocess(input_path: Path, output_path: Path, label: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    df = load_file(input_path)
    data_type = detect_data_type(df)
    print(f"  Data type: {data_type}  |  {len(df):,} rows  |  {df.shape[1]} columns")

    # Drop historical-only columns that cause missing-value problems
    drop_cols = [c for c in HISTORICAL_ONLY_COLS if c in df.columns]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} historical-only columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    df = df.dropna()
    print(f"  After dropna: {len(df):,} rows")

    if data_type == "historical":
        df = create_target_variables(df)

    df = create_event_relative_features(df, data_type)
    df = create_sg_features(df, data_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"  Saved: {len(df):,} rows  |  {df.shape[1]} columns  →  {output_path.name}")
    return df


# ===== MAIN =====

def main():
    print("=== GOLF DATA PREPROCESSING - PGA & EUROPEAN TOURS ===")
    print(f"Reading raw data from: {list(TOUR_PREPROCESSING.values())[0]['historical_input'].parent}")
    print(f"Writing processed data to: {INPUT_DIR}")

    results = {}
    for tour_key, cfg in TOUR_PREPROCESSING.items():
        results[tour_key] = {}

        if cfg["historical_input"].exists():
            results[tour_key]["historical"] = preprocess(
                cfg["historical_input"], cfg["historical_output"],
                f"{cfg['name']} - Historical"
            )
        else:
            print(f"\nHistorical file not found: {cfg['historical_input']}")

        if cfg["weekly_input"].exists():
            results[tour_key]["weekly"] = preprocess(
                cfg["weekly_input"], cfg["weekly_output"],
                f"{cfg['name']} - Weekly"
            )
        else:
            print(f"\nWeekly file not found (skipping): {cfg['weekly_input']}")

    print("\n=== PREPROCESSING COMPLETE ===")
    for tour_key, res in results.items():
        name = TOUR_PREPROCESSING[tour_key]["name"]
        print(f"\n{name}:")
        for dtype, df in res.items():
            print(f"  {dtype}: {len(df):,} rows  |  {df.shape[1]} columns")

    print(f"\nProcessed files saved to: {INPUT_DIR}")
    print("Ready for modelling.\n")


if __name__ == "__main__":
    main()
