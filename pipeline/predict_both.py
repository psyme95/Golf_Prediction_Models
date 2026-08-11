"""Weekly predictions from BOTH models — old pipeline and new golfmodel.

Temporary side-by-side runner for the paper-testing period; delete this file
when the old pipeline is deprecated.

What it does:
  1. Syncs This_Week_*.csv between pipeline/Input and Weekly_Modelling_Python/
     Input (newest wins), so both pipelines see the same field data whichever
     folder you dropped the files in.
  2. Old model: weekly preprocessing + prediction via the legacy scripts.
  3. New model: golfmodel preprocess --kind weekly + predict.
  4. Copies the old model's workbooks into pipeline/Output/Predictions with an
     _OldModel suffix, so all outputs for the week sit in one folder.

Run:  py -3.13 predict_both.py     (from pipeline/)
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
OLD_DIR      = PIPELINE_DIR.parent / "Weekly_Modelling_Python"
OLD_SCRIPT   = OLD_DIR / "Script"
NEW_INPUT    = PIPELINE_DIR / "Input"
OLD_INPUT    = OLD_DIR / "Input"
PRED_DIR     = PIPELINE_DIR / "Output" / "Predictions"

WEEKLY_CSVS = ["This_Week_PGA.csv", "This_Week_Euro.csv"]

# Old scripts may print characters a cp1252 console can't encode.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8:replace"}


def run(cmd, cwd, label):
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, cwd=cwd, env=ENV)
    if result.returncode != 0:
        sys.exit(f"FAILED: {label} (exit {result.returncode})")


def sync_weekly_csvs():
    """Copy each weekly CSV to whichever folder has the older/missing copy."""
    for name in WEEKLY_CSVS:
        a, b = NEW_INPUT / name, OLD_INPUT / name
        if not a.exists() and not b.exists():
            print(f"  {name}: not found in either Input folder — skipping")
            continue
        if not a.exists() or (b.exists() and b.stat().st_mtime > a.stat().st_mtime):
            shutil.copy2(b, a)
            print(f"  {name}: old Input → pipeline Input")
        elif not b.exists() or a.stat().st_mtime > b.stat().st_mtime:
            shutil.copy2(a, b)
            print(f"  {name}: pipeline Input → old Input")
        else:
            print(f"  {name}: in sync")


def main():
    print("=== SIDE-BY-SIDE WEEKLY PREDICTIONS (old + new model) ===")
    sync_weekly_csvs()

    # --- Old model: selective weekly preprocessing, then prediction ---
    old_preprocess = (
        "import importlib\n"
        "m = importlib.import_module('1_weekly_data_preprocessing')\n"
        "from config import TOUR_PREPROCESSING\n"
        "for c in TOUR_PREPROCESSING.values():\n"
        "    if c['weekly_input'].exists():\n"
        "        m.preprocess(c['weekly_input'], c['weekly_output'], c['name'] + ' - Weekly')\n"
    )
    run([sys.executable, "-c", old_preprocess], OLD_SCRIPT, "Old model: weekly preprocessing")
    run([sys.executable, "2_weekly_model_predictions.py"], OLD_SCRIPT, "Old model: predictions")

    # --- New model ---
    run([sys.executable, "-m", "golfmodel", "preprocess", "--kind", "weekly"],
        PIPELINE_DIR, "New model: weekly preprocessing")
    run([sys.executable, "-m", "golfmodel", "predict"], PIPELINE_DIR, "New model: predictions")

    # --- Gather old outputs next to the new ones ---
    date_str = datetime.now().strftime("%d-%m-%Y")
    print("\n=== OUTPUT WORKBOOKS ===")
    for tour in ["PGA", "Euro"]:
        old_out = OLD_DIR / "Output" / "Predictions" / f"{tour}_Predictions_{date_str}.xlsx"
        new_out = PRED_DIR / f"{tour}_Predictions_{date_str}.xlsx"
        if old_out.exists():
            dest = PRED_DIR / f"{tour}_Predictions_{date_str}_OldModel.xlsx"
            shutil.copy2(old_out, dest)
            print(f"  {tour} old model: {dest}")
        if new_out.exists():
            print(f"  {tour} new model: {new_out}")


if __name__ == "__main__":
    main()
