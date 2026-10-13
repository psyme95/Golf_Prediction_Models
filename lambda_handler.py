"""AWS Lambda entry point (template.yaml). S3 is the bots' disk.

Every run copies the bots' state from S3 into /tmp/golf (GOLF_ROOT), runs both
paper-trading bots, and writes back the files that changed. A run triggered by
uploading Input/This_Week_<TOUR>.csv first preprocesses and predicts that tour
and loads the model odds into the lay bot, as analysis/start_week.py does locally.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3

BUCKET = os.environ["BUCKET"]
ROOT = Path(os.environ["GOLF_ROOT"])
CODE = Path(__file__).resolve().parent
STATE = ["Output/PaperTrades/", "Output/LayTrades/"]
WEEKLY = "Input/This_Week_"

s3 = boto3.client("s3")
seen = {}   # S3 key -> md5 when downloaded, so unchanged files aren't uploaded again


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            path = ROOT / obj["Key"]
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, obj["Key"], str(path))
            seen[obj["Key"]] = md5(path)


def upload(prefix: str):
    for path in (ROOT / prefix).rglob("*"):
        key = path.relative_to(ROOT).as_posix()
        if path.is_file() and seen.get(key) != md5(path):
            print("upload", key, flush=True)
            s3.upload_file(str(path), BUCKET, key)


def run(*args) -> bool:
    print(">", *args, flush=True)
    return subprocess.run([sys.executable, *args], cwd=CODE).returncode == 0


def handler(event, context):
    shutil.rmtree(ROOT, ignore_errors=True)   # warm containers keep /tmp; S3 is the truth
    seen.clear()
    for prefix in STATE:
        download(prefix)
    if "Output/PaperTrades/paper_trades.csv" not in seen:
        raise RuntimeError(f"No bot state in s3://{BUCKET}/Output/: copy it in first (README).")

    ok = True
    key = event.get("detail", {}).get("object", {}).get("key", "")
    if key.startswith(WEEKLY):
        tour = key.removeprefix(WEEKLY).removesuffix(".csv")
        download(key)
        download(f"Output/Models/{tour}_Trained_Models_")
        ok = (run("-m", "golfmodel", "preprocess", "--kind", "weekly", "--tour", tour)
              and run("-m", "golfmodel", "predict", "--tour", tour))
        preds = list((ROOT / "Output" / "Predictions").glob(f"{tour}_Predictions_*.xlsx"))
        ok = ok and len(preds) == 1 and run("analysis/lay_trader.py", "load", str(preds[0]))

    ok &= run("analysis/paper_trader.py", "bot", "--once")
    ok &= run("analysis/lay_trader.py", "bot")
    for prefix in STATE + ["Output/Predictions/"]:
        upload(prefix)
    if not ok:
        raise RuntimeError("A step failed: see the log above.")
