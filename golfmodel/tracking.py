"""MLflow experiment tracking: one SQLite store under Output/, shared by every script.

Browse with:  mlflow ui --backend-store-uri sqlite:///Output/mlflow.db
"""

import os
import re
import subprocess
from contextlib import contextmanager

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import pandas as pd

from .config import OUTPUT_DIR, ROOT_DIR

TRACKING_URI = f"sqlite:///{(OUTPUT_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = OUTPUT_DIR / "mlruns"


def _git(*args) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True)
    return out.stdout.strip()


@contextmanager
def tracked_run(experiment: str, run_name: str, params: dict = None, tags: dict = None):
    """Start an MLflow run in `experiment`, tagged with the git commit and dirty flag."""
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    exp_id = (exp.experiment_id if exp else
              mlflow.create_experiment(experiment, artifact_location=(ARTIFACT_ROOT / experiment).as_uri()))
    tags = {"git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain"))), **(tags or {})}
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run


def metric_name(name: str) -> str:
    """MLflow allows only alphanumerics, _ - . / and spaces in metric names."""
    for a, b in [(">=", "_ge"), ("<=", "_le"), ("<", "_lt"), (">", "_gt"), ("%", "pct"), ("|", "_vs_"), (" ", "_")]:
        name = name.replace(a, b)
    return re.sub(r"[^A-Za-z0-9_\-./]", "_", name)


def log_table(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> None:
    """Log each numeric cell as a metric named <key values>_<metric>, e.g. Top20_2024_Lay_ROI_FixedLiab."""
    out = {}
    for _, row in df.iterrows():
        prefix = "_".join(str(row[k]) for k in keys)
        for m in metrics:
            v = pd.to_numeric(row.get(m), errors="coerce")
            if pd.notna(v):
                out[metric_name(f"{prefix}_{m}")] = float(v)
    if out:
        mlflow.log_metrics(out)
