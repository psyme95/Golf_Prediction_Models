"""Golf prediction pipeline CLI.

  python -m golfmodel preprocess  [--tour PGA|Euro]
  python -m golfmodel train       [--tour ...] [--trials N]
  python -m golfmodel predict     [--tour ...]
  python -m golfmodel season      [--tour ...] [--year Y] [--no-grid]
  python -m golfmodel walkforward [--tour ...] [--trials N] [--start-year Y]
                                  [--min-test-year Y] [--force-retrain] [--parallel]
  python -m golfmodel paper       --date dd-mm-yyyy
"""

import argparse
import subprocess
import sys
import warnings
from datetime import datetime

import pandas as pd

from .config import LOGS_DIR, OPTUNA_TRIALS, RAW_DIR, TOURS

warnings.filterwarnings("ignore")

# Consoles on legacy code pages (cp1252) can't encode characters like '→';
# degrade them to '?' instead of crashing with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

WF_DEFAULT_TRIALS = 30   # walk-forward default (vs production trials for train)


def _tours(args) -> list[str]:
    return [args.tour] if args.tour else list(TOURS)


def cmd_preprocess(args):
    from .data import preprocess
    kinds = [args.kind] if args.kind else ["historical", "weekly"]
    for tour in _tours(args):
        for kind in kinds:
            preprocess(tour, kind)


def cmd_train(args):
    from .modeling import train_tour
    for tour in _tours(args):
        train_tour(tour, n_trials=args.trials)


def cmd_predict(args):
    from .modeling import predict_weekly
    for tour in _tours(args):
        predict_weekly(tour)


def cmd_season(args):
    from .backtest import run_static
    for tour in _tours(args):
        run_static(tour, args.year, no_grid=args.no_grid)


def cmd_walkforward(args):
    if args.parallel and not args.tour:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        procs = {}
        for tour in TOURS:
            cmd = [sys.executable, "-u", "-m", "golfmodel", "walkforward", "--tour", tour,
                   "--trials", str(args.trials)]   # -u: unbuffered → logs stream live
            if args.start_year:    cmd += ["--start-year", str(args.start_year)]
            if args.min_test_year: cmd += ["--min-test-year", str(args.min_test_year)]
            if args.force_retrain: cmd += ["--force-retrain"]
            if args.tag:           cmd += ["--tag", args.tag]
            log_path = LOGS_DIR / f"{tour}_walkforward.log"
            log = log_path.open("w", encoding="utf-8")
            procs[tour] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"Launched {tour} → {log_path}")
        failed = False
        for tour, (proc, log) in procs.items():
            proc.wait()
            log.close()
            status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
            print(f"{tour}: {status}")
            failed |= proc.returncode != 0
        sys.exit(1 if failed else 0)

    from .backtest import run_walkforward
    for tour in _tours(args):
        run_walkforward(tour, n_trials=args.trials, start_year=args.start_year,
                        min_test_year=args.min_test_year,
                        force_retrain=args.force_retrain, tag=args.tag)


def cmd_paper(args):
    """Summarise a paper-testing workbook. BACK/LAY profit totals live in the
    header row next to a 'Profit:' label — located by label, not position."""
    markets = {"Winner": ("Winner_Market", 1000), "Top 5": ("Top5_Market", 200),
               "Top 10": ("Top10_Market", 100), "Top 20": ("Top20_Market", 50)}
    paper_dir = RAW_DIR.parent / "Output" / "Paper_Testing"

    for tour in TOURS:
        path = paper_dir / f"{tour}_{args.date}.xlsx"
        if not path.exists():
            print(f"{tour}: no file for {args.date} ({path.name})")
            continue

        rows = []
        for market, (sheet, max_liability) in markets.items():
            df = pd.read_excel(path, sheet_name=sheet)

            header = list(df.columns)
            try:
                i = header.index("Profit:")
                back_profit = header[header.index("BACK", i) + 1]
                lay_profit  = header[header.index("LAY", i) + 1]
            except ValueError:
                print(f"  {tour}/{market}: no 'Profit:' block in header — skipped")
                continue

            fixed_stake = round(max_liability / df["Normalised_Model_Odds"].max(), 2)
            lay = df[df["Bet"] == "LAY"]
            unit_profit = (lay["Profit"] / lay["Stake"]).sum()
            rows.append({
                "Market": market,
                "Back Profit": back_profit,
                "FL Lay Profit": lay_profit,
                "£1 Stake Profit": round(unit_profit, 2),
                "FS Lay Profit": round(unit_profit * fixed_stake, 2),
            })

        summary = pd.DataFrame(rows)
        totals = summary.select_dtypes("number").sum().rename("Total")
        summary = pd.concat([summary, totals.to_frame().T.assign(Market="Total")],
                            ignore_index=True)
        print(f"\n{tour}")
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(prog="golfmodel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, **extra_args):
        p = sub.add_parser(name)
        p.add_argument("--tour", choices=list(TOURS), default=None)
        for flag, kwargs in extra_args.items():
            p.add_argument(flag, **kwargs)
        p.set_defaults(fn=fn)
        return p

    add("preprocess", cmd_preprocess, **{
        "--kind": dict(choices=["historical", "weekly"], default=None,
                       help="process only this file kind (default: both)"),
    })
    add("train", cmd_train, **{"--trials": dict(type=int, default=OPTUNA_TRIALS)})
    add("predict", cmd_predict)
    add("season", cmd_season, **{
        "--year": dict(type=int, default=datetime.now().year),
        "--no-grid": dict(action="store_true"),
    })
    add("walkforward", cmd_walkforward, **{
        "--trials": dict(type=int, default=WF_DEFAULT_TRIALS),
        "--start-year": dict(type=int, default=None),
        "--min-test-year": dict(type=int, default=None),
        "--force-retrain": dict(action="store_true"),
        "--parallel": dict(action="store_true"),
        "--tag": dict(default=None, help="segregate cache/output for A/B runs"),
    })
    p_paper = sub.add_parser("paper")
    p_paper.add_argument("--date", required=True, help="dd-mm-yyyy")
    p_paper.set_defaults(fn=cmd_paper)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
