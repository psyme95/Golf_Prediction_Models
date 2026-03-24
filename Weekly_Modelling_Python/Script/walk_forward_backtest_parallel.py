"""
Parallel Walk-Forward Launcher
================================
Runs walk_forward_backtest.py for PGA and Euro tours simultaneously as
separate subprocesses. All CLI arguments are forwarded to both processes.

Usage (mirrors walk_forward_backtest.py exactly):
  python run_parallel.py                        # both tours, all windows, 30 trials
  python run_parallel.py --trials 75            # full Optuna
  python run_parallel.py --start-year 2022
  python run_parallel.py --min-year 2024
  python run_parallel.py --force-retrain
  python run_parallel.py                # single tour, no parallelism

Output is written to per-tour log files and also streamed live to the terminal
with a [PGA] / [Euro] prefix so interleaved lines remain readable.

Log files:
  Output/WalkForward/Logs/PGA_run.log
  Output/WalkForward/Logs/Euro_run.log
"""

import argparse
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT      = Path(__file__).parent / "walk_forward_backtest.py"
LOG_DIR     = Path(__file__).parent / "Output" / "WalkForward" / "Logs"
ALL_TOURS   = ["PGA", "Euro"]

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk_forward_backtest.py for both tours in parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Mirror the backtest script's arguments so they can be forwarded verbatim.
    parser.add_argument("--tour",         default=None)
    parser.add_argument("--trials",       type=int, default=None)
    parser.add_argument("--start-year",   type=int, default=None, dest="start_year")
    parser.add_argument("--min-year",     type=int, default=None, dest="min_year")
    parser.add_argument("--force-retrain",action="store_true",    dest="force_retrain")
    return parser.parse_args()


def build_forward_args(args) -> list[str]:
    """Reconstruct CLI args to forward to each subprocess (tour arg excluded)."""
    fwd = []
    if args.trials      is not None: fwd += ["--trials",     str(args.trials)]
    if args.start_year  is not None: fwd += ["--start-year", str(args.start_year)]
    if args.min_year    is not None: fwd += ["--min-year",   str(args.min_year)]
    if args.force_retrain:           fwd += ["--force-retrain"]
    return fwd


# ── Live output streaming ─────────────────────────────────────────────────────

def stream_output(stream, label: str, log_fh, lock: threading.Lock):
    """
    Read lines from a subprocess stdout/stderr stream, write to log file,
    and echo to terminal with a tour label prefix.
    Runs in its own thread — one per stream per process.
    """
    prefix = f"[{label}] "
    for raw_line in stream:
        line = raw_line.rstrip("\n")
        with lock:
            print(f"{prefix}{line}", flush=True)
            log_fh.write(line + "\n")
            log_fh.flush()


# ── Process management ────────────────────────────────────────────────────────

def launch_tour(tour: str, forward_args: list[str], log_path: Path,
                lock: threading.Lock) -> tuple[subprocess.Popen, list[threading.Thread]]:
    """
    Spawn a subprocess for one tour and start reader threads for its
    stdout and stderr streams.

    Returns the Popen object and the list of reader threads so the caller
    can join them after the process exits.
    """
    cmd = [sys.executable, str(SCRIPT), "--tour", tour] + forward_args
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")

    # Write a header into the log so it's clear when the run started
    header = f"=== {tour} started at {datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
    log_fh.write(header)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout — one stream to read
        text=True,
        bufsize=1,                  # line-buffered
    )

    reader = threading.Thread(
        target=stream_output,
        args=(proc.stdout, tour, log_fh, lock),
        daemon=True,
        name=f"reader-{tour}",
    )
    reader.start()

    return proc, [reader], log_fh


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    forward     = build_forward_args(args)
    print_lock  = threading.Lock()

    # Single-tour path: no parallelism needed, just forward directly
    if args.tour is not None:
        if args.tour not in ALL_TOURS:
            print(f"ERROR: unknown tour '{args.tour}'. Choose from: {ALL_TOURS}")
            sys.exit(1)
        tours = [args.tour]
    else:
        tours = ALL_TOURS

    if len(tours) == 1:
        print(f"Single tour specified ({tours[0]}) — running directly, no parallelism.")
        cmd = [sys.executable, str(SCRIPT), "--tour", tours[0]] + forward
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # ── Parallel path ─────────────────────────────────────────────────────────
    start_time = datetime.now()
    print(f"Launching {len(tours)} tours in parallel: {', '.join(tours)}")
    print(f"Forwarded args: {' '.join(forward) if forward else '(none)'}")
    print(f"Log directory:  {LOG_DIR.resolve()}")
    print(f"Started: {start_time:%Y-%m-%d %H:%M:%S}\n")

    processes   = {}   # tour → Popen
    reader_threads = {}  # tour → [Thread]
    log_handles = {}   # tour → file handle (kept open until reader thread finishes)

    for tour in tours:
        log_path = LOG_DIR / f"{tour}_run.log"
        proc, readers, log_fh = launch_tour(tour, forward, log_path, print_lock)
        processes[tour]      = proc
        reader_threads[tour] = readers
        log_handles[tour]    = log_fh

    # Wait for all processes to finish
    return_codes = {}
    for tour, proc in processes.items():
        proc.wait()
        # Join reader threads so all output is flushed before we continue
        for t in reader_threads[tour]:
            t.join()
        log_handles[tour].close()
        return_codes[tour] = proc.returncode

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    minutes, seconds = divmod(int(elapsed), 60)

    print(f"\n{'='*50}")
    print(f"PARALLEL RUN COMPLETE  ({minutes}m {seconds}s)")
    print(f"{'='*50}")
    for tour in tours:
        rc     = return_codes[tour]
        status = "OK" if rc == 0 else f"FAILED (exit code {rc})"
        log    = LOG_DIR / f"{tour}_run.log"
        print(f"  {tour:<6}  {status:<25}  log: {log}")

    # Exit non-zero if any tour failed
    if any(rc != 0 for rc in return_codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()