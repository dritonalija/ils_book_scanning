#!/usr/bin/env python3
"""Summarize the instrumented convergence runs.

Each convergence run writes two logs: a score trace and a table of operator
counters. This script reads the run manifest, matches every run to its logs,
and writes two summaries:

``convergence_checkpoint_summary.csv``
    The best score reached at fixed points on the search clock, plus how far
    that is along the run's total improvement.

``convergence_operator_summary.csv``
    Attempts and accepted moves per operator, summed over the runs.

The score trace measures construction and search on separate clocks, so the
checkpoints use the search clock only: the construction and initial local
search rows are skipped, and the score at checkpoint 0 is the score the
construction handed over.

    python analyze_convergence.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

DEFAULT_MANIFEST = "experiments/output_csv/convergence/experiment_results.csv"
DEFAULT_LOGS = "experiments/convergence_logs"
DEFAULT_OUTPUT = "experiments/analysis/convergence"
DEFAULT_CHECKPOINTS = [0, 30, 60, 120, 300, 600]

#: Phases that run before the search clock starts.
PRE_SEARCH_PHASES = {"construction", "initial_ls"}

OPERATOR_COUNTERS = ["attempts", "proxy_improved", "exact_checked", "accepted", "rejected"]


def log_path(logs_dir: Path, log_csv: str) -> Path:
    """Map a manifest ``log_csv`` entry onto the tracked log directory.

    The manifest records the path the run wrote to, which was outside version
    control. Only the part below that directory is kept.
    """
    relative = str(log_csv).replace("\\", "/")
    marker = "logs/convergence/"
    if marker in relative:
        relative = relative.split(marker, 1)[1]
    return logs_dir / relative


def read_trace(path: Path) -> tuple[float, float, list[tuple[float, float]]]:
    """Return the handover score, the final score, and the search trace.

    The trace is a list of ``(elapsed_s, best_score)`` pairs on the search
    clock, in file order.
    """
    initial = None
    final = None
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            best = float(row["best_score"])
            if initial is None:
                initial = best
            final = best
            if row["phase"] in PRE_SEARCH_PHASES:
                continue
            points.append((float(row["elapsed_s"]), best))
    if initial is None:
        raise ValueError(f"empty trace: {path}")
    return initial, final, points


def score_at(initial: float, points: list[tuple[float, float]], checkpoint: float) -> float:
    """Best score at a checkpoint, holding the last value seen before it."""
    score = initial
    for elapsed, best in points:
        if elapsed > checkpoint:
            break
        score = best
    return score


def checkpoint_summary(manifest: pd.DataFrame, logs_dir: Path,
                       checkpoints: list[int]) -> pd.DataFrame:
    rows = []
    for _index, run in manifest.iterrows():
        path = log_path(logs_dir, run["log_csv"])
        if not path.exists():
            raise FileNotFoundError(f"missing trace for {run['instance']} seed {run['seed']}: {path}")
        initial, final, points = read_trace(path)
        span = final - initial
        for checkpoint in checkpoints:
            score = score_at(initial, points, checkpoint)
            # A run that never improved has no scale to measure progress on.
            progress = (score - initial) / span if span > 0 else None
            rows.append({
                "dataset": run["dataset"],
                "instance": run["instance"],
                "seed": run["seed"],
                "checkpoint_s": checkpoint,
                "score_at_checkpoint": score,
                "progress_to_final_improvement": progress,
            })
    return pd.DataFrame(rows)


def operator_summary(manifest: pd.DataFrame, logs_dir: Path) -> pd.DataFrame:
    totals: dict[str, dict[str, int]] = {}
    runs: dict[str, int] = {}
    for _index, run in manifest.iterrows():
        trace = log_path(logs_dir, run["log_csv"])
        path = trace.parent / (trace.stem + ".operators.csv")
        if not path.exists():
            raise FileNotFoundError(f"missing operator log: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                operator = row["operator"]
                bucket = totals.setdefault(operator, dict.fromkeys(OPERATOR_COUNTERS, 0))
                for counter in OPERATOR_COUNTERS:
                    bucket[counter] += int(row.get(counter) or 0)
                runs[operator] = runs.get(operator, 0) + 1

    rows = []
    for operator, bucket in totals.items():
        attempts = bucket["attempts"]
        exact_checked = bucket["exact_checked"]
        accepted = bucket["accepted"]
        run_count = runs[operator]
        rows.append({
            "operator": operator,
            "runs": run_count,
            **bucket,
            "mean_attempts_per_run": attempts / run_count if run_count else 0.0,
            "acceptance_rate": accepted / attempts if attempts else 0.0,
            "exact_acceptance_rate": accepted / exact_checked if exact_checked else None,
        })
    summary = pd.DataFrame(rows).sort_values("accepted", ascending=False)
    return summary.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the instrumented convergence runs."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=DEFAULT_CHECKPOINTS,
        help="Points on the search clock, in seconds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(ROOT / args.manifest)
    logs_dir = ROOT / args.logs_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = checkpoint_summary(manifest, logs_dir, args.checkpoints)
    checkpoints.to_csv(output_dir / "convergence_checkpoint_summary.csv", index=False)
    print(f"{len(checkpoints)} checkpoint rows -> {args.output_dir}/convergence_checkpoint_summary.csv")

    operators = operator_summary(manifest, logs_dir)
    operators.to_csv(output_dir / "convergence_operator_summary.csv", index=False)
    print(f"{len(operators)} operators -> {args.output_dir}/convergence_operator_summary.csv")


if __name__ == "__main__":
    main()
