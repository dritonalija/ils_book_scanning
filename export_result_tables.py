#!/usr/bin/env python3
"""Turn the raw run exports into tidy per-seed result tables.

Every experiment batch writes one wide ``experiment_results.csv`` with run
provenance, environment and file-path columns. This script keeps the successful
runs, trims each batch to the columns that describe the result, and sorts the
rows so the output is stable across machines.

One table per batch is written to ``--output-dir``:

    final_seed_results.csv                    tuned algorithm, full run set
    weak_sorted_seed_results.csv              weak-initialization variant
    component_ablation_seed_results.csv       component ablation
    operator_group_ablation_seed_results.csv  operator-group ablation
    convergence_seed_results.csv              instrumented convergence runs
    external_baseline_results.csv             external methods

Instance paths are reduced to file names, so the tables do not carry the
directory layout of the machine that produced them.

    python export_result_tables.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = "experiments/output_csv"
DEFAULT_OUTPUT = "experiments/result_tables"

RUN_COLUMNS = ["initial_score", "final_score", "improvement_pct", "elapsed_s"]

#: batch -> (source CSV, keeps a variant column, output name)
#:
#: An ablation batch runs several variants over the same instances, and
#: ``run_label`` is the column that names the variant in every batch. The
#: dedicated ``variant`` and ``component`` columns are not usable for this:
#: in the operator-group batch both hold one constant value for all runs.
BATCHES = {
    "final": (
        "experiments_full/experiment_results.csv",
        False,
        "final_seed_results.csv",
    ),
    "weak_sorted": (
        "weak_sorted_init/experiment_results.csv",
        False,
        "weak_sorted_seed_results.csv",
    ),
    "component_ablation": (
        "component_ablation/experiment_results.csv",
        True,
        "component_ablation_seed_results.csv",
    ),
    "operator_group_ablation": (
        "operator_group_ablation/experiment_results.csv",
        True,
        "operator_group_ablation_seed_results.csv",
    ),
    "convergence": (
        "convergence/experiment_results.csv",
        False,
        "convergence_seed_results.csv",
    ),
}

EXTERNAL = (
    "external_baselines/experiment_results.csv",
    ["dataset", "instance_name", "algorithm", "run_label"],
    ["dataset", "instance_name", "algorithm", "run_label", "seed",
     "final_score", "elapsed_s"],
    "external_baseline_results.csv",
)


def instance_name(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def add_instance_basename(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "instance" in frame.columns:
        frame["instance_name"] = frame["instance"].map(instance_name)
    return frame


def load_successful(path: Path) -> pd.DataFrame:
    """Read a run export and keep the runs that completed."""
    frame = add_instance_basename(pd.read_csv(path))
    return frame[frame["status"] == "ok"].copy()


def export_batch(source: Path, has_variants: bool) -> pd.DataFrame:
    frame = load_successful(source)
    group_columns = []
    if has_variants:
        frame["variant"] = frame["run_label"]
        group_columns = ["variant"]
    sort_keys = ["dataset", "instance_name", *group_columns, "seed"]
    columns = ["dataset", "instance_name", *group_columns, "seed", *RUN_COLUMNS]
    return frame.sort_values(sort_keys)[columns]


def export_external(source: Path) -> pd.DataFrame:
    _name, sort_keys, columns, _output = EXTERNAL
    frame = load_successful(source).sort_values(sort_keys)
    return frame[columns]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export tidy per-seed result tables from the run exports."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch",
        nargs="+",
        choices=[*BATCHES, "external"],
        default=[*BATCHES, "external"],
        help="Batches to export. Defaults to all of them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in args.batch:
        if batch == "external":
            source_name, _sort, _columns, output_name = EXTERNAL
            table = export_external(input_dir / source_name)
        else:
            source_name, has_variants, output_name = BATCHES[batch]
            table = export_batch(input_dir / source_name, has_variants)
        table.to_csv(output_dir / output_name, index=False)
        print(f"{len(table):>5} rows -> {args.output_dir}/{output_name}")


if __name__ == "__main__":
    main()
