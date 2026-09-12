#!/usr/bin/env python3
"""Recompute the headline results and compare them against a stored baseline.

This is a regression check over the tracked records. It recomputes the totals,
the win/loss/tie counts, the gaps and the paired tests from
``experiments/result_tables/``, then compares every value against
``experiments/expected_results.json``. It prints one line per check and exits
non-zero if any of them moved.

What it proves: the records in the repository still produce the numbers they
produced when the baseline was written. What it does not prove: that those
numbers are right. The baseline is generated from the same records, so it
guards against drift in the data or in the analysis code, not against an error
in the experiments.

    python export_result_tables.py        # produce the tables first
    python verify_results.py
    python verify_results.py --update     # rewrite the baseline after a change

The evaluation set is the instance set shared by every batch: the instances
present in the weak-initialization run, which is the smallest batch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
DEFAULT_TABLES = "experiments/result_tables"
DEFAULT_BASELINE = "experiments/expected_results.json"

#: Relative tolerance for float comparisons. Integers are compared exactly.
TOLERANCE = 1e-9

#: Ablation batches, each one its own test family.
ABLATIONS = {
    "component_ablation": "component_ablation_seed_results.csv",
    "operator_group_ablation": "operator_group_ablation_seed_results.csv",
}


def exact_wilcoxon(reference: pd.Series, comparison: pd.Series) -> float:
    """Two-sided exact signed-rank test over the nonzero paired differences.

    The zero differences are dropped before the call. Leaving them in makes
    scipy refuse the exact null and fall back to the normal approximation,
    which gives a different p-value at any sample size worth reporting.
    """
    difference = (reference - comparison).dropna()
    nonzero = difference[difference != 0]
    if nonzero.empty:
        return 1.0
    try:
        _statistic, p_value = wilcoxon(nonzero, alternative="two-sided",
                                       zero_method="wilcox", method="exact")
    except TypeError:  # scipy < 1.9 spells the selector "mode"
        _statistic, p_value = wilcoxon(nonzero, alternative="two-sided",
                                       zero_method="wilcox", mode="exact")
    return float(p_value)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction over one family of tests."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    size = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, (size - rank) * p_value)
        adjusted[name] = min(1.0, running)
    return adjusted


def gap_pct(reference: float, comparison: float) -> float:
    """Relative gap in percent. Negative means the reference scores higher."""
    if comparison == 0:
        return 0.0
    return (comparison - reference) / comparison * 100.0


def per_instance(frame: pd.DataFrame, column: str = "final_score") -> pd.DataFrame:
    """Collapse the seeds of each instance to its mean and its best."""
    grouped = frame.groupby(["dataset", "instance_name"])[column]
    return pd.DataFrame({"mean": grouped.mean(), "best": grouped.max()})


def compute(tables: Path) -> dict[str, object]:
    final = pd.read_csv(tables / "final_seed_results.csv")
    weak = pd.read_csv(tables / "weak_sorted_seed_results.csv")
    external = pd.read_csv(tables / "external_baseline_results.csv")

    evaluation = set(weak["instance_name"].unique())
    final_eval = final[final["instance_name"].isin(evaluation)]
    ils = per_instance(final_eval)

    portfolio = external[
        (external["algorithm"] == "BestExternal")
        & (external["instance_name"].isin(evaluation))
    ].set_index(["dataset", "instance_name"])["final_score"]

    paired = ils.join(portfolio.rename("portfolio"), how="inner")
    difference = paired["best"] - paired["portfolio"]

    results: dict[str, object] = {
        "evaluation_instances": len(evaluation),
        "ils_total_best_of_ten": int(ils["best"].sum()),
        "ils_total_run_means": float(ils["mean"].sum()),
        "portfolio_total": int(paired["portfolio"].sum()),
        "portfolio_better": int((difference > 0).sum()),
        "portfolio_worse": int((difference < 0).sum()),
        "portfolio_equal": int((difference == 0).sum()),
        "portfolio_gap_pct": gap_pct(paired["best"].sum(), paired["portfolio"].sum()),
        "portfolio_p_exact": exact_wilcoxon(paired["best"], paired["portfolio"]),
    }

    for family, name in ABLATIONS.items():
        table = pd.read_csv(tables / name)
        table = table[table["instance_name"].isin(evaluation)]
        raw: dict[str, float] = {}
        for variant, subset in table.groupby("variant"):
            variant_means = per_instance(subset)["mean"]
            paired_means = ils["mean"].loc[variant_means.index]
            raw[str(variant)] = exact_wilcoxon(paired_means, variant_means)
        adjusted = holm(raw)
        results[family] = {
            "family_size": len(raw),
            "p_exact": raw,
            "p_holm": adjusted,
        }

    weak_means = per_instance(weak)["mean"]
    results["weak_start"] = {
        "p_exact": exact_wilcoxon(ils["mean"].loc[weak_means.index], weak_means),
        "gap_pct": gap_pct(ils["mean"].sum(), weak_means.sum()),
    }
    return results


def flatten(values: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    if isinstance(values, dict):
        for key, value in values.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = values
    return flat


def same(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return math.isclose(float(left), float(right), rel_tol=TOLERANCE,
                            abs_tol=TOLERANCE)
    return left == right


def report(computed: dict[str, object], expected: dict[str, object]) -> int:
    current = flatten(computed)
    baseline = flatten(expected)
    failures = []
    for key in sorted(set(current) | set(baseline)):
        got = current.get(key)
        want = baseline.get(key)
        if key not in baseline:
            failures.append((key, got, "not in baseline"))
            print(f"  [NEW ] {key:<52} {got}")
        elif key not in current:
            failures.append((key, "missing", want))
            print(f"  [GONE] {key:<52} expected {want}")
        elif same(got, want):
            print(f"  [ok  ] {key:<52} {got}")
        else:
            failures.append((key, got, want))
            print(f"  [FAIL] {key:<52} {got}  (expected {want})")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for key, got, want in failures:
            print(f"  {key}: got {got}, expected {want}")
        return 1
    print("All results reproduce from the tracked records.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the headline results and compare them to the baseline."
    )
    parser.add_argument("--tables-dir", default=DEFAULT_TABLES)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write the computed values as the new baseline instead of checking.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    computed = compute(ROOT / args.tables_dir)
    baseline_path = ROOT / args.baseline

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(computed, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"Baseline written to {args.baseline}")
        return 0

    if not baseline_path.exists():
        print(f"No baseline at {args.baseline}. Run with --update to create it.")
        return 1
    expected = json.loads(baseline_path.read_text(encoding="utf-8"))
    return report(computed, expected)


if __name__ == "__main__":
    raise SystemExit(main())
