#!/usr/bin/env python3
"""Summarize experiment CSVs and run paired statistical comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def load_results(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(normalize_columns(frame))
    if not frames:
        raise ValueError("No result CSV files were provided")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["status"].fillna("ok") == "ok"].copy()
    data["final_score"] = pd.to_numeric(data["final_score"], errors="coerce")
    data["elapsed_s"] = pd.to_numeric(data.get("elapsed_s", 0), errors="coerce")
    data["gap_to_bound_pct"] = pd.to_numeric(
        data.get("gap_to_bound_pct", pd.NA),
        errors="coerce",
    )
    data["upper_bound"] = pd.to_numeric(
        data.get("upper_bound", pd.NA),
        errors="coerce",
    )
    missing_gap = data["gap_to_bound_pct"].isna()
    can_compute_gap = missing_gap & data["upper_bound"].notna() & (data["upper_bound"] > 0)
    data.loc[can_compute_gap, "gap_to_bound_pct"] = (
        (data.loc[can_compute_gap, "upper_bound"] - data.loc[can_compute_gap, "final_score"])
        / data.loc[can_compute_gap, "upper_bound"]
        * 100.0
    )
    data = data.dropna(subset=["final_score", "instance", "method"])
    return data


def normalize_columns(frame):
    frame = frame.copy()
    if "final_score" not in frame.columns and "score" in frame.columns:
        frame["final_score"] = frame["score"]
    if "status" not in frame.columns:
        frame["status"] = "ok"
    if "seed" not in frame.columns:
        frame["seed"] = 1
    if "dataset" not in frame.columns:
        frame["dataset"] = frame["instance"].map(infer_dataset)
    if "algorithm" not in frame.columns:
        frame["algorithm"] = "external"
    if "run_label" not in frame.columns:
        frame["run_label"] = ""
    if "elapsed_s" not in frame.columns:
        frame["elapsed_s"] = 0.0
    if "gap_to_bound_pct" not in frame.columns:
        frame["gap_to_bound_pct"] = pd.NA
    if "upper_bound" not in frame.columns:
        frame["upper_bound"] = pd.NA
    frame["method"] = frame.apply(method_label, axis=1)
    return frame


def infer_dataset(instance):
    parts = str(instance).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "input":
        return parts[1]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def method_label(row):
    algorithm = str(row.get("algorithm", "")).strip()
    run_label = str(row.get("run_label", "")).strip()
    if run_label and run_label.lower() != "nan" and run_label != algorithm:
        return f"{algorithm}:{run_label}"
    return algorithm or run_label or "unknown"


def aggregate_per_instance(data, paired_metric):
    grouped = data.groupby(["dataset", "method", "instance"], dropna=False)
    summary = grouped.agg(
        runs=("final_score", "count"),
        mean_score=("final_score", "mean"),
        std_score=("final_score", "std"),
        best_score=("final_score", "max"),
        worst_score=("final_score", "min"),
        median_score=("final_score", "median"),
        mean_gap_to_bound_pct=("gap_to_bound_pct", "mean"),
        median_gap_to_bound_pct=("gap_to_bound_pct", "median"),
        mean_elapsed_s=("elapsed_s", "mean"),
    ).reset_index()
    metric_column = {
        "mean": "mean_score",
        "best": "best_score",
        "median": "median_score",
    }[paired_metric]
    summary["paired_score"] = summary[metric_column]
    return summary


def add_scale_free_metrics(per_instance, reference_method):
    per_instance = per_instance.copy()
    best = per_instance.groupby("instance", dropna=False)["paired_score"].max()
    per_instance["score_ratio_to_instance_best"] = per_instance.apply(
        lambda row: (
            row["paired_score"] / best[row["instance"]]
            if best[row["instance"]] > 0 else pd.NA
        ),
        axis=1,
    )

    reference = per_instance[per_instance["method"] == reference_method][
        ["instance", "paired_score"]
    ].rename(columns={"paired_score": "reference_score"})
    per_instance = per_instance.merge(reference, on="instance", how="left")
    per_instance["score_ratio_to_reference"] = per_instance.apply(
        lambda row: (
            row["paired_score"] / row["reference_score"]
            if pd.notna(row["reference_score"]) and row["reference_score"] > 0
            else pd.NA
        ),
        axis=1,
    )
    per_instance["gap_to_reference_pct"] = per_instance.apply(
        lambda row: (
            (row["reference_score"] - row["paired_score"]) / row["reference_score"] * 100.0
            if pd.notna(row["reference_score"]) and row["reference_score"] > 0
            else pd.NA
        ),
        axis=1,
    )
    return per_instance


def aggregate_methods(per_instance):
    grouped = per_instance.groupby(["dataset", "method"], dropna=False)
    by_dataset = grouped.agg(
        instances=("instance", "count"),
        total_score=("paired_score", "sum"),
        mean_instance_score=("paired_score", "mean"),
        mean_gap_to_bound_pct=("mean_gap_to_bound_pct", "mean"),
        mean_score_ratio_to_instance_best=("score_ratio_to_instance_best", "mean"),
        mean_score_ratio_to_reference=("score_ratio_to_reference", "mean"),
        mean_gap_to_reference_pct=("gap_to_reference_pct", "mean"),
        mean_runs=("runs", "mean"),
        mean_elapsed_s=("mean_elapsed_s", "mean"),
    ).reset_index()

    overall = per_instance.groupby(["method"], dropna=False).agg(
        dataset=("dataset", lambda values: "ALL"),
        instances=("instance", "count"),
        total_score=("paired_score", "sum"),
        mean_instance_score=("paired_score", "mean"),
        mean_gap_to_bound_pct=("mean_gap_to_bound_pct", "mean"),
        mean_score_ratio_to_instance_best=("score_ratio_to_instance_best", "mean"),
        mean_score_ratio_to_reference=("score_ratio_to_reference", "mean"),
        mean_gap_to_reference_pct=("gap_to_reference_pct", "mean"),
        mean_runs=("runs", "mean"),
        mean_elapsed_s=("mean_elapsed_s", "mean"),
    ).reset_index()
    overall = overall[[
        "dataset",
        "method",
        "instances",
        "total_score",
        "mean_instance_score",
        "mean_gap_to_bound_pct",
        "mean_score_ratio_to_instance_best",
        "mean_score_ratio_to_reference",
        "mean_gap_to_reference_pct",
        "mean_runs",
        "mean_elapsed_s",
    ]]
    return pd.concat([by_dataset, overall], ignore_index=True)


def adjusted_p_values(p_values, method):
    indexed = [
        (idx, float(p))
        for idx, p in enumerate(p_values)
        if pd.notna(p)
    ]
    adjusted = [pd.NA] * len(p_values)
    if not indexed:
        return adjusted

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)

    if method == "holm":
        running_max = 0.0
        for rank, (idx, p_value) in enumerate(indexed, start=1):
            value = min(1.0, (m - rank + 1) * p_value)
            running_max = max(running_max, value)
            adjusted[idx] = running_max
        return adjusted

    if method == "bh":
        running_min = 1.0
        for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
            value = min(1.0, p_value * m / rank)
            running_min = min(running_min, value)
            adjusted[idx] = running_min
        return adjusted

    raise ValueError(f"Unknown p-value adjustment method: {method}")


def add_p_value_corrections(tests):
    if tests.empty:
        return tests
    tests = tests.copy()
    tests["p_value_holm"] = pd.NA
    tests["p_value_bh"] = pd.NA
    group_cols = ["dataset", "reference_method", "alternative"]
    for _key, idx in tests.groupby(group_cols, dropna=False).groups.items():
        positions = list(idx)
        p_values = tests.loc[positions, "p_value"].tolist()
        tests.loc[positions, "p_value_holm"] = adjusted_p_values(p_values, "holm")
        tests.loc[positions, "p_value_bh"] = adjusted_p_values(p_values, "bh")
    tests["p_value_corrected"] = tests["p_value_holm"]
    tests["p_correction_primary"] = "holm"
    return tests


def wilcoxon_table(per_instance, reference_method, alternative):
    methods = sorted(method for method in per_instance["method"].unique()
                     if method != reference_method)
    rows = []
    datasets = sorted(per_instance["dataset"].unique()) + ["ALL"]

    for dataset in datasets:
        if dataset == "ALL":
            subset = per_instance
        else:
            subset = per_instance[per_instance["dataset"] == dataset]
        ref = subset[subset["method"] == reference_method][
            ["instance", "paired_score"]
        ].rename(columns={"paired_score": "reference_score"})
        if ref.empty:
            continue
        for method in methods:
            cmp = subset[subset["method"] == method][
                ["instance", "paired_score"]
            ].rename(columns={"paired_score": "comparison_score"})
            merged = ref.merge(cmp, on="instance", how="inner")
            if len(merged) < 2:
                continue
            diff = merged["reference_score"] - merged["comparison_score"]
            non_zero = diff[diff != 0]
            if len(non_zero) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = wilcoxon(
                    merged["reference_score"],
                    merged["comparison_score"],
                    alternative=alternative,
                    zero_method="wilcox",
                )
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            cmp_mean = merged["comparison_score"].mean()
            ref_mean = merged["reference_score"].mean()
            improvement_pct = (
                (ref_mean - cmp_mean) / cmp_mean * 100.0
                if cmp_mean > 0 else 0.0
            )
            rows.append({
                "dataset": dataset,
                "reference_method": reference_method,
                "comparison_method": method,
                "paired_instances": len(merged),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "reference_mean": ref_mean,
                "comparison_mean": cmp_mean,
                "mean_difference": diff.mean(),
                "mean_improvement_pct": improvement_pct,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "alternative": alternative,
            })
    return pd.DataFrame(rows)


def friedman_table(per_instance):
    rows = []
    datasets = sorted(per_instance["dataset"].unique()) + ["ALL"]

    for dataset in datasets:
        subset = (
            per_instance
            if dataset == "ALL"
            else per_instance[per_instance["dataset"] == dataset]
        )
        pivot = subset.pivot_table(
            index="instance",
            columns="method",
            values="paired_score",
            aggfunc="mean",
        ).dropna(axis=0, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 3:
            continue

        statistic, p_value = friedmanchisquare(
            *[pivot[column].to_numpy() for column in pivot.columns]
        )
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        avg_ranks = ranks.mean(axis=0).sort_values()
        rows.append({
            "dataset": dataset,
            "paired_instances": pivot.shape[0],
            "method_count": pivot.shape[1],
            "methods": ";".join(map(str, pivot.columns)),
            "friedman_statistic": statistic,
            "p_value": p_value,
            "average_ranks": ";".join(
                f"{method}:{rank:.6f}"
                for method, rank in avg_ranks.items()
            ),
        })
    return pd.DataFrame(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create paper-ready summaries and Wilcoxon comparisons."
    )
    parser.add_argument("--results-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument(
        "--reference-method",
        default="ILS_iRace:full",
        help="Method label used as the reference in paired tests.",
    )
    parser.add_argument(
        "--paired-metric",
        choices=["mean", "best", "median"],
        default="mean",
        help="Aggregate multi-seed runs per instance before paired tests.",
    )
    parser.add_argument(
        "--wilcoxon-alternative",
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="'greater' tests whether reference scores are larger.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.results_csv)
    per_instance = aggregate_per_instance(data, args.paired_metric)
    per_instance = add_scale_free_metrics(per_instance, args.reference_method)
    method_summary = aggregate_methods(per_instance)
    tests = wilcoxon_table(
        per_instance,
        args.reference_method,
        args.wilcoxon_alternative,
    )
    tests = add_p_value_corrections(tests)
    friedman = friedman_table(per_instance)

    per_instance.to_csv(output_dir / "per_instance_summary.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    tests.to_csv(output_dir / "wilcoxon_tests.csv", index=False)
    friedman.to_csv(output_dir / "friedman_tests.csv", index=False)

    print(f"Wrote {output_dir / 'per_instance_summary.csv'}")
    print(f"Wrote {output_dir / 'method_summary.csv'}")
    print(f"Wrote {output_dir / 'wilcoxon_tests.csv'}")
    print(f"Wrote {output_dir / 'friedman_tests.csv'}")


if __name__ == "__main__":
    main()
