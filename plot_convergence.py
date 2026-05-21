#!/usr/bin/env python3
"""Plot convergence curves from solver --log-csv files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_instance_list(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return {
            line.strip().replace("\\", "/")
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        }


def instance_id_from_log(log_root, path):
    rel = path.relative_to(log_root).as_posix()
    parts = rel.split("/")
    if len(parts) >= 5:
        return "/".join(parts[3:]).replace(".csv", ".txt")
    return path.stem


def collect_logs(log_root, instances):
    paths = sorted(Path(log_root).rglob("*.csv"))
    selected = []
    for path in paths:
        instance = instance_id_from_log(Path(log_root), path)
        if instances is None or instance in instances:
            selected.append((instance, path))
    return selected


def plot_instance(instance, paths, output_dir, max_series, x_axis):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = 0
    for path in paths[:max_series]:
        frame = pd.read_csv(path)
        if x_axis not in frame.columns or "best_score" not in frame.columns:
            continue
        frame = frame[[x_axis, "best_score"]].dropna()
        if frame.empty:
            continue
        label_parts = list(path.parts)
        seed_index = next(
            (idx for idx, part in enumerate(label_parts) if part.startswith("seed_")),
            None,
        )
        seed = label_parts[seed_index] if seed_index is not None else path.stem
        run_label = (
            label_parts[seed_index - 1]
            if seed_index is not None and seed_index > 0
            else ""
        )
        ax.plot(frame[x_axis], frame["best_score"], linewidth=1.4, label=f"{run_label}/{seed}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(instance)
    ax.set_xlabel("ILS round" if x_axis == "round" else "Elapsed time (s)")
    ax.set_ylabel("Best score")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()

    safe_name = instance.replace("/", "__").replace("\\", "__").replace(":", "_")
    output_path = output_dir / f"{safe_name}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Plot convergence behavior.")
    parser.add_argument("--log-root", default="logs/convergence")
    parser.add_argument("--output-dir", default="results/plots/convergence")
    parser.add_argument("--instances-file", default=None)
    parser.add_argument("--limit-instances", type=int, default=10)
    parser.add_argument("--max-series-per-instance", type=int, default=10)
    parser.add_argument(
        "--x-axis",
        choices=["elapsed_s", "round"],
        default="elapsed_s",
        help="Use elapsed time or ILS round on the x-axis.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = read_instance_list(args.instances_file)
    logs = collect_logs(log_root, requested)
    by_instance = {}
    for instance, path in logs:
        by_instance.setdefault(instance, []).append(path)

    written = []
    for instance in sorted(by_instance)[: args.limit_instances]:
        output_path = plot_instance(
            instance,
            by_instance[instance],
            output_dir,
            args.max_series_per_instance,
            args.x_axis,
        )
        if output_path:
            written.append(output_path)
            print(f"Wrote {output_path}")

    if not written:
        print("No convergence plots were written. Check --log-root and --instances-file.")


if __name__ == "__main__":
    main()
