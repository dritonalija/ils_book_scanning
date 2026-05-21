#!/usr/bin/env python3
"""Parallel, resumable experiment runner for the Book Scanning ILS solver."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app import validated_output_score
from models import Parser, Solver
from models.solver import VALID_VARIANTS
from models.tweaks import Tweaks
from parameter_sets import DEFAULT_PARAMETER_SET_NAME, PARAMETER_SETS, get_parameter_set


RESULT_FIELDS = [
    "status",
    "algorithm",
    "run_label",
    "variant",
    "component",
    "omitted_operator",
    "instance",
    "dataset",
    "seed",
    "param_set",
    "total_budget_s",
    "time_limit_s",
    "init_max_time_s",
    "initial_score",
    "final_score",
    "upper_bound",
    "gap_to_bound_pct",
    "improvement_pct",
    "elapsed_s",
    "output_path",
    "log_csv",
    "operator_stats_csv",
    "git_commit",
    "git_dirty",
    "python_version",
    "platform",
    "error",
]


SKIP_KEY_FIELDS = [
    "algorithm",
    "run_label",
    "variant",
    "component",
    "omitted_operator",
    "instance",
    "seed",
    "param_set",
    "total_budget_s",
    "time_limit_s",
    "init_max_time_s",
]


def compute_improvement(initial, final):
    return ((final - initial) / initial * 100.0) if initial > 0 else 0.0


def compute_gap_to_bound(score, upper_bound):
    return ((upper_bound - score) / upper_bound * 100.0) if upper_bound > 0 else 0.0


def normalized_path(path):
    return Path(path).as_posix()


def dataset_name(path):
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "input":
        return parts[1]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def result_key(row):
    values = []
    for field in SKIP_KEY_FIELDS:
        value = row.get(field, "")
        if field in {"total_budget_s", "time_limit_s", "init_max_time_s"}:
            try:
                value = f"{float(value):.6f}"
            except (TypeError, ValueError):
                value = ""
        elif field == "seed":
            try:
                value = str(int(value))
            except (TypeError, ValueError):
                value = str(value)
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, str(dirty).lower()
    except Exception:
        return "unknown", "unknown"


def evenly_spaced_seeds(start, end, count):
    if count <= 0:
        raise ValueError("runs-per-instance must be positive")
    if start > end:
        raise ValueError("seed range start must be <= end")
    if count == 1:
        return [start]
    if (end - start + 1) < count:
        return list(range(start, start + count))
    raw = [
        int(round(start + i * (end - start) / (count - 1)))
        for i in range(count)
    ]
    seeds = []
    used = set()
    for seed in raw:
        seed = min(end, max(start, seed))
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    candidate = start
    while len(seeds) < count:
        if candidate not in used:
            seeds.append(candidate)
            used.add(candidate)
        candidate += 1
    return seeds[:count]


def parse_seed_list(args):
    if args.seeds:
        return list(dict.fromkeys(args.seeds))
    return evenly_spaced_seeds(
        args.seed_range[0],
        args.seed_range[1],
        args.runs_per_instance,
    )


def read_instance_file(path):
    instances = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            instances.append(item)
    return instances


def collect_instances(args):
    instances = []
    for file_path in args.instances_file:
        instances.extend(read_instance_file(file_path))
    for input_dir in args.input_dir:
        root = Path(input_dir)
        instances.extend(
            normalized_path(path)
            for path in sorted(root.rglob("*.txt"))
        )

    seen = set()
    unique = []
    for instance in instances:
        norm = normalized_path(instance)
        if norm in seen:
            continue
        if not Path(norm).exists():
            raise FileNotFoundError(f"Instance does not exist: {norm}")
        unique.append(norm)
        seen.add(norm)

    if args.limit_instances is not None:
        unique = unique[: args.limit_instances]
    return unique


def load_completed(results_csv):
    if not Path(results_csv).exists():
        return set()
    completed = set()
    with open(results_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                completed.add(result_key(row))
    return completed


def safe_output_relpath(instance):
    path = Path(instance)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            parts = [
                part.replace(":", "")
                for part in path.parts
                if part not in {path.anchor, os.sep}
            ]
            return Path("__absolute__") / Path(*parts)
    return Path(*path.parts)


def should_log_convergence(instance, seed, convergence_instances, convergence_seeds):
    if not convergence_instances:
        return False
    if normalized_path(instance) not in convergence_instances:
        return False
    return not convergence_seeds or seed in convergence_seeds


def build_jobs(args, instances, seeds, git_commit, git_dirty):
    convergence_instances = {
        normalized_path(path)
        for path in (
            read_instance_file(args.convergence_instances_file)
            if args.convergence_instances_file else []
        )
    }
    convergence_seeds = set(args.convergence_seeds or [])
    all_operators = Tweaks.operator_labels()
    jobs = []

    variants = list(args.variants)
    if args.component_variants:
        for variant in [
            "ls_only",
            "no_perturb",
            "no_restart",
            "no_accept",
            "no_proxy",
        ]:
            if variant not in variants:
                variants.append(variant)

    run_specs = []
    for variant in variants:
        run_specs.append({
            "run_label": variant,
            "variant": variant,
            "component": "full" if variant == "full" else "variant_ablation",
            "omitted_operator": "",
            "operators": None,
            "time_limit_s": args.time_limit,
            "init_max_time_s": args.init_max_time,
            "total_budget_s": args.total_budget,
        })

    if args.include_construction_only:
        run_specs.append({
            "run_label": "construction_only",
            "variant": "full",
            "component": "construction_only",
            "omitted_operator": "",
            "operators": None,
            "time_limit_s": 0.0,
            "init_max_time_s": args.init_max_time,
            "total_budget_s": args.init_max_time,
        })

    omitted_operators = []
    if args.operator_ablations:
        if "all" in args.operator_ablations:
            omitted_operators = all_operators
        else:
            omitted_operators = args.operator_ablations
    for omitted in omitted_operators:
        enabled = [label for label in all_operators if label != omitted]
        run_specs.append({
            "run_label": f"drop_{omitted}",
            "variant": "full",
            "component": "operator_ablation",
            "omitted_operator": omitted,
            "operators": enabled,
            "time_limit_s": args.time_limit,
            "init_max_time_s": args.init_max_time,
            "total_budget_s": args.total_budget,
        })

    omitted_groups = []
    if args.operator_group_ablations:
        if "all" in args.operator_group_ablations:
            omitted_groups = sorted(Tweaks.GROUPS)
        else:
            omitted_groups = args.operator_group_ablations
    for group_name in omitted_groups:
        omitted = set(Tweaks.GROUPS[group_name])
        enabled = [label for label in all_operators if label not in omitted]
        run_specs.append({
            "run_label": f"drop_{group_name}_operators",
            "variant": "full",
            "component": "operator_group_ablation",
            "omitted_operator": f"group:{group_name}",
            "operators": enabled,
            "time_limit_s": args.time_limit,
            "init_max_time_s": args.init_max_time,
            "total_budget_s": args.total_budget,
        })

    for spec in run_specs:
        if spec["variant"] not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {spec['variant']}")

    for instance in instances:
        rel_path = safe_output_relpath(instance)
        for seed in seeds:
            for spec in run_specs:
                run_root = (
                    Path(args.output_dir)
                    / args.algorithm
                    / spec["run_label"]
                    / f"seed_{seed:03d}"
                )
                output_path = run_root / rel_path
                log_csv = ""
                if should_log_convergence(
                    instance,
                    seed,
                    convergence_instances,
                    convergence_seeds,
                ):
                    log_csv = str(
                        Path(args.log_dir)
                        / args.algorithm
                        / spec["run_label"]
                        / f"seed_{seed:03d}"
                        / rel_path.with_suffix(".csv")
                    )
                operator_stats_csv = ""
                if args.operator_stats:
                    operator_stats_csv = str(
                        Path(args.operator_stats_dir)
                        / args.algorithm
                        / spec["run_label"]
                        / f"seed_{seed:03d}"
                        / rel_path.with_suffix(".operators.csv")
                    )

                jobs.append({
                    "algorithm": args.algorithm,
                    "run_label": spec["run_label"],
                    "variant": spec["variant"],
                    "component": spec["component"],
                    "omitted_operator": spec["omitted_operator"],
                    "operators": spec["operators"],
                    "instance": normalized_path(instance),
                    "dataset": dataset_name(instance),
                    "seed": seed,
                    "param_set": args.param_set,
                    "time_limit_s": float(spec["time_limit_s"]),
                    "init_max_time_s": float(spec["init_max_time_s"]),
                    "total_budget_s": float(spec["total_budget_s"]),
                    "output_path": str(output_path),
                    "log_csv": log_csv,
                    "operator_stats_csv": operator_stats_csv,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                })
    return jobs


def apply_parameter_set_kwargs(param_set):
    params = get_parameter_set(param_set)
    params.pop("description", None)
    if "alphas" in params:
        params["alpha_values"] = params.pop("alphas")
    return params


def run_job(job):
    started = time.time()
    row = {field: "" for field in RESULT_FIELDS}
    row.update({
        "status": "error",
        "algorithm": job["algorithm"],
        "run_label": job["run_label"],
        "variant": job["variant"],
        "component": job["component"],
        "omitted_operator": job["omitted_operator"],
        "instance": job["instance"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "param_set": job["param_set"],
        "total_budget_s": f"{job['total_budget_s']:.6f}",
        "time_limit_s": f"{job['time_limit_s']:.6f}",
        "init_max_time_s": f"{job['init_max_time_s']:.6f}",
        "output_path": job["output_path"],
        "log_csv": job["log_csv"],
        "operator_stats_csv": job["operator_stats_csv"],
        "git_commit": job["git_commit"],
        "git_dirty": job["git_dirty"],
        "python_version": job["python_version"],
        "platform": job["platform"],
    })

    try:
        data = Parser(job["instance"]).parse()
        params = apply_parameter_set_kwargs(job["param_set"])
        solver_kwargs = {
            "instance_name": os.path.basename(job["instance"]),
            "time_limit": job["time_limit_s"],
            "init_max_time": job["init_max_time_s"],
            "variant": job["variant"],
            "operators": job["operators"],
            "log_csv": job["log_csv"] or None,
            "operator_stats_csv": job["operator_stats_csv"] or None,
        }
        solver_kwargs.update(params)

        solver = Solver(seed=int(job["seed"]), verbose=False)
        result = solver.iterated_local_search(data, **solver_kwargs)

        output_path = Path(job["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.export(str(output_path))
        validator_score = validated_output_score(job["instance"], str(output_path))
        if validator_score != result.fitness_score:
            raise ValueError(
                f"Score mismatch: solver={result.fitness_score}, "
                f"validator={validator_score}"
            )

        upper_bound = data.calculate_upper_bound()
        elapsed = time.time() - started
        row.update({
            "status": "ok",
            "initial_score": result.initial_score,
            "final_score": result.fitness_score,
            "upper_bound": upper_bound,
            "gap_to_bound_pct": f"{compute_gap_to_bound(result.fitness_score, upper_bound):.6f}",
            "improvement_pct": f"{compute_improvement(result.initial_score, result.fitness_score):.6f}",
            "elapsed_s": f"{elapsed:.6f}",
            "error": "",
        })
    except Exception as exc:
        row.update({
            "status": "error",
            "elapsed_s": f"{time.time() - started:.6f}",
            "error": f"{exc}\n{traceback.format_exc(limit=8)}",
        })
    return row


def open_results_writer(results_csv):
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = open(path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run reproducible multi-seed ILS experiments in parallel."
    )
    parser.add_argument(
        "--instances-file",
        action="append",
        default=[],
        help="Text file with one input instance path per line. Can be repeated.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Directory searched recursively for .txt instances. Can be repeated.",
    )
    parser.add_argument("--limit-instances", type=positive_int, default=None)
    parser.add_argument("--algorithm", default="ILS_iRace")
    parser.add_argument(
        "--param-set",
        default=DEFAULT_PARAMETER_SET_NAME,
        choices=sorted(PARAMETER_SETS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full"],
        choices=sorted(VALID_VARIANTS),
    )
    parser.add_argument(
        "--component-variants",
        action="store_true",
        help="Also run core ablations: ls_only, no_perturb, no_restart, no_accept, no_proxy.",
    )
    parser.add_argument(
        "--operator-ablations",
        nargs="*",
        default=[],
        help="Run full variant while omitting selected operators, or pass 'all'.",
    )
    parser.add_argument(
        "--operator-group-ablations",
        nargs="*",
        default=[],
        help=(
            "Run full variant while omitting operator groups "
            f"({', '.join(sorted(Tweaks.GROUPS))}), or pass 'all'."
        ),
    )
    parser.add_argument("--include-construction-only", action="store_true")
    parser.add_argument(
        "--total-time-limit",
        type=float,
        default=None,
        help="Approximate total budget per run including initialization. If set, ILS time = total - init.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="ILS improvement budget. Ignored when --total-time-limit is set.",
    )
    parser.add_argument("--init-max-time", type=float, default=120.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--seed-range", type=int, nargs=2, default=[1, 100])
    parser.add_argument("--runs-per-instance", type=positive_int, default=10)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--output-dir", default="output/experiments")
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "CSV for raw per-run results. Defaults to "
            "<output-dir>/experiment_results.csv."
        ),
    )
    parser.add_argument("--log-dir", default="logs/convergence")
    parser.add_argument(
        "--convergence-instances-file",
        default=None,
        help="Only these instances receive convergence logs.",
    )
    parser.add_argument(
        "--convergence-seeds",
        type=int,
        nargs="+",
        default=[],
        help="Optional seed filter for convergence logs.",
    )
    parser.add_argument("--operator-stats", action="store_true")
    parser.add_argument("--operator-stats-dir", default="logs/operators")
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args):
    if not args.instances_file and not args.input_dir:
        raise ValueError("Provide at least one --instances-file or --input-dir")
    invalid_operators = [
        item
        for item in args.operator_ablations
        if item != "all" and item not in Tweaks.operator_labels()
    ]
    if invalid_operators:
        raise ValueError(
            "Unknown operator(s) for ablation: " + ", ".join(invalid_operators)
        )
    invalid_groups = [
        item
        for item in args.operator_group_ablations
        if item != "all" and item not in Tweaks.GROUPS
    ]
    if invalid_groups:
        raise ValueError(
            "Unknown operator group(s) for ablation: " + ", ".join(invalid_groups)
        )
    if args.total_time_limit is not None:
        if args.total_time_limit <= 0:
            raise ValueError("--total-time-limit must be positive")
        if args.init_max_time >= args.total_time_limit:
            raise ValueError("--init-max-time must be smaller than --total-time-limit")
        args.total_budget = args.total_time_limit
        args.time_limit = args.total_time_limit - args.init_max_time
    else:
        args.total_budget = args.time_limit + args.init_max_time
    if args.time_limit < 0:
        raise ValueError("--time-limit cannot be negative")
    if args.results_csv is None:
        args.results_csv = str(Path(args.output_dir) / "experiment_results.csv")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        instances = collect_instances(args)
        seeds = parse_seed_list(args)
        git_commit, git_dirty = git_metadata()
        jobs = build_jobs(args, instances, seeds, git_commit, git_dirty)
        completed = set() if args.force else load_completed(args.results_csv)
        pending = [job for job in jobs if result_key(job) not in completed]
    except Exception as exc:
        parser.error(str(exc))
        return 2

    print(f"Instances: {len(instances)}")
    print(f"Seeds: {seeds}")
    print(f"Total jobs: {len(jobs)} | Pending: {len(pending)} | Skipped: {len(jobs) - len(pending)}")
    print(
        f"Budget: init={args.init_max_time:.0f}s, "
        f"ILS={args.time_limit:.0f}s, total~={args.total_budget:.0f}s"
    )
    print(f"Workers: {args.workers}")
    if args.dry_run:
        return 0
    if not pending:
        return 0

    handle, writer = open_results_writer(args.results_csv)
    ok = 0
    failed = 0
    try:
        if args.workers == 1:
            for idx, job in enumerate(pending, start=1):
                row = run_job(job)
                writer.writerow(row)
                handle.flush()
                ok += row["status"] == "ok"
                failed += row["status"] != "ok"
                print(
                    f"[{idx}/{len(pending)}] {row['status']} "
                    f"{row['run_label']} seed={row['seed']} "
                    f"{row['instance']} score={row['final_score']}"
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_job, job) for job in pending]
                for idx, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    writer.writerow(row)
                    handle.flush()
                    ok += row["status"] == "ok"
                    failed += row["status"] != "ok"
                    print(
                        f"[{idx}/{len(pending)}] {row['status']} "
                        f"{row['run_label']} seed={row['seed']} "
                        f"{row['instance']} score={row['final_score']}"
                    )
    finally:
        handle.close()

    print(f"Done. ok={ok}, failed={failed}, results={args.results_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
