"""Command-line interface for the Book Scanning ILS solver."""

import argparse
import csv
import os
import random
import time

from models import Parser
from models import Solver
from models.tweaks import Tweaks
from models.solver import VALID_VARIANTS


DEFAULT_ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]


def compute_improvement(initial, final):
    return ((final - initial) / initial * 100) if initial > 0 else 0.0


def write_summary_header(csv_writer):
    csv_writer.writerow([
        "instance",
        "initial_score",
        "final_score",
        "elapsed_s",
        "running_elapsed_s",
        "improvement_pct",
        "running_total_initial",
        "running_total_final",
        "running_total_improvement_pct",
    ])


def append_summary_row(csv_writer, file_name, elapsed_s, running_elapsed_s,
                       initial, final, total_initial, total_final):
    csv_writer.writerow([
        file_name,
        initial,
        final,
        f"{elapsed_s:.6f}",
        f"{running_elapsed_s:.6f}",
        f"{compute_improvement(initial, final):.6f}",
        total_initial,
        total_final,
        f"{compute_improvement(total_initial, total_final):.6f}",
    ])


def run_instance(args, input_path, output_path):
    parser = Parser(input_path)
    data = parser.parse()
    solver = Solver(seed=args.seed, verbose=not args.quiet)
    alpha_values = args.alphas
    instance_name = os.path.basename(input_path)
    operator_weights = {
        label: getattr(args, f"w_{label}")
        for label in Tweaks.operator_labels()
        if getattr(args, f"w_{label}") is not None
    }

    log_csv = None
    if args.log_csv:
        log_stem = os.path.splitext(instance_name)[0]
        log_csv = os.path.join(args.log_csv, f"{log_stem}.csv")

    result = solver.iterated_local_search(
        data,
        instance_name=instance_name,
        time_limit=args.time_limit,
        max_iterations=args.max_iterations,
        pool_size=args.pool_size,
        init_max_time=args.init_max_time,
        init_budget_ratio=args.init_budget_ratio,
        restart_init_budget_ratio=args.restart_init_budget_ratio,
        restart_threshold=args.restart_threshold,
        perturb_strength_base=args.perturb_strength_base,
        perturb_strength_growth=args.perturb_strength_growth,
        accept_worse_prob=args.accept_worse_prob,
        alpha_values=alpha_values,
        weighted_beta=args.weighted_beta,
        grasp_rcl=args.grasp_rcl,
        grasp_max_time=args.grasp_max_time,
        noisy_restarts=args.noisy_restarts,
        local_no_improve_limit=args.local_no_improve_limit,
        ls_order_weight=args.ls_order_weight,
        ls_insert_weight=args.ls_insert_weight,
        ls_strategic_weight=args.ls_strategic_weight,
        operator_weights=operator_weights or None,
        operators=args.operators,
        enable_initial_local_search=args.enable_initial_local_search,
        enable_direct_intensify=args.enable_direct_intensify,
        perturb_replace_bias=args.perturb_replace_bias,
        restart_fresh_probability=args.restart_fresh_probability,
        variant=args.variant,
        log_csv=log_csv,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.export(output_path)
    return result


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Iterated Local Search for Book Scanning")

    parser.add_argument("input", nargs="?", help="Single input instance path")
    parser.add_argument("output", nargs="?", help="Single output solution path")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="ILS search time budget in seconds")
    parser.add_argument("--init-max-time", type=float, default=120.0,
                        help="Maximum initial construction time in seconds")
    parser.add_argument("--init-budget-ratio", type=float, default=None,
                        help="Optional cap for initial construction as a fraction of ILS time")
    parser.add_argument("--restart-init-budget-ratio", type=float, default=0.30,
                        help="Fraction of remaining ILS time allowed for fresh restart construction")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--restart-threshold", type=int, default=None)
    parser.add_argument("--perturb-strength-base", type=int, default=None)
    parser.add_argument("--perturb-strength-growth", type=int, default=None)
    parser.add_argument("--accept-worse-prob", type=float, default=0.04)
    parser.add_argument("--weighted-beta", type=float, default=0.12)
    parser.add_argument("--grasp-rcl", type=float, default=0.05)
    parser.add_argument("--grasp-max-time", type=float, default=5.0)
    parser.add_argument("--noisy-restarts", type=int, default=None)
    parser.add_argument("--local-no-improve-limit", type=int, default=None)
    parser.add_argument("--ls-order-weight", type=float, default=1.0)
    parser.add_argument("--ls-insert-weight", type=float, default=1.0)
    parser.add_argument("--ls-strategic-weight", type=float, default=1.0)
    parser.add_argument(
        "--enable-initial-local-search",
        action="store_true",
        help="Run the pre-ILS local-search pass on the initial solution",
    )
    parser.add_argument(
        "--enable-direct-intensify",
        action="store_true",
        help="Enable the optional pre-ILS direct intensification phase",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=Tweaks.operator_labels(),
        default=None,
        help="Restrict local search to this operator subset",
    )
    parser.add_argument("--perturb-replace-bias", type=float, default=0.65)
    parser.add_argument("--restart-fresh-probability", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--validate", action="store_true",
                        help="Validate outputs after generation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--variant", type=str, default="full",
                        choices=sorted(VALID_VARIANTS),
                        help="Algorithm variant for ablation study")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Directory for convergence CSV logs")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Path to batch summary CSV updated after each instance")
    for label in Tweaks.operator_labels():
        parser.add_argument(
            f"--w-{label.replace('_', '-')}",
            dest=f"w_{label}",
            type=float,
            default=None,
            help=f"Override weight for operator {label}",
        )

    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    if args.input:
        output_path = args.output or os.path.join(
            args.output_dir, os.path.basename(args.input))
        result = run_instance(args, args.input, output_path)
        print(f"Final score for {os.path.basename(args.input)}: "
              f"{result.fitness_score:,}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        results = []
        total_initial = 0
        total_final = 0
        running_elapsed = 0.0
        summary_file = None
        summary_writer = None
        if args.summary_csv:
            os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
            summary_file = open(args.summary_csv, "w", newline="")
            summary_writer = csv.writer(summary_file)
            write_summary_header(summary_writer)
            summary_file.flush()
        print("---------- ITERATED LOCAL SEARCH WITH RANDOM RESTARTS ----------")
        try:
            for file in sorted(os.listdir(args.input_dir)):
                if not file.endswith(".txt"):
                    continue
                input_path = os.path.join(args.input_dir, file)
                output_path = os.path.join(args.output_dir, file)
                instance_start = time.time()
                result = run_instance(args, input_path, output_path)
                elapsed_s = time.time() - instance_start
                results.append((file, result.initial_score, result.fitness_score))
                total_initial += result.initial_score
                total_final += result.fitness_score
                running_elapsed += elapsed_s
                if summary_writer:
                    append_summary_row(
                        summary_writer,
                        file,
                        elapsed_s,
                        running_elapsed,
                        result.initial_score,
                        result.fitness_score,
                        total_initial,
                        total_final,
                    )
                    summary_file.flush()
                print(f"Final score for {file}: {result.fitness_score:,}")
                if args.summary_csv:
                    print(f"Updated summary CSV: {args.summary_csv}")
                print("----------------------")
        finally:
            if summary_file:
                summary_file.close()

        print(f"\n{'=' * 70}")
        print(f"  Summary (variant={args.variant})")
        print(f"{'=' * 70}")
        print(f"{'Instance':<35} {'Initial':>12} {'Final':>12} {'Improv%':>10}")
        print(f"{'-' * 70}")
        for file, initial, final in results:
            improvement = compute_improvement(initial, final)
            print(f"{file:<35} {initial:>12,} {final:>12,} "
                  f"{improvement:>+9.2f}%")
        print(f"{'-' * 70}")
        total_imp = compute_improvement(total_initial, total_final)
        print(f"{'TOTAL':<35} {total_initial:>12,} {total_final:>12,} "
              f"{total_imp:>+9.2f}%")
        print(f"{'=' * 70}")

    if args.validate:
        from validator.multiple_validator import validate_all_solutions
        print("\nValidating all solutions...")
        validate_all_solutions(
            input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
