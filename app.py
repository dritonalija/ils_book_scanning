"""Command-line interface for the Book Scanning ILS solver."""

import argparse
import csv
import os
import sys
import time

from models import Parser
from models import Solver
from models.tweaks import Tweaks
from models.solver import VALID_VARIANTS
from parameter_sets import (
    ALPHA_POOLS,
    DEFAULT_PARAMETER_SET_NAME,
    PARAMETER_SETS,
    get_parameter_set,
)
from validator.validator import (
    read_input_file,
    read_output_file,
    validate_solution,
)


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


def default_summary_csv_path(input_dir, output_dir):
    input_name = os.path.basename(os.path.normpath(input_dir)) or "batch"
    return os.path.join(output_dir, f"batch_summary_{input_name}.csv")


def collect_explicit_dests(parser, argv):
    explicit_dests = set()
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_dests.add(dest)

    return explicit_dests


def apply_parameter_set(args, explicit_dests):
    parameter_set = get_parameter_set(args.param_set)
    parameter_set.pop("description", None)

    for dest, value in parameter_set.items():
        if dest in explicit_dests:
            continue
        setattr(args, dest, value)

    return args


def validated_output_score(input_path, output_path):
    verdict = validate_solution(input_path, output_path, isConsoleApplication=True)
    if verdict != "Valid":
        details = validate_solution(input_path, output_path)
        raise ValueError(
            f"Validator rejected {os.path.basename(output_path)}:\n{details}"
        )

    _books, _libs, _days, book_scores, _libraries = read_input_file(input_path)
    _num_libraries, solution = read_output_file(output_path)
    scanned_books = set()
    total_score = 0
    for _lib_id, _num_books, books in solution:
        unique_books = [book for book in books if book not in scanned_books]
        scanned_books.update(unique_books)
        total_score += sum(
            book_scores[book]
            for book in unique_books
            if 0 <= book < len(book_scores)
        )
    return total_score


def run_instance(args, input_path, output_path):
    parser = Parser(input_path)
    data = parser.parse()
    solver = Solver(seed=args.seed, verbose=not args.quiet)
    alpha_values = args.alphas
    instance_name = os.path.basename(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    operator_stats_csv = None
    if args.operator_stats_csv:
        operator_stats_csv = args.operator_stats_csv
    elif args.operator_stats_dir:
        stats_stem = os.path.splitext(instance_name)[0]
        operator_stats_csv = os.path.join(
            args.operator_stats_dir,
            f"{stats_stem}.operators.csv",
        )
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
        pool_size=args.pool_size,
        init_max_time=args.init_max_time,
        init_budget_ratio=args.init_budget_ratio,
        restart_init_budget_ratio=args.restart_init_budget_ratio,
        restart_threshold=args.restart_threshold,
        perturb_strength_base=args.perturb_strength_base,
        perturb_strength_growth=args.perturb_strength_growth,
        accept_worse_prob=args.accept_worse_prob,
        sa_final_temperature_ratio=args.sa_final_temperature_ratio,
        alpha_values=alpha_values,
        weighted_beta=args.weighted_beta,
        grasp_rcl=args.grasp_rcl,
        grasp_max_time=args.grasp_max_time,
        local_no_improve_limit=args.local_no_improve_limit,
        ls_order_weight=args.ls_order_weight,
        ls_insert_weight=args.ls_insert_weight,
        ls_strategic_weight=args.ls_strategic_weight,
        operator_weights=operator_weights or None,
        operators=args.operators,
        enable_initial_local_search=args.enable_initial_local_search,
        perturb_replace_bias=args.perturb_replace_bias,
        restart_fresh_probability=args.restart_fresh_probability,
        variant=args.variant,
        log_csv=log_csv,
        operator_stats_csv=operator_stats_csv,
    )
    result.export(output_path)
    validator_score = validated_output_score(input_path, output_path)
    if validator_score != result.fitness_score:
        raise ValueError(
            f"Score mismatch for {instance_name}: "
            f"solver={result.fitness_score}, validator={validator_score}"
        )
    return result


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Iterated Local Search for Book Scanning")

    parser.add_argument("input", nargs="?", help="Single input instance path")
    parser.add_argument("output", nargs="?", help="Single output solution path")
    parser.add_argument(
        "--param-set",
        default=DEFAULT_PARAMETER_SET_NAME,
        choices=sorted(PARAMETER_SETS),
        help="Named parameter set to load before applying individual CLI overrides",
    )
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
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--restart-threshold", type=int, default=None)
    parser.add_argument("--perturb-strength-base", type=int, default=None)
    parser.add_argument("--perturb-strength-growth", type=int, default=None)
    parser.add_argument("--accept-worse-prob", type=float, default=0.04)
    parser.add_argument(
        "--sa-final-temperature-ratio",
        type=float,
        default=0.05,
        help="Final/initial temperature ratio for geometric SA cooling",
    )
    parser.add_argument("--weighted-beta", type=float, default=0.12)
    parser.add_argument("--grasp-rcl", type=float, default=0.05)
    parser.add_argument("--grasp-max-time", type=float, default=5.0)
    parser.add_argument("--local-no-improve-limit", type=int, default=None)
    parser.add_argument("--ls-order-weight", type=float, default=1.0)
    parser.add_argument("--ls-insert-weight", type=float, default=1.0)
    parser.add_argument("--ls-strategic-weight", type=float, default=1.0)
    parser.add_argument(
        "--enable-initial-local-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the pre-ILS local-search pass on the initial solution",
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
                        default=ALPHA_POOLS["default"])
    parser.add_argument("--variant", type=str, default="full",
                        choices=sorted(VALID_VARIANTS),
                        help="Algorithm variant for ablation study")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Directory for convergence CSV logs")
    parser.add_argument(
        "--operator-stats-csv",
        type=str,
        default=None,
        help="Single-instance path for local-search operator statistics CSV",
    )
    parser.add_argument(
        "--operator-stats-dir",
        type=str,
        default=None,
        help="Directory for per-instance local-search operator statistics CSVs",
    )
    parser.add_argument("--summary-csv", type=str, default=None,
                        help=(
                            "Path to batch summary CSV updated after each instance; "
                            "defaults to <output-dir>/batch_summary_<input-folder>.csv"
                        ))
    for label in Tweaks.operator_labels():
        parser.add_argument(
            f"--w-{label.replace('_', '-')}",
            dest=f"w_{label}",
            type=float,
            default=None,
            help=f"Override weight for operator {label}",
        )

    return parser


def parse_arguments(parser, argv=None):
    if argv is None:
        argv = sys.argv[1:]
    explicit_dests = collect_explicit_dests(parser, argv)
    args = parser.parse_args(argv)
    return apply_parameter_set(args, explicit_dests)


def main():
    parser = build_argument_parser()
    args = parse_arguments(parser)
    if args.operator_stats_csv and not args.input:
        parser.error("--operator-stats-csv can only be used with a single input instance")
    if args.operator_stats_csv and args.operator_stats_dir:
        parser.error("--operator-stats-csv and --operator-stats-dir are mutually exclusive")

    single_input_path = None
    single_output_path = None
    if args.input:
        output_path = args.output or os.path.join(
            args.output_dir, os.path.basename(args.input))
        single_input_path = args.input
        single_output_path = output_path
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
        summary_csv = args.summary_csv or default_summary_csv_path(
            args.input_dir,
            args.output_dir,
        )
        os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
        summary_file = open(summary_csv, "w", newline="")
        summary_writer = csv.writer(summary_file)
        write_summary_header(summary_writer)
        summary_file.flush()
        print("---------- ITERATED LOCAL SEARCH WITH MULTI-START STAGNATION RESTARTS ----------")
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
                print(f"Updated summary CSV: {summary_csv}")
                print("----------------------")
        finally:
            if summary_file:
                summary_file.close()

        print(f"\n{'=' * 70}")
        print(f"  Summary (variant={args.variant}, param_set={args.param_set})")
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
        if single_input_path:
            verdict = validate_solution(
                single_input_path,
                single_output_path,
                isConsoleApplication=True,
            )
            print(f"\nValidation for {os.path.basename(single_output_path)}: {verdict}")
        else:
            from validator.multiple_validator import validate_all_solutions
            print("\nValidating all solutions...")
            validate_all_solutions(
                input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
