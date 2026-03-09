import argparse
import os
import random

from models import Parser
from models import Solver


def run_instance(args, input_path, output_path):
    parser = Parser(input_path)
    data = parser.parse()
    solver = Solver(seed=args.seed, verbose=not args.quiet)
    alpha_values = args.alphas

    log_csv = None
    if args.log_csv:
        instance_name = os.path.splitext(os.path.basename(input_path))[0]
        log_csv = os.path.join(args.log_csv, f"{instance_name}.csv")

    result = solver.iterated_local_search(
        data,
        time_limit=args.time_limit,
        max_iterations=args.max_iterations,
        pool_size=args.pool_size,
        init_max_time=args.init_max_time,
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
        perturb_replace_bias=args.perturb_replace_bias,
        restart_fresh_probability=args.restart_fresh_probability,
        variant=args.variant,
        log_csv=log_csv,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.export(output_path)
    return result


def main():
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
    parser.add_argument("--perturb-replace-bias", type=float, default=0.65)
    parser.add_argument("--restart-fresh-probability", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--validate", action="store_true",
                        help="Validate outputs after generation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--variant", type=str, default="full",
                        choices=list(Solver.__module__ and [
                            'full', 'no_perturb', 'no_restart',
                            'no_accept', 'random_walk', 'ls_only']),
                        help="Algorithm variant for ablation study")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Directory for convergence CSV logs")
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
        print("---------- ITERATED LOCAL SEARCH WITH RANDOM RESTARTS ----------")
        for file in sorted(os.listdir(args.input_dir)):
            if not file.endswith(".txt"):
                continue
            input_path = os.path.join(args.input_dir, file)
            output_path = os.path.join(args.output_dir, file)
            result = run_instance(args, input_path, output_path)
            results.append((file, result.initial_score, result.fitness_score))
            print(f"Final score for {file}: {result.fitness_score:,}")
            print("----------------------")

        print(f"\n{'=' * 70}")
        print(f"  Summary (variant={args.variant})")
        print(f"{'=' * 70}")
        print(f"{'Instance':<35} {'Initial':>12} {'Final':>12} {'Improv%':>10}")
        print(f"{'-' * 70}")
        total_initial = 0
        total_final = 0
        for file, initial, final in results:
            improvement = ((final - initial) / initial * 100
                           if initial > 0 else 0)
            print(f"{file:<35} {initial:>12,} {final:>12,} "
                  f"{improvement:>+9.2f}%")
            total_initial += initial
            total_final += final
        print(f"{'-' * 70}")
        total_imp = ((total_final - total_initial) / total_initial * 100
                     if total_initial > 0 else 0)
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
