#!/usr/bin/env python3
"""
iRace target runner adapter.

Called by iRace via target-runner.sh with positional arguments:
  <candidate_id> <instance_id> <seed> <instance_path> [boundMax] --param1 val1 ...

By default, each candidate receives a tuning budget of 300 seconds for ILS and
up to 120 seconds for initial construction. Final evaluation uses 600 seconds
for ILS; the tuned elite should be transfer-checked at that budget before final
reporting. Override with IRACE_TIME_LIMIT and IRACE_INIT_BUDGET_CAP for pilot
or exact-budget tuning runs.

Prints a single negative score to stdout (iRace minimises).
All solver output is suppressed.
"""

from contextlib import redirect_stdout
import os
import sys

from parameter_sets import ALPHA_POOLS

DEFAULT_IRACE_TIME_LIMIT = float(os.environ.get('IRACE_TIME_LIMIT', '300.0'))
DEFAULT_INIT_BUDGET_RATIO = 0.30
DEFAULT_INIT_BUDGET_CAP = float(os.environ.get('IRACE_INIT_BUDGET_CAP', '120.0'))
DEFAULT_RESTART_INIT_BUDGET_RATIO = 0.30

FLOAT_PARAMS = {
    'accept_worse_prob',
    'sa_final_temperature_ratio',
    'grasp_rcl',
    'perturb_replace_bias',
    'restart_fresh_probability',
    'ls_order_weight',
    'ls_insert_weight',
    'ls_strategic_weight',
    'restart_init_budget_ratio',
}
INT_PARAMS = {
    'restart_threshold',
    'perturb_strength_base',
    'perturb_strength_growth',
    'local_no_improve_limit',
}
CAT_PARAMS = {'alpha_pool'}
KNOWN_PARAMS = FLOAT_PARAMS | INT_PARAMS | CAT_PARAMS


def parse_irace_arguments(argv):
    if len(argv) < 5:
        raise ValueError(
            "Usage: irace_runner.py <id> <inst_id> <seed> <instance> "
            "[boundMax] --param1 val1 ..."
        )

    seed = int(argv[3])
    instance_path = argv[4]

    if len(argv) > 5 and not argv[5].startswith('--'):
        time_limit = float(argv[5])
        remaining_args = argv[6:]
    else:
        time_limit = DEFAULT_IRACE_TIME_LIMIT
        remaining_args = argv[5:]

    params = {}
    i = 0
    while i < len(remaining_args):
        key = remaining_args[i].lstrip('-').replace('-', '_')
        if i + 1 < len(remaining_args):
            params[key] = remaining_args[i + 1]
        i += 2

    return seed, instance_path, time_limit, params


def build_solver_kwargs(instance_path, time_limit, params):
    kwargs = {
        'instance_name': os.path.basename(instance_path),
        'time_limit': time_limit,
        'init_max_time': min(
            time_limit * DEFAULT_INIT_BUDGET_RATIO,
            DEFAULT_INIT_BUDGET_CAP,
        ),
        'restart_init_budget_ratio': DEFAULT_RESTART_INIT_BUDGET_RATIO,
    }

    unknown = sorted(set(params) - KNOWN_PARAMS)
    if unknown:
        raise ValueError(f"Unknown iRace parameter(s): {', '.join(unknown)}")

    for key, value in params.items():
        if key in FLOAT_PARAMS:
            kwargs[key] = float(value)
        elif key in INT_PARAMS:
            kwargs[key] = int(value)
        elif key == 'alpha_pool':
            pool_name = value.strip("\"'")
            if pool_name not in ALPHA_POOLS:
                raise ValueError(f"Unknown alpha_pool '{value}'")
            kwargs['alpha_values'] = ALPHA_POOLS[pool_name]

    return kwargs


def main():
    try:
        seed, instance_path, time_limit, params = parse_irace_arguments(sys.argv)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        from models import Parser, Solver

        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
            parser = Parser(instance_path)
            data = parser.parse()
            solver = Solver(seed=seed, verbose=False)
            kwargs = build_solver_kwargs(instance_path, time_limit, params)
            result = solver.iterated_local_search(data, **kwargs)
            score = result.fitness_score
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(-score)


if __name__ == "__main__":
    main()
