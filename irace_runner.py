#!/usr/bin/env python3
"""
iRace target runner adapter.

Called by iRace via target-runner.sh with positional arguments:
  <candidate_id> <instance_id> <seed> <instance_path> [boundMax] --param1 val1 ...

Prints a single negative score to stdout (iRace minimises).
All solver output is suppressed.
"""

import os
import sys


def main():
    if len(sys.argv) < 5:
        print("Usage: irace_runner.py <id> <inst_id> <seed> <instance> "
              "[boundMax] --param1 val1 ...")
        sys.exit(1)

    candidate_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = int(sys.argv[3])
    instance_path = sys.argv[4]

    if len(sys.argv) > 5 and not sys.argv[5].startswith('--'):
        time_limit = float(sys.argv[5])
        remaining_args = sys.argv[6:]
    else:
        time_limit = 60.0
        remaining_args = sys.argv[5:]

    params = {}
    i = 0
    while i < len(remaining_args):
        key = remaining_args[i].lstrip('-').replace('-', '_')
        if i + 1 < len(remaining_args):
            params[key] = remaining_args[i + 1]
        i += 2

    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    try:
        from models import Parser, Solver
        from models.tweaks import Tweaks

        parser = Parser(instance_path)
        data = parser.parse()
        solver = Solver(seed=seed, verbose=False)

        kwargs = {
            'instance_name': os.path.basename(instance_path),
            'time_limit': time_limit,
            'init_max_time': min(time_limit * 0.3, 30.0),
        }

        float_params = {
            'accept_worse_prob', 'grasp_rcl', 'grasp_max_time',
            'perturb_replace_bias', 'restart_fresh_probability',
            'ls_order_weight', 'ls_insert_weight', 'ls_strategic_weight',
        }
        int_params = {
            'restart_threshold', 'perturb_strength_base',
            'perturb_strength_growth', 'local_no_improve_limit',
        }
        alpha_pool_map = {
            'default': [0.5, 1.0, 1.5, 2.0],
            'wide': [0.5, 1.0, 1.5, 2.0, 3.0],
            'dense': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            'signup': [0.75, 1.0, 1.5, 2.0, 3.0],
            'explore': [0.4, 0.5, 0.75, 1.0, 1.5, 2.0],
        }
        operator_weight_keys = {f'w_{label}' for label in Tweaks.operator_labels()}
        operator_weights = {}

        for key, val in params.items():
            if key in float_params:
                kwargs[key] = float(val)
            elif key in int_params:
                kwargs[key] = int(val)
            elif key == 'alpha_pool':
                kwargs['alpha_values'] = alpha_pool_map[val]
            elif key == 'operators':
                kwargs['operators'] = [item for item in val.split(',') if item]
            elif key in operator_weight_keys:
                operator_weights[key[2:]] = float(val)

        if operator_weights:
            kwargs['operator_weights'] = operator_weights

        result = solver.iterated_local_search(data, **kwargs)
        score = result.fitness_score
    except Exception as e:
        sys.stdout = real_stdout
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.stdout = real_stdout
    print(-score)


if __name__ == "__main__":
    main()
