"""Named parameter sets shared by the CLI and iRace tooling."""

from copy import deepcopy


ALPHA_POOLS = {
    "default": [0.5, 1.0, 1.5, 2.0],
    "wide": [0.5, 1.0, 1.5, 2.0, 3.0],
    "dense": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "signup": [0.75, 1.0, 1.5, 2.0, 3.0],
    "explore": [0.4, 0.5, 0.75, 1.0, 1.5, 2.0],
}

PARAMETER_SETS = {
    "default": {
        "description": "Baseline configuration reproducing the earlier manually chosen CLI defaults.",
        "accept_worse_prob": 0.04,
        "sa_final_temperature_ratio": 0.05,
        "grasp_rcl": 0.05,
        "restart_init_budget_ratio": 0.30,
        "perturb_replace_bias": 0.65,
        "restart_fresh_probability": 0.35,
        "ls_order_weight": 1.0,
        "ls_insert_weight": 1.0,
        "ls_strategic_weight": 1.0,
        "enable_initial_local_search": False,
        "alphas": ALPHA_POOLS["default"],
    },
    "irace": {
        "description": (
            "Previous iRace-seeded configuration retained for comparisons; "
            "accept_worse_prob is interpreted by the current SA acceptance rule."
        ),
        "accept_worse_prob": 0.1466,
        "sa_final_temperature_ratio": 0.05,
        "restart_threshold": 7,
        "perturb_strength_base": 4,
        "perturb_strength_growth": 3,
        "grasp_rcl": 0.2318,
        "restart_init_budget_ratio": 0.3648,
        "perturb_replace_bias": 0.3751,
        "restart_fresh_probability": 0.2679,
        "local_no_improve_limit": 449,
        "ls_order_weight": 0.7117,
        "ls_insert_weight": 2.1158,
        "ls_strategic_weight": 0.7780,
        "enable_initial_local_search": False,
        "alphas": ALPHA_POOLS["explore"],
    },
    "irace_final": {
        "description": (
            "Final iRace elite configuration 486 from the 5000-experiment run; "
            "accept_worse_prob is interpreted by the current SA acceptance rule."
        ),
        "accept_worse_prob": 0.0320,
        "sa_final_temperature_ratio": 0.1699,
        "restart_threshold": 3,
        "perturb_strength_base": 2,
        "perturb_strength_growth": 0,
        "grasp_rcl": 0.1308,
        "restart_init_budget_ratio": 0.1491,
        "perturb_replace_bias": 0.6794,
        "restart_fresh_probability": 0.4035,
        "local_no_improve_limit": 316,
        "ls_order_weight": 0.6317,
        "ls_insert_weight": 2.3634,
        "ls_strategic_weight": 1.9964,
        "enable_initial_local_search": False,
        "alphas": ALPHA_POOLS["explore"],
    },
}

DEFAULT_PARAMETER_SET_NAME = "irace_final"


def get_parameter_set(name):
    """Return a defensive copy so callers can safely mutate list values."""
    return deepcopy(PARAMETER_SETS[name])
