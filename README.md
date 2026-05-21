# Iterated Local Search for Book Scanning

An Iterated Local Search (ILS) solver for the Google Hash Code Book Scanning
problem. The current implementation combines a multi-start construction phase,
exact JIT-accelerated evaluation, randomized local search, adaptive
perturbation, and multi-start restarts after prolonged stagnation.

## Table of Contents

- [Iterated Local Search for Book Scanning](#iterated-local-search-for-book-scanning)
  - [Table of Contents](#table-of-contents)
  - [Datasets](#datasets)
  - [What The Solver Does](#what-the-solver-does)
  - [High-Level Flow](#high-level-flow)
  - [Initial Solution Phase](#initial-solution-phase)
    - [Constructors currently tried](#constructors-currently-tried)
    - [How candidate screening works](#how-candidate-screening-works)
    - [Meaning of the construction methods](#meaning-of-the-construction-methods)
    - [Potential modes](#potential-modes)
    - [Construction parameters](#construction-parameters)
  - [Exact Evaluation And Numba](#exact-evaluation-and-numba)
  - [Local Search](#local-search)
    - [Tweak operators](#tweak-operators)
  - [Batch Runs And Summaries](#batch-runs-and-summaries)
  - [iRace Tuning](#irace-tuning)
    - [What each iRace file does](#what-each-irace-file-does)
    - [Run iRace Locally](#run-irace-locally)
  - [ILS Main Loop](#ils-main-loop)
    - [Acceptance rule](#acceptance-rule)
  - [Perturbation](#perturbation)
    - [Perturbation types](#perturbation-types)
  - [Restart Strategy](#restart-strategy)
  - [Instance Profiles](#instance-profiles)
  - [CLI Usage](#cli-usage)
    - [Single instance](#single-instance)
    - [Batch mode](#batch-mode)
    - [Ablation variants](#ablation-variants)
    - [Convergence logging](#convergence-logging)
  - [Docker Usage](#docker-usage)
  - [CLI Parameters](#cli-parameters)
    - [Core run control](#core-run-control)
    - [Initial-solution parameters](#initial-solution-parameters)
    - [ILS and local-search parameters](#ils-and-local-search-parameters)
    - [Perturbation and restart parameters](#perturbation-and-restart-parameters)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [References](#references)

## Datasets

The `input/` directory currently contains four primary dataset collections:

| Folder | Instances | Source | Purpose |
|---|---:|---|---|
| `google_hashcode/` | 5 | Google Hash Code 2020 Book Scanning problem instances | Official benchmark instances from the original dataset, also included in iRace elite testing |
| `real_world/` | 45 | [Book Scanning Problem Input Generator](https://bookscanning-ig.netlify.app/) | Generated instances intended to reflect real-world scanning scenarios |
| `synthetic/` | 139 | [uran-lajci/Book.Scanning.Dataset](https://github.com/uran-lajci/Book.Scanning.Dataset) | Larger synthetic training and evaluation pool |
| `seed_based/` | 135 | [dritonalija/book_scanning_dataset](https://github.com/dritonalija/book_scanning_dataset) | Seed-controlled generated instances for reproducible experiments |

Two practical notes about the current layout:

- batch runs should target a specific dataset folder such as `input/google_hashcode`
  or `input/seed_based`
- iRace tuning uses `input/synthetic/`, `input/seed_based/`, and
  `input/real_world/`, split into deterministic train/test lists via
  `instances.txt` and `instances-test.txt`
- `google_hashcode/` is appended to `instances-test.txt` so iRace elite testing
  also considers the classic benchmark cases

## What The Solver Does

The run is split into two phases:

1. Initial solution generation, capped by `--init-max-time` with a hard upper
   bound of 120 seconds inside the solver.
2. ILS improvement, controlled by `--time-limit`.

## High-Level Flow

```text
1. Generate several initial library orders
2. Score them quickly with exact JIT evaluation
3. Rebuild only the top few exactly and keep the best one
4. Optionally polish it once with local search
5. Repeat until time runs out:
   - perturb the current home-base solution
   - run local search
   - accept or reject the candidate
   - restart when search stagnates
6. Return the best solution found
```

## Initial Solution Phase

The initializer is implemented in `models/initial_solution.py`.

### Constructors currently tried

The solver screens multiple candidate orders before exact rebuilding:

- `Sorted`
- `Heap Greedy top/raw` for several `alpha` values
- `Heap Greedy cap/raw`
- `Heap Greedy cap/rare`
- `Fill Greedy raw`
- `Fill Greedy rare`
- `Branch Fill Greedy`
- `Static Greedy cap/raw`
- `Static Greedy cap/rare`
- `Noisy Heap cap/raw`
- `Noisy Heap cap/rare`
- `GRASP`

### How candidate screening works

The constructor phase is intentionally two-stage:

1. Build an order with a computationally inexpensive constructor.
2. Score that order with `data.fast_evaluate(order)`.
3. Keep all screened candidates.
4. Rebuild only the top `3` candidates exactly with `Solution.from_order(...)`.
5. Choose the best exact initial solution.

This keeps initialization fast while preserving exact final scoring for the
selected starts.

### Meaning of the construction methods

- `Sorted`: ranks by `(signup_days, -total_book_score)`.
- `Heap Greedy`: uses a priority queue and recomputes library value lazily as
  time and used books change.
- `Fill Greedy`: favors libraries whose remaining useful books can fill their
  available scanning capacity, with optional rarity weighting.
- `Branch Fill Greedy`: uses a stronger fill-ratio greedy with an upper-bound
  heap; this is most useful on dense, overlapping instances.
- `Static Greedy`: evaluates all remaining libraries each step and picks the
  current best.
- `Noisy Heap`: same as heap greedy but with small random noise to diversify
  starts.
- `GRASP`: repeatedly samples from a restricted candidate list for up to
  `--grasp-max-time` seconds.

### Potential modes

Several constructors use precomputed library potentials from
`models/instance_data.py`:

- `top/raw`: top high-score books
- `top/rare`: top books with rarity bonus
- `cap/raw`: books likely scannable within capacity, plain score
- `cap/rare`: books likely scannable within capacity, rarity-aware score

These potentials improve the efficiency of the construction phase without
requiring a full exact rebuild for every candidate.

### Construction parameters

- `alpha`: controls how strongly signup time is penalized. Larger values favor
  faster-signup libraries more aggressively.
- `grasp_rcl`: restricted candidate list ratio for GRASP.
- `grasp_max_time`: dedicated GRASP budget. The solver still exposes it as a
  CLI option with default `5s`; the iRace setup can tune it in the range
  `0s` to `8s`.

## Exact Evaluation And Numba

Evaluation is implemented in `models/evaluation.py`.

In simple terms:

- `Numba` makes the scoring code much faster
- the final solution rebuild still uses the same scoring logic
- `fast_evaluate` is the exact objective used during search; by default it is a
  small portfolio that returns the best score among sequential, global, and
  balanced-global assignment decoders for a fixed library order
- local search first uses a cheap sequential proxy to screen order moves, then
  checks surviving moves with `fast_evaluate`, and only rebuilds the full
  solution for strict exact-objective improvements
- on small/medium library-count instances, a balanced global assignment mode is
  also considered when scoring and rebuilding a fixed order
- the pure-Python rebuild fallback evaluates the same sequential, global, and
  balanced-global assignment modes as the accelerated path
- reusable work buffers avoid repeated allocation of full book/library masks
  during high-frequency scoring calls
- if `numba` is not available, the solver still works, just more slowly

So `Numba` is a speed improvement, not a change to the objective.
The `sequential_only` ablation variant disables the global and balanced-global
assignment modes so the reported score comes only from the official sequential
library-order assignment. The `no_proxy` ablation keeps the same exact
objective but removes the cheap proxy screen, forcing local search to evaluate
candidate order moves directly with `fast_evaluate`.

## Local Search

Local search is implemented in `models/local_search.py`.

It has two stages:

1. Proxy-guided exploration.
   The search samples one tweak operator at a time. Fast order operators build a
   candidate library order, screen it with a cheap sequential proxy, then verify
   non-worse proxy moves with the exact `fast_evaluate` objective. Non-order
   operators that already rebuild a solution are accepted only when the exact
   solution score improves.
2. Exact refinement.
   The remaining local-search time is spent without the proxy. A unified
   refinement pass samples broader order-neighborhood moves and then
   deterministically polishes the most important part of the order.

The exact refinement tries:

- sampled exact order-neighborhood moves
- adjacent swaps in the signed prefix
- short reinsertion moves in the signed prefix
- boundary swaps between signed and unsigned libraries

This keeps the paper story simple: proxy evaluation is used only as a
surrogate filter, while every accepted move is validated by the same exact
decoder used for the exported solution. The final exact tail also reduces the
risk that a useful solution is missed only because the proxy was imperfect.

### Tweak operators

The operators are defined in `models/tweaks.py`.

Current operators:

- `remove_library`
- `move_signed`
- `swap_signed`
- `block_reinsert`
- `swap_signed_with_unsigned`
- `reverse_segment`
- `insert_library`
- `swap_neighbor_libraries`
- `critical_path_insert`
- `diversity_swap`
- `sampled_best_exchange`
- `coverage_exchange`
- `paired_choice_flip`

Their default sampling weights are:

```python
{
    "remove_library": 2.8,
    "move_signed": 2.4,
    "swap_signed": 2.2,
    "block_reinsert": 1.9,
    "swap_signed_with_unsigned": 1.6,
    "reverse_segment": 1.3,
    "insert_library": 1.1,
    "swap_neighbor_libraries": 0.8,
    "critical_path_insert": 0.7,
    "diversity_swap": 0.6,
    "sampled_best_exchange": 0.45,
    "coverage_exchange": 0.2,
    "paired_choice_flip": 0.1,
}
```

The solver can also scale these operators through three grouped multipliers:

- `ls_order_weight`: affects reorder-style operators such as swaps, moves,
  reversals, and block reinsertion
- `ls_insert_weight`: affects operators that exchange signed and unsigned
  libraries or insert/remove libraries
- `ls_strategic_weight`: affects the more targeted operators
  such as `diversity_swap`, `sampled_best_exchange`, `coverage_exchange`,
  and `paired_choice_flip`

For manual experiments, the CLI also exposes per-operator overrides through
`--w-...` flags. The final iRace configuration does not tune these
individually; it tunes only the three grouped multipliers to keep the search
space compact and interpretable.

Two strategic operators are instance-class-conditional. `coverage_exchange`
applies a maximum-coverage exchange only when the input has uniform
score/signup/rate structure. `paired_choice_flip` is enabled only when that
uniform instance also matches a paired binary-choice structure. They are part
of the full solver for uniform structured instances, while the `no_structured`
variant disables them for the general-neighborhood ablation.

## Batch Runs And Summaries

The current codebase supports batch solving directly through `app.py`. Batch
runs can maintain a live summary CSV with one row per instance.

Example:

```powershell
python .\app.py `
  --input-dir .\input\seed_based `
  --output-dir .\output\seed_based `
  --time-limit 120 `
  --summary-csv .\output\seed_based\batch_summary.csv `
  --validate
```

Real-world batch evaluation uses the same pattern:

```powershell
python .\app.py `
  --input-dir .\input\real_world `
  --output-dir .\output\real_world `
  --time-limit 120 `
  --validate
```

Useful outputs:

- solution files in the chosen `--output-dir`, validated immediately after export
- one operator-statistics sidecar CSV per solution
- an optional extra batch validation pass if `--validate` is enabled
- a batch summary CSV, written by default as
  `<output-dir>/batch_summary_<input-folder>.csv`, with the columns:

```text
instance, initial_score, final_score, upper_bound, gap_to_bound_pct, elapsed_s, running_elapsed_s, improvement_pct, running_total_initial, running_total_final, running_total_upper_bound, running_total_gap_to_bound_pct, running_total_improvement_pct
```

## iRace Tuning

Use the full iRace configuration from `parameters.txt` and `scenario.txt`.

The final tuning surface contains `14` parameters:

- acceptance and restart control
- simulated-annealing cooling ratio
- perturbation strength and type bias
- construction controls (`alpha_pool`, `grasp_rcl`)
- restart construction budget ratio
- local-search stagnation limit
- the three grouped local-search operator weights

The full tuning scenario uses `maxExperiments = 1500`, while
`scenario-test.txt` provides a reduced `320`-experiment configuration.

Both iRace scenarios use `trainInstancesFile` and `testInstancesFile`.
`instances.txt` contains the deterministic training split, and
`instances-test.txt` contains the deterministic hold-out split plus the
official Google Hash Code instances for elite testing.

The current split is:

- training: `256` instances total
- test: `68` instances total
- synthetic: `112` train, `27` test
- seed_based: `108` train, `27` test
- real_world: `36` train, `9` test
- google_hashcode: `0` train, `5` test

The split is deterministic and reproducible: every fifth instance in the
ordered list of each dataset is assigned to the iRace test set, with the
remaining instances used for tuning. The five Google Hash Code instances are
then added to the iRace test set so elite selection also accounts for them.

The iRace scenarios keep progressive instance sampling enabled
(`sampleInstances = 1`) and evaluate the top `5` elites on the hold-out split
(`testNbElites = 5`). They also run up to `4` target-runner jobs in parallel
(`targetRunnerParallel = 4`).

### What each iRace file does

- `parameters.txt`: declares the parameter search space for iRace
- `scenario.txt`: main iRace run with the full `1500`-experiment budget
- `scenario-test.txt`: smaller iRace run with a `320`-experiment budget
- `instances.txt`: training instances used during tuning
- `instances-test.txt`: hold-out instances used by iRace for elite testing
- `target-runner.sh`: shell entry point called by iRace
- `targetRunnerParallel = 4`: parallel target-runner jobs used by each scenario
- `irace_runner.py`: Python adapter that receives candidate parameters from
  iRace, runs the solver on one instance, and prints a single cost value back
  to iRace

`irace_runner.py` exists because iRace expects a target runner that can:

- read one candidate configuration and one instance from the command line
- execute the solver with those values
- suppress normal solver logs
- print one numeric result in the format iRace expects

In this repository, the solver maximises score, while iRace minimises cost, so
`irace_runner.py` returns the negative score.

### Run iRace Locally

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install the R package:

```bash
Rscript -e "install.packages('irace', repos='https://cloud.r-project.org/')"
```

Make sure the target runner is executable:

```bash
chmod +x target-runner.sh
```

Run the reduced scenario:

```bash
Rscript -e "library(irace); irace(scenario=readScenario('scenario-test.txt'))"
```

Run the full scenario:

```bash
Rscript -e "library(irace); irace(scenario=readScenario('scenario.txt'))"
```


## ILS Main Loop

The ILS engine is implemented in `models/solver.py`.

For each round:

1. perturb the current home-base solution
2. run local search on the perturbed candidate
3. update the global best if improved
4. decide whether the candidate becomes the new home base
5. trigger restart if stagnation reaches the threshold

### Acceptance rule

The candidate is always accepted if it is at least as good as the home base.
Worse candidates use the standard simulated-annealing Metropolis rule:

```text
p(accept) = exp(-gap / T)
```

where `gap` is the relative score loss against the current home base. The
temperature follows a geometric cooling schedule over the ILS time budget:

```text
T(t) = T0 * sa_final_temperature_ratio^(elapsed_time / time_limit)
T0 = 0.01 / -ln(accept_worse_prob)
```

So `--accept-worse-prob` is the initial acceptance probability for a candidate
that is `1%` worse than the home base. This keeps the probability parameter
interpretable while using a textbook acceptance form.

## Perturbation

Perturbation is controlled by:

- `--perturb-strength-base`
- `--perturb-strength-growth`
- `--perturb-replace-bias`

The effective strength grows with stagnation:

```text
strength = base + stagnant_rounds * growth
```

and is then clipped using the active instance profile.

### Perturbation types

- `replace subset`: removes low-contribution signed libraries and inserts strong
  unsigned candidates
- `reorder segment`: reverses or rescored-sorts a contiguous signed block
- `shuffle segment`: randomly shuffles a contiguous signed block

The move selection rule is:

- with probability `replace_bias`, use replace-subset if there are unsigned libraries
- otherwise, use reorder up to cumulative probability `0.85`
- otherwise, use shuffle

## Restart Strategy

Restarts happen after `restart_threshold` stagnant rounds.

The new state is chosen from one of two sources:

- a fresh construction, with probability `--restart-fresh-probability`
- an elite solution from the initial candidate pool or the home-base pool

For small restart budgets, the solver uses a lightweight adaptive constructor
instead of rerunning the full initialization pipeline. Fresh adaptive restarts
sample from heap-greedy and fill-greedy construction families.

Each restarted solution also gets a short local-search pass before it becomes
the new home base.

## Instance Profiles

The solver chooses a profile automatically from instance size. The selection is
based on instance features that are directly available from the problem input:

- number of libraries
- number of days
- total book occurrences across all libraries

The profiles are:

- `small`
- `medium`
- `large`
- `huge`

These profiles set defaults for:

- iteration caps
- local-search iteration budgets
- no-improvement limits
- restart thresholds
- perturbation strength scaling
- restart init budgets
- segment caps used in perturbation

This keeps the strategy mostly dependent on instance size rather than on a
single fixed parameter set.

## CLI Usage

### Single instance

```bash
python app.py input/google_hashcode/e_so_many_books.txt output/google_hashcode/e_so_many_books.txt --time-limit 120
```

By default, `app.py` now loads the `irace` parameter set. To switch back to the
legacy CLI behavior, use:

```bash
python app.py input/google_hashcode/e_so_many_books.txt output/google_hashcode/e_so_many_books.txt --param-set default
```

### Batch mode

Batch mode should target one dataset directory at a time.

```bash
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --time-limit 300 --validate
```

### Ablation variants

```bash
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant full
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant no_perturb
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant no_restart
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant no_accept
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant no_proxy
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant no_structured
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant random_walk
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant sequential_only
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant ls_only
```

- `full`: complete solver configuration.
- `no_proxy`: disables surrogate screening inside local search; useful for the
  proxy ablation in the paper.
- `no_structured`: disables the conditional uniform/paired-instance operators;
  useful for showing how much they contribute on structured uniform instances.
- `sequential_only`: disables the portfolio decoder and scores only with the
  sequential assignment rule.

### Convergence logging

```bash
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --log-csv logs/google_hashcode
```

This writes one CSV per instance with:

```text
timestamp, elapsed_s, phase, round, current_score, best_score, event
```

Operator-statistics CSVs are opt-in. For one instance, choose the exact file:

```bash
python app.py input/google_hashcode/e_so_many_books.txt output/google_hashcode/e_so_many_books.txt --operator-stats-csv logs/e_operators.csv
```

For batch runs, choose a directory and the solver writes one operator CSV per
instance:

```bash
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --operator-stats-dir logs/operators
```

Operator-statistics CSVs contain:

```text
operator, attempts, proxy_improved, exact_checked, accepted, rejected, acceptance_rate, exact_acceptance_rate
```

## Docker Usage

The repository includes a Docker setup for both solver runs and iRace tuning.

Build the image:

```bash
docker compose build
```

Run the small iRace scenario:

```bash
docker compose run --rm irace-test
```

Run the full iRace scenario:

```bash
docker compose run --rm irace
```

Run the solver in batch mode inside Docker:

```bash
docker compose run --rm solver
```

The Docker services are defined in:

- `Dockerfile`
- `docker-compose.yml`
- `target-runner.sh`

Outputs are written to:

- `irace_output/` for iRace artifacts
- `output/` for solver outputs

## CLI Parameters

### Core run control

| Parameter | Default | Meaning |
|---|---:|---|
| `input` | None | Single input instance path |
| `output` | None | Single output path |
| `--param-set` | `irace` | Named parameter set loaded before any explicit CLI overrides |
| `--input-dir` | `input` | Batch input directory. In this repository, pass a dataset subfolder such as `input/google_hashcode` |
| `--output-dir` | `output` | Batch output directory |
| `--time-limit` | `300` | ILS improvement budget in seconds |
| `--init-max-time` | `120` | Initial construction budget in seconds |
| `--init-budget-ratio` | None | Optional cap for initial construction as a fraction of ILS time |
| `--restart-init-budget-ratio` | `0.3648` via `irace` | Fraction of remaining ILS time allocated to restart initialization |
| `--seed-solution` | None | Existing solution file used as the starting point before ILS; single-instance mode only |
| `--seed` | `54` | Random seed |
| `--quiet` | off | Suppress solver progress logs |
| `--validate` | off | Print single-output validation, or run an extra batch validation pass after per-instance validation |
| `--variant` | `full` | Ablation mode |
| `--log-csv` | None | Output directory for convergence CSVs |
| `--operator-stats-csv` | None | Single-instance path for local-search operator statistics CSV |
| `--operator-stats-dir` | None | Directory for per-instance local-search operator statistics CSVs |
| `--summary-csv` | auto | Batch summary CSV updated after each instance; defaults to `<output-dir>/batch_summary_<input-folder>.csv` |

Reproducibility note: `Solver(seed=...)` keeps its own deterministic random
state and restores Python's process-global random state after each run. The
legacy construction and tweak modules still draw through Python's `random`
module during the run, so experiments should be treated as single-process,
single-threaded seeded runs.

### Initial-solution parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--alphas` | `0.4 0.5 0.75 1.0 1.5 2.0` via `irace` | Alpha array used by construction heuristics |
| `--grasp-rcl` | `0.2318` via `irace` | GRASP restricted candidate list ratio |
| `--grasp-max-time` | `5.0` | Maximum GRASP construction time |

Named parameter sets:

- `default`: the baseline configuration that reproduces the earlier manually chosen CLI defaults
- `irace`: a configuration seeded from the latest iRace postselection run; its
  acceptance probability is interpreted by the current SA acceptance rule

Any individual CLI flag such as `--accept-worse-prob` or `--alphas` still
overrides the selected parameter set.

In the current implementation, the `irace` parameter set is the active default
for solver runs, while `default` remains available as a baseline configuration
for comparisons and ablation-style evaluation.

### ILS and local-search parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--pool-size` | auto | Elite home-base pool size |
| `--restart-threshold` | `7` via `irace` | Stagnant rounds before restart |
| `--local-no-improve-limit` | `449` via `irace` | Stop local search after this many non-improving steps |
| `--accept-worse-prob` | `0.1466` via `irace` | Initial SA acceptance probability for a `1%` worse candidate |
| `--sa-final-temperature-ratio` | `0.05` via `irace` | Final/initial temperature ratio in geometric SA cooling |
| `--ls-order-weight` | `0.7117` via `irace` | Multiplier for reorder-style local-search operators |
| `--ls-insert-weight` | `2.1158` via `irace` | Multiplier for insert/exchange local-search operators |
| `--ls-strategic-weight` | `0.7780` via `irace` | Multiplier for targeted local-search operators |
| `--enable-initial-local-search` | off | Optional pre-ILS local-search pass; not part of the current iRace tuning space |
| `--operators` | all | Restrict local search to a chosen subset of operators |

The outer ILS loop is time-driven: once started, it continues exploring new
perturbed and restarted solutions until the allotted `--time-limit` budget is
exhausted.
Each local-search phase is also time-driven, with `--local-no-improve-limit`
used only as a stagnation cutoff so control can return to perturbation or
restart logic when the current basin stops improving.

### Algorithm constants

| Constant | Value | Meaning |
|---|---:|---|
| `SA_REFERENCE_GAP` | `0.01` | Reference worsening used to map `--accept-worse-prob` to the initial temperature |
| `SA_MIN_TEMPERATURE` | `1e-12` | Lower bound for the SA temperature |
| `RESTART_PROTECTION_GAP` | `0.003` | Score gap used by restart-protection logic |
| `RESTART_PROTECTION_MULTIPLIER` | `3` | Multiplier for protected restart thresholds |

Instance-profile defaults are selected from input dimensions and total book
occurrences:

| Profile | Trigger summary | Pool | Restart | Perturb | Noisy | Local no-improve | LS time | Restart init max | Segment caps | Strength cap/divisor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `huge` | very high library count, occurrence count, or day count | `4` | `3` | `8/3` | `0` | `120` | `1.1/0.5/0.7s` | `20s` | `20/10` | `8/5` |
| `large` | high library count, occurrence count, or day count | `5` | `4` | `6/2` | `1` | `220` | `1.8/0.9/1.2s` | `30s` | `18/9` | `6/6` |
| `medium` | medium library count, occurrence count, or day count | `6` | `5` | `4/2` | `2` | `320` | `2.6/1.3/1.6s` | `45s` | `14/8` | `5/7` |
| `small` | fallback profile | `8` | `6` | `3/1` | `3` | `320` | `3.2/1.6/2.0s` | `60s` | `10/6` | `4/8` |

### Perturbation and restart parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--perturb-strength-base` | `4` via `irace` | Base perturbation strength |
| `--perturb-strength-growth` | `3` via `irace` | Additional strength per stagnant round |
| `--perturb-replace-bias` | `0.3751` via `irace` | Bias toward replace-subset perturbation |
| `--restart-fresh-probability` | `0.2679` via `irace` | Probability that restart uses a fresh construction |

## Project Structure

```text
.
|-- app.py
|-- models/
|   |-- __init__.py
|   |-- book.py
|   |-- evaluation.py
|   |-- initial_solution.py
|   |-- instance_data.py
|   |-- library.py
|   |-- local_search.py
|   |-- parser.py
|   |-- solution.py
|   |-- solver.py
|   `-- tweaks.py
|-- validator/
|   |-- multiple_validator.py
|   `-- validator.py
|-- irace_runner.py
|-- parameters.txt
|-- scenario.txt
|-- scenario-test.txt
|-- instances.txt
|-- instances-test.txt
|-- requirements.txt
`-- input/
    |-- google_hashcode/
    |-- real_world/
    |-- seed_based/
    |-- synthetic/
    `-- test/
```

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`numba` is optional but strongly recommended for larger experimental runs.

## References

Luke, S. (2014). *Essentials of Metaheuristics* (2nd ed., Version 2.1).

- Algorithm 16: Iterated Local Search with Random Restarts
- Algorithm 108: Greedy Randomized Adaptive Search Procedures

Algorithms inspired by nature 2025 class repository:

- https://github.com/ArianitHalimi/AIN_25

Dataset references:

- `google_hashcode/`: Google Hash Code 2020 Book Scanning problem instances
- `real_world/`: https://bookscanning-ig.netlify.app/
- `synthetic/`: https://github.com/uran-lajci/Book.Scanning.Dataset
- `seed_based/`: https://github.com/dritonalija/book_scanning_dataset
