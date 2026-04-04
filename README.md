# Iterated Local Search for Book Scanning

An Iterated Local Search (ILS) solver for the Google Hash Code Book Scanning
problem. The current implementation combines a multi-start construction phase,
exact JIT-accelerated evaluation, randomized local search, adaptive
perturbation, and random restarts.

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
| `real_world/` | 51 | [Book Scanning Problem Input Generator](https://bookscanning-ig.netlify.app/) | Generated instances intended to reflect real-world scanning scenarios |
| `synthetic/` | 139 | [uran-lajci/Book.Scanning.Dataset](https://github.com/uran-lajci/Book.Scanning.Dataset) | Larger synthetic training and evaluation pool |
| `seed_based/` | 135 | [dritonalija/book_scanning_dataset](https://github.com/dritonalija/book_scanning_dataset) | Seed-controlled generated instances for reproducible experiments |

Two practical notes about the current layout:

- batch runs should target a specific dataset folder such as `input/google_hashcode`
  or `input/seed_based`
- iRace tuning uses only `input/synthetic/` and `input/seed_based/`, split into
  deterministic train/test lists via `instances.txt` and `instances-test.txt`
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
  CLI option with default `5s`, but the current iRace setup keeps it fixed
  instead of tuning it.

## Exact Evaluation And Numba

Evaluation is implemented in `models/evaluation.py`.

In simple terms:

- `Numba` makes the scoring code much faster
- the final solution rebuild still uses the same scoring logic
- some search steps use faster proxy scores so they can test more moves
- if `numba` is not available, the solver still works, just more slowly

So `Numba` is a speed improvement, not a change to the objective.

## Local Search

Local search is implemented in `models/local_search.py`.

It is a first-improvement hill climber:

- start from the current solution
- sample one tweak operator at a time
- accept strict improvements immediately
- stop when time runs out, iteration cap is hit, or the no-improvement limit is reached

After the randomized phase, a small deterministic polish pass is applied if
time remains. Its budget is scaled by instance size, and it tries:

- adjacent swaps in the signed prefix
- short reinsertion moves in the signed prefix

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
}
```

The solver can also scale these operators through three grouped multipliers:

- `ls_order_weight`: affects reorder-style operators such as swaps, moves,
  reversals, and block reinsertion
- `ls_insert_weight`: affects operators that exchange signed and unsigned
  libraries or insert/remove libraries
- `ls_strategic_weight`: affects the more targeted operators
  such as `diversity_swap`

For manual experiments, the CLI also exposes per-operator overrides through
`--w-...` flags. The final iRace configuration does not tune these
individually; it tunes only the three grouped multipliers to keep the search
space compact and interpretable.

## Batch Runs And Summaries

The current codebase supports batch solving directly through `app.py`. During a
batch run you can ask the solver to keep a live summary CSV with one row per
instance.

Example:

```powershell
python .\app.py `
  --input-dir .\input\seed_based `
  --output-dir .\output\seed_based `
  --time-limit 120 `
  --summary-csv .\output\seed_based\batch_summary.csv `
  --validate
```

Useful outputs:

- solution files in the chosen `--output-dir`
- per-run validation output if `--validate` is enabled
- a batch summary CSV with the columns:

```text
instance, initial_score, final_score, elapsed_s, running_elapsed_s, improvement_pct, running_total_initial, running_total_final, running_total_improvement_pct
```

## iRace Tuning

Use the full iRace configuration from `parameters.txt` and `scenario.txt`.

The final tuning surface contains `13` parameters:

- acceptance and restart control
- perturbation strength and type bias
- construction controls (`alpha_pool`, `grasp_rcl`)
- restart construction budget ratio
- local-search stagnation limit
- the three grouped local-search operator weights

The full tuning scenario uses `maxExperiments = 900`, while
`scenario-test.txt` provides a reduced `320`-experiment configuration.

Both iRace scenarios use `trainInstancesFile` and `testInstancesFile`.
`instances.txt` contains the deterministic training split, and
`instances-test.txt` contains the deterministic hold-out split plus the

The current split is:

- training: `220` instances total
- test: `59` instances total
- synthetic: `112` train, `27` test
- seed_based: `108` train, `27` test
- google_hashcode: `0` train, `5` test

The split is deterministic and reproducible: every fifth instance in the
ordered list of each dataset is assigned to the iRace test set, with the
remaining instances used for tuning. The five Google Hash Code instances are
then added to the iRace test set so elite selection also accounts for them.

The iRace scenarios keep progressive instance sampling enabled
(`sampleInstances = 1`) and evaluate the top `5` elites on the hold-out split
(`testNbElites = 5`).

### What each iRace file does

- `parameters.txt`: declares the parameter search space for iRace
- `scenario.txt`: main iRace run with the full `900`-experiment budget
- `scenario-test.txt`: smaller iRace run with a `320`-experiment budget
- `instances.txt`: training instances used during tuning
- `instances-test.txt`: hold-out instances used by iRace for elite testing
- `target-runner.sh`: shell entry point called by iRace
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
Otherwise:

- if the relative gap is at most `0.3%`, it is accepted with probability `0.20`
- else acceptance uses:

```text
accept_worse_prob * max(0.05, 1 - gap) * (1 + min(1, stagnant_rounds / 8))
```

This introduces limited flexibility near plateaus while still favoring
high-quality solutions. In the code, these constants are named and treated as a
stagnation-aware probabilistic acceptance rule rather than an unnamed
heuristic condition.

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
instead of rerunning the full initialization pipeline.

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
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant random_walk
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --variant ls_only
```

### Convergence logging

```bash
python app.py --input-dir input/google_hashcode --output-dir output/google_hashcode --log-csv logs/google_hashcode
```

This writes one CSV per instance with:

```text
timestamp, elapsed_s, phase, round, current_score, best_score, event
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
| `--input-dir` | `input` | Batch input directory. In this repository, pass a dataset subfolder such as `input/google_hashcode` |
| `--output-dir` | `output` | Batch output directory |
| `--time-limit` | `300` | ILS improvement budget in seconds |
| `--init-max-time` | `120` | Initial construction budget in seconds |
| `--init-budget-ratio` | None | Optional cap for initial construction as a fraction of ILS time |
| `--restart-init-budget-ratio` | `0.30` | Fraction of remaining ILS time allocated to restart initialization |
| `--seed` | `54` | Random seed |
| `--quiet` | off | Suppress solver progress logs |
| `--validate` | off | Validate generated outputs |
| `--variant` | `full` | Ablation mode |
| `--log-csv` | None | Output directory for convergence CSVs |
| `--summary-csv` | None | Batch summary CSV updated after each instance |

### Initial-solution parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--alphas` | `0.5 1.0 1.5 2.0` | Alpha array used by construction heuristics |
| `--grasp-rcl` | `0.05` | GRASP restricted candidate list ratio |
| `--grasp-max-time` | `5.0` | Maximum GRASP construction time |
| `--noisy-restarts` | auto | Number of noisy construction variants from the profile |

### ILS and local-search parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--max-iterations` | auto | Maximum outer ILS rounds |
| `--pool-size` | auto | Elite home-base pool size |
| `--restart-threshold` | auto | Stagnant rounds before restart |
| `--local-no-improve-limit` | auto | Stop local search after this many non-improving steps |
| `--accept-worse-prob` | `0.04` | Base probability for accepting worse candidates |
| `--ls-order-weight` | `1.0` | Multiplier for reorder-style local-search operators |
| `--ls-insert-weight` | `1.0` | Multiplier for insert/exchange local-search operators |
| `--ls-strategic-weight` | `1.0` | Multiplier for targeted local-search operators |
| `--operators` | all | Restrict local search to a chosen subset of operators |

### Perturbation and restart parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--perturb-strength-base` | auto | Base perturbation strength |
| `--perturb-strength-growth` | auto | Additional strength per stagnant round |
| `--perturb-replace-bias` | `0.65` | Bias toward replace-subset perturbation |
| `--restart-fresh-probability` | `0.35` | Probability that restart uses a fresh construction |
| `--enable-initial-local-search` | off | Run the optional pre-ILS local-search pass |
| `--enable-direct-intensify` | off | Run the optional direct intensification phase before ILS |

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
