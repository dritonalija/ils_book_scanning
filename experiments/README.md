# Experimental Records

This folder holds the version-controlled record of every experiment batch. The
raw `output/` and `logs/` directories stay out of Git because they contain
solution files and large run artifacts, so the files here are the published
copy of the results.

Everything in this folder is an input to the analysis scripts in the repository
root. A fresh clone can rebuild every summary from it.

## Contents

### `output_csv/`

One run export per batch, in the structure the runs wrote.

| Batch | Runs | What it varies |
|---|---:|---|
| `experiments_full/` | 3,240 | the tuned algorithm over the full instance set |
| `weak_sorted_init/` | 680 | a weaker initial solution |
| `component_ablation/` | 2,720 | one algorithm component removed per variant |
| `operator_group_ablation/` | 2,040 | one operator group removed per variant |
| `convergence/` | 100 | instrumented runs that log their score trace |
| `external_baselines/` | 2,592 | eight external methods over all 324 instances |

Each row is one run. The `variant`, `component` and `omitted_operator` columns
record what the run changed, and `run_label` names the variant. In the
operator-group batch, `variant` and `component` hold one constant value for
every row, so `run_label` is the column that separates the variants there.

`external_baselines/` is the odd one out: it holds external methods rather than
runs of this solver, so its rows carry no initial score and no search time. The
`BestExternal` rows are the per-instance best of the other methods. A method
that failed on an instance is recorded with a status other than `ok`, which is
why the MILP methods cover fewer instances than the rest.

### `convergence_logs/`

Two logs per instrumented run, 200 files:

- `<instance>.csv` — the score trace. One row per status point, with
  `elapsed_s`, `current_score` and `best_score`. Construction and the initial
  local search are measured on their own clocks, so the search clock starts
  again at zero in the `ils` rows.
- `<instance>.operators.csv` — attempts, proxy improvements, exact checks,
  acceptances and rejections per operator for that run.

The `log_csv` column of `output_csv/convergence/experiment_results.csv` records
the path each run wrote to. Only the part below `logs/convergence/` matches
this folder.

### `analysis/convergence/`

Built from the logs above by `analyze_convergence.py`:

- `convergence_checkpoint_summary.csv` — the best score each run had reached at
  0, 30, 60, 120, 300 and 600 seconds of search, and how far along its total
  improvement that is. A run that never improved leaves the progress column
  empty, because there is no scale to measure it on.
- `convergence_operator_summary.csv` — the operator counters summed over the
  100 runs, sorted by accepted moves.

### `result_tables/`

Tidy per-seed tables built from `output_csv/` by `export_result_tables.py`:
successful runs only, trimmed to the result columns, with instance paths
reduced to file names. These are the input to `verify_results.py`.

### `instance_catalog.csv`

Structural properties of all 324 instances, built from the instance files by
`build_instance_catalog.py`.

### `expected_results.json`

The baseline that `verify_results.py` checks against.

### `irace/`

The iRace tuning snapshot: the `.Rdata` object, the tuning scenarios, and the
parameter search space.

## Regenerating

The derived files here are outputs. Change the script, not the file:

| File | Built by |
|---|---|
| `instance_catalog.csv` | `build_instance_catalog.py` |
| `analysis/convergence/*.csv` | `analyze_convergence.py` |
| `result_tables/*.csv` | `export_result_tables.py` |
| `expected_results.json` | `verify_results.py --update` |

The run exports under `output_csv/` and the logs under `convergence_logs/` are
not derived. They come from the runs themselves, and the ignored `output/` and
`logs/` directories are the only other place they exist. Do not delete them
expecting a script to rebuild them; that needs the experiments run again.

The command checklist and the experiment protocol are in
`../docs/experiments.md`.
