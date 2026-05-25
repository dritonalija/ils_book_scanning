# Experiment Workflow

This document defines the experiment workflow for the ILS Book Scanning solver.
It is intended to be used as the command checklist for running tuning,
evaluation, ablation, convergence, and statistical analysis.

## Common Settings

Final evaluation settings:

- ILS budget: `600s`
- initial construction cap: `120s`
- runs per instance: `10`
- seed range: `1..100`
- worker count: `6`, adjust downward if the machine shows memory pressure,
  swap usage, or sustained high load

The generated 10 seeds are:

```text
1, 12, 23, 34, 45, 56, 67, 78, 89, 100
```

Unless `--results-csv` is provided explicitly, `run_experiments.py` writes raw
per-run results to:

```text
<output-dir>/experiment_results.csv
```

Each run is also exported as a solution file under:

```text
<output-dir>/<algorithm>/<run_label>/seed_XXX/<instance_path>
```

The runner is resumable. If a run is interrupted, start the same command again.
Rows already marked with `status=ok` in `experiment_results.csv` are skipped.

## 1. Prepare The Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the iRace target runner is executable:

```bash
chmod +x target-runner.sh
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run a short target-runner smoke test:

```bash
IRACE_TIME_LIMIT=1 IRACE_INIT_BUDGET_CAP=1 \
python irace_runner.py 1 1 54 input/synthetic/synthetic_5.txt --alpha-pool default
```

Optionally freeze the exact runtime environment:

```bash
pip freeze > requirements-lock.txt
```

## 2. iRace Pilot

Use the small iRace scenario first to verify that the tuning pipeline runs
cleanly:

```bash
IRACE_TIME_LIMIT=120 IRACE_INIT_BUDGET_CAP=30 \
Rscript -e "library(irace); irace(scenario=readScenario('scenario-test.txt'))"
```

Expected purpose:

- verify R/iRace setup
- verify `target-runner.sh`
- verify Python runner imports
- verify candidate parameters are accepted
- catch filesystem or permission issues before the full tuning run

## 3. Full iRace Tuning

Run the full iRace scenario:

```bash
Rscript -e "library(irace); irace(scenario=readScenario('scenario.txt'))"
```

Current tuning configuration:

- iRace seed: `54`
- iRace ILS budget per target run: `300s`
- iRace initial construction cap: `120s`
- `maxExperiments = 5000`
- `parallel = 6`

Current extracted tuning result in this repository:

- parameter set: `irace_final`
- iRace elite configuration ID: `486`

If iRace is rerun:

1. Extract the winning elite configuration from the iRace output.
2. Add it to `parameter_sets.py` as a new parameter set, for example
   `irace_final`.
3. Do not overwrite old parameter sets unless intentionally replacing them.
4. Commit the parameter set before final evaluation.

## 4. Transfer Check At 600s

Before running the full final evaluation, verify that the tuned configuration
transfers from the `300s` tuning budget to the `600s` final budget.

Use a small validation subset, for example `diverse_instances.txt`:

```bash
python run_experiments.py \
  --instances-file diverse_instances.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --time-limit 600 \
  --init-max-time 120 \
  --seeds 1 45 100 \
  --workers 6 \
  --output-dir output/transfer_check
```

Compare against the previous/default parameter set if needed:

```bash
python run_experiments.py \
  --instances-file diverse_instances.txt \
  --algorithm ILS_DefaultParams \
  --param-set irace \
  --variants full \
  --time-limit 600 \
  --init-max-time 120 \
  --seeds 1 45 100 \
  --workers 6 \
  --output-dir output/transfer_check_default
```

Analyze the transfer check:

```bash
python analyze_results.py \
  --results-csv \
    output/transfer_check/experiment_results.csv \
    output/transfer_check_default/experiment_results.csv \
  --reference-method ILS_iRace:full \
  --paired-metric mean \
  --wilcoxon-alternative greater \
  --output-dir results/analysis_transfer_check
```

Proceed to final evaluation after confirming that the tuned configuration is
competitive at `600s`.

## 5. Full Final Evaluation

Run the final solver configuration on all training and test instances:

```bash
python run_experiments.py \
  --instances-file instances.txt \
  --instances-file instances-test.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --time-limit 600 \
  --init-max-time 120 \
  --runs-per-instance 10 \
  --seed-range 1 100 \
  --workers 6 \
  --output-dir output/experiments_full
```

Raw results:

```text
output/experiments_full/experiment_results.csv
```

This run produces:

```text
(number of instances in instances.txt + instances-test.txt) * 10
```

individual solver runs.

## 6. Component Ablation

Run component ablation only on the test set:

```bash
python run_experiments.py \
  --instances-file instances-test.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --component-variants \
  --time-limit 600 \
  --init-max-time 120 \
  --runs-per-instance 10 \
  --seed-range 1 100 \
  --workers 6 \
  --output-dir output/component_ablation
```

This evaluates:

```text
full
ls_only
no_perturb
no_restart
no_accept
no_proxy
```

`sequential_only` and `no_structured` remain available as manual variants, but
they are not included in the default component set. `sequential_only` evaluates
the decoder choice, while `no_structured` is relevant only for structured
uniform instances.

Raw results:

```text
output/component_ablation/experiment_results.csv
```

## 7. Convergence Runs

Run convergence logging on the 10 selected diverse test instances:

```bash
python run_experiments.py \
  --instances-file diverse_instances.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --time-limit 600 \
  --init-max-time 120 \
  --seeds 1 45 100 \
  --workers 3 \
  --output-dir output/convergence \
  --convergence-instances-file diverse_instances.txt \
  --convergence-seeds 1 45 100
```

Raw results:

```text
output/convergence/experiment_results.csv
```

Convergence logs:

```text
logs/convergence/ILS_iRace/full/seed_XXX/...
```

Plot round-vs-score curves:

```bash
python plot_convergence.py \
  --log-root logs/convergence \
  --instances-file diverse_instances.txt \
  --x-axis round \
  --output-dir results/plots/convergence_round
```

Plot time-vs-score curves:

```bash
python plot_convergence.py \
  --log-root logs/convergence \
  --instances-file diverse_instances.txt \
  --x-axis elapsed_s \
  --output-dir results/plots/convergence_time
```

## 8. Operator Group Ablation

Run grouped local-search operator ablation on the test set:

```bash
python run_experiments.py \
  --instances-file instances-test.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --operator-group-ablations all \
  --time-limit 600 \
  --init-max-time 120 \
  --runs-per-instance 10 \
  --seed-range 1 100 \
  --workers 6 \
  --output-dir output/operator_group_ablation
```

This evaluates the impact of removing each operator group:

```text
drop_order_operators
drop_insert_operators
drop_strategic_operators
```

Raw results:

```text
output/operator_group_ablation/experiment_results.csv
```

## 9. Individual Operator Ablation

Operator-level ablation is more expensive than component ablation. Run it on a
smaller subset first.

Example on the 10 diverse instances with 5 runs:

```bash
python run_experiments.py \
  --instances-file diverse_instances.txt \
  --algorithm ILS_iRace \
  --param-set irace_final \
  --variants full \
  --operator-ablations all \
  --time-limit 600 \
  --init-max-time 120 \
  --runs-per-instance 5 \
  --seed-range 1 100 \
  --workers 6 \
  --output-dir output/operator_ablation
```

Raw results:

```text
output/operator_ablation/experiment_results.csv
```

## 10. Baseline Results From Other Algorithms

External baseline results should be converted to CSV before analysis.

Minimum accepted format:

```csv
status,algorithm,instance,seed,final_score,elapsed_s
ok,Erecto,input/synthetic/synthetic_5.txt,1,28600,600
ok,Algorithm_A,input/synthetic/synthetic_5.txt,1,27900,600
```

Recommended format:

```csv
status,algorithm,run_label,instance,dataset,seed,final_score,elapsed_s
ok,Erecto,full,input/synthetic/synthetic_5.txt,synthetic,1,28600,600
ok,Algorithm_A,full,input/synthetic/synthetic_5.txt,synthetic,1,27900,600
```

Save the file as:

```text
results/baselines.csv
```

The `instance` column must match the instance paths used by this repository,
for example:

```text
input/synthetic/synthetic_5.txt
input/real_world/B1000k_L115_D230.txt
```

If a baseline has one result per instance, use `seed=1`. If it has multiple
runs, use the actual run seeds or run ids.

## 11. Statistical Analysis

Analyze final ILS results against external baselines:

```bash
python analyze_results.py \
  --results-csv \
    output/experiments_full/experiment_results.csv \
    results/baselines.csv \
  --reference-method ILS_iRace:full \
  --paired-metric mean \
  --wilcoxon-alternative greater \
  --output-dir results/analysis_full
```

Analyze component ablation:

```bash
python analyze_results.py \
  --results-csv output/component_ablation/experiment_results.csv \
  --reference-method ILS_iRace:full \
  --paired-metric mean \
  --wilcoxon-alternative greater \
  --output-dir results/analysis_component_ablation
```

Analyze operator ablation:

```bash
python analyze_results.py \
  --results-csv output/operator_ablation/experiment_results.csv \
  --reference-method ILS_iRace:full \
  --paired-metric mean \
  --wilcoxon-alternative greater \
  --output-dir results/analysis_operator_ablation
```

Analyze operator group ablation:

```bash
python analyze_results.py \
  --results-csv output/operator_group_ablation/experiment_results.csv \
  --reference-method ILS_iRace:full \
  --paired-metric mean \
  --wilcoxon-alternative greater \
  --output-dir results/analysis_operator_group_ablation
```

Each analysis writes:

```text
per_instance_summary.csv
method_summary.csv
wilcoxon_tests.csv
friedman_tests.csv
```

The Wilcoxon table includes raw and corrected p-values:

```text
p_value
p_value_holm
p_value_bh
p_value_corrected
```

## 12. Recommended Execution Order

Use this order:

1. Environment setup and smoke tests
2. iRace pilot
3. full iRace tuning
4. add tuned parameter set to `parameter_sets.py`
5. transfer check at `600s`
6. full final evaluation
7. component ablation
8. operator group ablation
9. convergence runs and plots
10. optional individual operator ablation
11. statistical analysis

Keep each experiment in its own `--output-dir` so raw results and solution files
remain separated.
