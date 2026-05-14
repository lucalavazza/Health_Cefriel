# Health_Cefriel

`Health_Cefriel` is a research codebase for health and fitness data analysis, causal discovery, structural causal modeling, and counterfactual analysis.

The project is organized as a numbered pipeline of scripts. Starting from a raw activity dataset, it:

1. explores the source data,
2. builds monthly participant-level training and testing datasets,
3. learns a causal graph,
4. fits causal models with DoWhy,
5. derives a linear SCM,
6. validates the resulting equations,
7. runs persona-based and narrative counterfactual analyses.

## Repository Overview

- `steps/`
  Numbered pipeline scripts. This is the main entry area for the analyses.
  - `steps/step_0_data_analysis.py`
    Exploratory data analysis and descriptive plots from the raw dataset.
  - `steps/step_1_dataset_management.py`
    Preprocessing pipeline. Splits the raw dataset by participant, aggregates by month, standardizes numeric variables, and creates labelled and one-hot encoded datasets.
  - `steps/step_2_causal_learn_analysis.py`
    Causal graph discovery with CausalLearn. The learned PC graph is the graph used downstream by the SCM scripts.
  - `steps/step_2a_time_series_analysis.py`
    Aggregate time-series causal discovery with Tigramite.
  - `steps/step_2b_time_series_analysis_single_pids.py`
    Per-participant time-series causal discovery.
  - `steps/step_3a_dowhy_analysis.py`
    DoWhy causal effect estimation, refutation, intervention simulation, and an example counterfactual analysis.
  - `steps/step_3b_linear_scm.py`
    Builds a DoWhy-based SCM, exports algebraic equations, computes GCM and linear-SCM counterfactuals, and reports agreement diagnostics.
  - `steps/step_3c_scm_validation.py`
    Validates the algebraic equations exported by `steps/step_3b_linear_scm.py`.
  - `steps/step_4a_personas_analysis.py`
    Counterfactual scenarios for selected persona participants.
  - `steps/step_4b_influence_analysis.py`
    Narrative influence analysis for a selected participant.
- `pipeline/`
  Shared internals used by the numbered scripts and the pipeline runner.
  - `pipeline/config.py`
    Shared paths, experiment settings, dataset schema constants, and graph-loading helper.
  - `pipeline/preprocessing.py`
    Shared preprocessing helpers, including scaling metadata and original-unit conversion utilities.
  - `pipeline/artifacts.py`
    Cleanup, output-archiving, and run-manifest helpers for the pipeline runner.
- `checks/`
  Lightweight automated validation scripts for generated outputs and pipeline consistency.
  - `checks/validate_pipeline_outputs.py`
    Verifies that the expected downstream artifacts exist, are non-empty, and remain consistent with the active pipeline configuration.
- `run_pipeline.py`
  CLI wrapper to clean outputs, run grouped steps, and archive artifacts under `artifacts/`.

The `artifacts/` directory is created only if you use `run_pipeline.py` with output snapshots enabled.

## Repository Layout

```text
.
├── pipeline/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── config.py
│   └── preprocessing.py
├── checks/
│   ├── __init__.py
│   └── validate_pipeline_outputs.py
├── steps/
│   ├── __init__.py
│   ├── step_0_data_analysis.py
│   ├── step_1_dataset_management.py
│   ├── step_2_causal_learn_analysis.py
│   ├── step_2a_time_series_analysis.py
│   ├── step_2b_time_series_analysis_single_pids.py
│   ├── step_3a_dowhy_analysis.py
│   ├── step_3b_linear_scm.py
│   ├── step_3c_scm_validation.py
│   ├── step_4a_personas_analysis.py
│   └── step_4b_influence_analysis.py
├── run_pipeline.py
├── datasets/
├── participants/
├── graphs/
├── linear_scm/
└── ppt/
```

Important generated locations:

- `data_analysis/`
  EDA figures from `steps/step_0_data_analysis.py`.
- `datasets/`
  Raw input dataset plus generated processed datasets and preprocessing metadata.
- `participants/`
  Per-participant and per-month derived CSV files generated during step 1.
- `graphs/causallearn/`
  Learned graph images and edge exports.
- `graphs/time_series_graphs/`
  Tigramite outputs from `2a` and `2b`.
- `graphs/tests/`
  Example DoWhy intervention and counterfactual outputs from `3a`.
- `graphs/counterfactuals/`
  Persona plots from `4a`.
- `graphs/influences/`
  Influence-analysis plots from `4b`.
- `linear_scm/`
  SCM structure, equations, counterfactual exports, epsilon summaries, validation metrics, and agreement diagnostics from `3b` and `3c`.

## Installation

### Requirements

- Python 3.8+
- Conda is the easiest setup path

`environment.yml` currently pins Python `3.10`. The scripts are also compatible with Python `3.8`, which is useful if you are running them from an existing environment.

The repository includes:

- `environment.yml`
  Recommended Conda environment definition.
- `requirements.txt`
  `pip` dependencies for non-Conda setups.

Recommended Conda setup:

```bash
git clone https://github.com/lucalavazza/Health_Cefriel.git
cd Health_Cefriel
conda env create -f environment.yml
conda activate health_cefriel
```

Alternative `venv` setup:

```bash
git clone https://github.com/lucalavazza/Health_Cefriel.git
cd Health_Cefriel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your shell does not expose the environment as `python`, use `python3` or the full interpreter path instead.

## Input Data

The pipeline starts from:

- `datasets/health_fitness_dataset.csv`

Step `steps/step_1_dataset_management.py` generates, among other files:

- `datasets/averaged_health_fitness_dataset_training.csv`
- `datasets/averaged_health_fitness_dataset_testing.csv`
- `datasets/regularised_averaged_health_fitness_dataset_training.csv`
- `datasets/regularised_averaged_health_fitness_dataset_testing.csv`
- `datasets/encoded_regularised_averaged_health_fitness_dataset_training.csv`
- `datasets/encoded_regularised_averaged_health_fitness_dataset_testing.csv`
- `datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv`
- `datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv`
- `datasets/preprocessing_metadata.json`

`preprocessing_metadata.json` is used downstream to convert standardized variables back to original units for reporting and intervention definitions.

## Workflow

### Full numbered order

1. `steps/step_0_data_analysis.py`
2. `steps/step_1_dataset_management.py`
3. `steps/step_2_causal_learn_analysis.py`
4. `steps/step_2a_time_series_analysis.py`
5. `steps/step_2b_time_series_analysis_single_pids.py`
6. `steps/step_3a_dowhy_analysis.py`
7. `steps/step_3b_linear_scm.py`
8. `steps/step_3c_scm_validation.py`
9. `steps/step_4a_personas_analysis.py`
10. `steps/step_4b_influence_analysis.py`

### Main SCM workflow

The core causal workflow is:

1. `steps/step_1_dataset_management.py`
2. `steps/step_2_causal_learn_analysis.py`
3. `steps/step_3a_dowhy_analysis.py`
4. `steps/step_3b_linear_scm.py`
5. `steps/step_3c_scm_validation.py`
6. `steps/step_4a_personas_analysis.py`
7. `steps/step_4b_influence_analysis.py`

`steps/step_0_data_analysis.py`, `steps/step_2a_time_series_analysis.py`, and `steps/step_2b_time_series_analysis_single_pids.py` are useful companion analyses, but they are not required to run the main SCM path.

### Interventions and units

From the current version of the codebase onward:

- numeric intervention scenarios are defined in original units,
- the scripts convert them internally to standardized space when needed,
- plots and summary outputs are reported back in original units when scaling metadata is available.

This affects `steps/step_3a_dowhy_analysis.py`, `steps/step_3b_linear_scm.py`, `steps/step_4a_personas_analysis.py`, and `steps/step_4b_influence_analysis.py`.

## Running The Codebase

### Step by step

Run the step modules from the repository root:

```bash
python3 -m steps.step_1_dataset_management
python3 -m steps.step_2_causal_learn_analysis
python3 -m steps.step_3a_dowhy_analysis
python3 -m steps.step_3b_linear_scm
python3 -m steps.step_3c_scm_validation
python3 -m steps.step_4a_personas_analysis
python3 -m steps.step_4b_influence_analysis
```

Optional analyses:

```bash
python3 -m steps.step_0_data_analysis
python3 -m steps.step_2a_time_series_analysis
python3 -m steps.step_2b_time_series_analysis_single_pids
```

### CLI wrapper

You can also use `run_pipeline.py` from the repository root. It is a thin orchestrator around the modules in `steps/`.

```bash
python3 run_pipeline.py --list
python3 run_pipeline.py full
python3 run_pipeline.py dataset causal_graph linear_scm
python3 run_pipeline.py optional
python3 run_pipeline.py validate
python3 run_pipeline.py clean
python3 run_pipeline.py --run-label experiment_a full
```

If the project dependencies live in a different interpreter:

```bash
python3 run_pipeline.py --python /path/to/python full
```

By default, `run_pipeline.py` snapshots each completed step under `artifacts/<run-label>/...` and writes a `manifest.json` with the step name, timestamp, git commit, interpreter, and output summary.

#### Positional arguments

`run_pipeline.py` accepts zero or more positional step selectors:

- individual step names such as `dataset`, `causal_graph`, or `linear_scm`
- pipeline groups such as `full` and `optional`
- the special command `clean`

If you run the command without positional arguments, it defaults to:

```bash
python3 run_pipeline.py full
```

Available step names:

- `eda`
- `dataset`
- `causal_graph`
- `time_series`
- `time_series_pids`
- `dowhy`
- `linear_scm`
- `scm_validation`
- `personas`
- `influence`
- `validate`

Available pipeline groups:

- `full`
  Runs the main SCM workflow.
- `optional`
  Runs EDA and time-series analyses.

The special command:

- `clean`
  Removes generated outputs while preserving `datasets/health_fitness_dataset.csv` and `ppt/`

You can combine selectors in one command. For example:

```bash
python3 run_pipeline.py clean dataset causal_graph
python3 run_pipeline.py dataset causal_graph dowhy linear_scm scm_validation
python3 run_pipeline.py dataset causal_graph dowhy linear_scm scm_validation personas influence validate
```

When `clean` appears in the same command, it runs first and the selected analysis steps run afterward.

#### Optional flags

- `--list`
  Prints the available step names, pipeline groups, and the `clean` command, then exits.
- `--python /path/to/python`
  Chooses the interpreter used to launch the step scripts. This is useful when the right environment is not the one running `run_pipeline.py` itself.
- `--run-label LABEL`
  Sets the archive label under `artifacts/`. If omitted, a UTC timestamp is used.
- `--no-snapshot`
  Disables output archiving under `artifacts/<run-label>/`. The selected steps still run normally.
- `--include-artifacts`
  Only meaningful with `clean`. It also removes archived runs stored under `artifacts/`.

Examples:

```bash
python3 run_pipeline.py --list
python3 run_pipeline.py --no-snapshot full
python3 run_pipeline.py --run-label may_rerun dataset causal_graph dowhy
python3 run_pipeline.py clean --include-artifacts
python3 run_pipeline.py --python /Users/you/miniconda3/envs/health_cefriel/bin/python full
```

### Validation

After rerunning graph-dependent steps, you can validate the current outputs with either:

```bash
python3 -m checks.validate_pipeline_outputs
```

or:

```bash
python3 run_pipeline.py validate
```

The validation script checks:

- the active causal-discovery summary and its referenced graph artifacts,
- the expected `3a`, `3b`, `3c`, `4a`, and `4b` outputs,
- key file non-emptiness,
- `linear_scm/metrics.csv` structure,
- consistency between `metrics.csv`, `algebraic_equations.txt`, and the original-units counterfactual exports.

## Reproducibility

### Canonical reruns

After activating the environment, the recommended rerun commands are:

Main SCM workflow:

```bash
python3 run_pipeline.py clean
python3 run_pipeline.py full
python3 run_pipeline.py validate
```

Extended rerun including EDA and time-series analyses:

```bash
python3 run_pipeline.py clean
python3 run_pipeline.py full optional validate
```

If you prefer to keep existing outputs and only recompute the main SCM path:

```bash
python3 run_pipeline.py full validate
```

### Step-by-step reproducibility

If you want to rerun the pipeline manually instead of using `run_pipeline.py`, the canonical order is:

```bash
python3 -m steps.step_1_dataset_management
python3 -m steps.step_2_causal_learn_analysis
python3 -m steps.step_3a_dowhy_analysis
python3 -m steps.step_3b_linear_scm
python3 -m steps.step_3c_scm_validation
python3 -m steps.step_4a_personas_analysis
python3 -m steps.step_4b_influence_analysis
python3 -m checks.validate_pipeline_outputs
```

Optional companion analyses:

```bash
python3 -m steps.step_0_data_analysis
python3 -m steps.step_2a_time_series_analysis
python3 -m steps.step_2b_time_series_analysis_single_pids
```

### Expected outputs by step

- `steps.step_1_dataset_management`
  Expected outputs:
  `datasets/averaged_*`, `datasets/regularised_*`, `datasets/encoded_*`, `datasets/labelled_*`, `datasets/preprocessing_metadata.json`, and per-participant files under `participants/`.
- `steps.step_2_causal_learn_analysis`
  Expected outputs:
  `graphs/causallearn/edges/`, `graphs/causallearn/graphs/`, and `graphs/causallearn/labelled_execution_times.json`.
- `steps.step_3a_dowhy_analysis`
  Expected outputs:
  `graphs/tests/test-intervention.pdf` and the `test-counterfactual-pid=*` PDFs.
- `steps.step_3b_linear_scm`
  Expected outputs:
  `linear_scm/scm.txt`, `linear_scm/scm_coefficients.json`, `linear_scm/algebraic_equations.txt`, the counterfactual JSON files, epsilon summaries, and `linear_vs_gcm_mae_original_units.{json,csv}`.
- `steps.step_3c_scm_validation`
  Expected outputs:
  `linear_scm/metrics.csv` and `linear_scm/metrics_interpretation.txt`.
- `steps.step_4a_personas_analysis`
  Expected outputs:
  three PDFs per selected PID under `graphs/counterfactuals/`:
  base plot, original-units plot, and percentage-difference plot.
- `steps.step_4b_influence_analysis`
  Expected outputs:
  the influence PDFs under `graphs/influences/`, including the tennis-only, tennis-plus-steps, and duration-adjustment scenarios.

### Artifact map for manuscript or deck preparation

This repository does not currently include manuscript source files, so exact figure numbers are not encoded in the repo itself. The mapping below is the canonical source by result family.

- Causal graph figure:
  `graphs/causallearn/graphs/labelling/PC-labelling/labelling_causal_graph_causal-learn_pc_fisherz.png`
- Example intervention / counterfactual figures:
  `graphs/tests/test-intervention.pdf` and `graphs/tests/test-counterfactual-pid=42*.pdf`
- Persona result figures:
  `graphs/counterfactuals/*.pdf`
  and the assembled presentation files under `ppt/personas_with_graphs.{pdf,pptx}`
- Influence-analysis figures:
  `graphs/influences/*.pdf`
  and the assembled presentation files under `ppt/inference and influences.{pdf,pptx}`
- Structural-equation and SCM table sources:
  `linear_scm/algebraic_equations.txt`, `linear_scm/scm.txt`, `linear_scm/scm_coefficients.json`
- Validation / quantitative table sources:
  `linear_scm/metrics.csv`, `linear_scm/metrics_interpretation.txt`, `linear_scm/linear_vs_gcm_mae_original_units.csv`, `linear_scm/linear_vs_gcm_mae_original_units.json`

### What to cite as the validated final state

After a successful rerun, the minimum final-state checks are:

```bash
python3 run_pipeline.py validate
```

and:

- `graphs/causallearn/labelled_execution_times.json` points to the active PC graph files
- `linear_scm/metrics.csv` is present and non-empty
- `linear_scm/linear_vs_gcm_mae_original_units.csv` is present and non-empty
- `graphs/counterfactuals/` contains all persona PDFs
- `graphs/influences/` contains the expected influence PDFs

## Cleanup And Reruns

To remove generated outputs while keeping the raw dataset and `ppt/` untouched:

```bash
python3 run_pipeline.py clean
```

To also remove archived runs under `artifacts/`:

```bash
python3 run_pipeline.py clean --include-artifacts
```

`clean` removes generated contents under:

- `data_analysis/`
- `graphs/`
- `linear_scm/`
- `participants/`
- all generated files in `datasets/`, while preserving `datasets/health_fitness_dataset.csv`

### Practical rerun boundaries

- If you change preprocessing in `steps/step_1_dataset_management.py`:
  rerun `1` and everything downstream that depends on the processed datasets.
- If you change graph discovery in `steps/step_2_causal_learn_analysis.py`:
  rerun `2`, then `3a`, `3b`, `3c`, `4a`, and `4b`.
- If you change the linear SCM or validation logic:
  rerun `3b`, `3c`, `4a`, and `4b`.
- If you change only persona or influence scenarios:
  rerun just `4a` and/or `4b`.

Before rerunning `3b`, `3c`, `4a`, and `4b`, it is a good idea to clear:

- `linear_scm/`
- `graphs/counterfactuals/`
- `graphs/influences/`

That cleanup is recommended for clarity, not strictly required for correctness.

## Main Outputs

### Causal graph

Key downstream graph artifacts include:

- `graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy`
- `graphs/causallearn/edges/txt/labelling_causal_graph_causal-learn_pc_fisherz.txt`

These learned edges are used by the DoWhy and SCM scripts.

### DoWhy example outputs

`steps/step_3a_dowhy_analysis.py` writes example outputs under `graphs/tests/`, including:

- an intervention comparison plot,
- a counterfactual example for `TEST_COUNTERFACTUAL_PID`,
- percentage-difference plots,
- original-unit companion plots when metadata is available.

### Linear SCM outputs

`steps/step_3b_linear_scm.py` writes:

- `linear_scm/scm.txt`
- `linear_scm/scm_coefficients.json`
- `linear_scm/algebraic_equations.txt`
- `linear_scm/cf_results.json`
- `linear_scm/cf_results_original_units.json`
- `linear_scm/linear_cf_results.json`
- `linear_scm/linear_cf_results_original_units.json`
- `linear_scm/epsilon_results.json`
- `linear_scm/epsilon_means_by_pid_and_var.json`
- `linear_scm/linear_vs_gcm_mae_original_units.json`
- `linear_scm/linear_vs_gcm_mae_original_units.csv`

The agreement files summarize how closely the linear SCM reproduces the GCM counterfactual outputs in original units.

### Validation outputs

`steps/step_3c_scm_validation.py` writes:

- `linear_scm/metrics.csv`
- `linear_scm/metrics_interpretation.txt`

The metrics file includes validation scores per target variable, including original-unit RMSE and MAE when scaling metadata exists.

### Persona and influence outputs

- `steps/step_4a_personas_analysis.py` writes persona-specific plots under `graphs/counterfactuals/`
- `steps/step_4b_influence_analysis.py` writes influence-analysis plots under `graphs/influences/`

Both scripts produce original-unit reporting where applicable.
