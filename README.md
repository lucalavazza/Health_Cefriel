# SNCS Fitness Causal Pipeline

This repository branch contains the release package supporting the manuscript
_How Reliable Is a Causal-Analysis Pipeline? Evidence from a Synthetic
Health-Fitness Dataset_.

The release focuses on the SNCS audit workflow. It contains the executable
pipeline, the derived dataset used by the workflow, and the corrected numerical
artifacts reported in the paper.

## Contents

- `sncs_mpu.py`: canonical SNCS workflow implementation.
- `run_pipeline.py`: compatibility entry point for running the SNCS workflow.
- `run_experiment.py`: command-line dispatcher.
- `datasets/averaged_health_fitness_dataset.csv`: derived monthly input table
  with 36,000 rows, corresponding to twelve monthly records for each of 3,000
  participants.
- `artifacts/sncs_mpu_corrected/`: corrected result artifacts used in the
  manuscript.
- `tests/test_sncs_mpu.py`: focused regression tests for the SNCS workflow.
- `requirements-sncs-mpu.txt`: Python dependencies for the release workflow.

Bulk exploratory outputs, participant-level split files, IDE metadata, cache
files, and superseded artifacts are intentionally excluded from this release
branch.

## Reproducing The Reported Artifacts

Use Python 3.10.12. From the repository root:

```bash
python -m venv .venv_sncs_mpu
source .venv_sncs_mpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sncs-mpu.txt
python run_pipeline.py sncs_mpu
```

The default command reads:

```text
datasets/averaged_health_fitness_dataset.csv
```

and writes:

```text
artifacts/sncs_mpu_corrected/
```

To write a new output directory instead of overwriting the archived artifacts:

```bash
python run_experiment.py --pipeline sncs_mpu \
  --input datasets/averaged_health_fitness_dataset.csv \
  --output artifacts/sncs_mpu_rerun
```

## Validation

Run the focused tests with:

```bash
python -m pytest tests/test_sncs_mpu.py
```

The archived manifest in `artifacts/sncs_mpu_corrected/manifest.json` records
the numerical run commit, random seed, preprocessing choices, causal-discovery
settings, intervention value, retained variables, and package versions used for
the reported results.

## Data

The source data are the public _FitLife: Health & Fitness Tracking Dataset_
distributed through Kaggle under the CC0 Public Domain licence. The release
workflow uses the derived monthly table in `datasets/averaged_health_fitness_dataset.csv`
and then aggregates it internally to one participant-level row per participant
before discovery, fitting, and held-out evaluation.

## Citation

If you use this release, cite the accompanying paper and the archived software
release DOI once available.
