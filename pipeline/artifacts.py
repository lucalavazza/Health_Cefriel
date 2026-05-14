from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import (
    ARTIFACTS_DIR,
    CAUSALLEARN_EDGE_NPY_DIR,
    CAUSALLEARN_EDGE_TXT_DIR,
    CAUSALLEARN_GRAPHS_DIR,
    COUNTERFACTUALS_DIR,
    DATASETS_DIR,
    GRAPHS_DIR,
    INFLUENCES_DIR,
    LINEAR_SCM_DIR,
    PROCESSED_DATASETS,
    RAW_DATASET,
    ROOT_DIR,
    TESTS_OUTPUT_DIR,
    TIME_SERIES_OUTPUT_DIR,
    TIME_SERIES_PIDS_OUTPUT_DIR,
)


GENERATED_DIRECTORIES = [
    ROOT_DIR / "data_analysis",
    GRAPHS_DIR,
    LINEAR_SCM_DIR,
    ROOT_DIR / "participants",
]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".DS_Store"}

STEP_OUTPUTS = {
    "eda": [ROOT_DIR / "data_analysis"],
    "dataset": PROCESSED_DATASETS,
    "causal_graph": [CAUSALLEARN_GRAPHS_DIR, CAUSALLEARN_EDGE_NPY_DIR, CAUSALLEARN_EDGE_TXT_DIR],
    "time_series": [TIME_SERIES_OUTPUT_DIR],
    "time_series_pids": [TIME_SERIES_PIDS_OUTPUT_DIR],
    "dowhy": [TESTS_OUTPUT_DIR],
    "linear_scm": [LINEAR_SCM_DIR],
    "scm_validation": [LINEAR_SCM_DIR / "metrics.csv", LINEAR_SCM_DIR / "metrics_interpretation.txt"],
    "personas": [COUNTERFACTUALS_DIR],
    "influence": [INFLUENCES_DIR],
}


def default_run_label():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_generated_artifacts(include_archives=False):
    for directory in GENERATED_DIRECTORIES:
        if not directory.exists():
            continue
        for child in directory.iterdir():
            _remove_path(child)

    if DATASETS_DIR.exists():
        for child in DATASETS_DIR.iterdir():
            if child == RAW_DATASET:
                continue
            _remove_path(child)

    for path in ROOT_DIR.rglob("*"):
        if path.name in CACHE_NAMES:
            _remove_path(path)

    if include_archives and ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)


def _relative_to_root(path: Path):
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve())
    except ValueError:
        return path


def summarize_path(path: Path):
    summary = {
        "path": str(_relative_to_root(path)),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary
    if path.is_file():
        summary["kind"] = "file"
        summary["size_bytes"] = path.stat().st_size
        return summary

    summary["kind"] = "directory"
    files = [p for p in path.rglob("*") if p.is_file()]
    summary["file_count"] = len(files)
    summary["total_size_bytes"] = sum(p.stat().st_size for p in files)
    return summary


def snapshot_step_outputs(step: str, run_label: str):
    output_paths = STEP_OUTPUTS.get(step, [])
    step_archive_dir = ARTIFACTS_DIR / run_label / step
    if step_archive_dir.exists():
        shutil.rmtree(step_archive_dir)
    step_archive_dir.mkdir(parents=True, exist_ok=True)

    for output_path in output_paths:
        if not output_path.exists():
            continue
        target_path = step_archive_dir / _relative_to_root(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            shutil.copytree(output_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(output_path, target_path)
    return step_archive_dir


def write_step_manifest(step: str, script_name: str, run_label: str, python_executable: str):
    output_paths = STEP_OUTPUTS.get(step, [])
    step_archive_dir = ARTIFACTS_DIR / run_label / step
    step_archive_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "step": step,
        "script": script_name,
        "run_label": run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": python_executable,
        "git_commit": get_git_commit(),
        "outputs": [summarize_path(path) for path in output_paths],
    }

    manifest_path = step_archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
