import argparse
import os
import subprocess
import sys
from pathlib import Path

from pipeline.artifacts import STEP_OUTPUTS, clean_generated_artifacts, default_run_label, snapshot_step_outputs, write_step_manifest


ROOT = Path(__file__).resolve().parent
STEP_TO_MODULE = {
    "eda": "steps.step_0_data_analysis",
    "dataset": "steps.step_1_dataset_management",
    "causal_graph": "steps.step_2_causal_learn_analysis",
    "time_series": "steps.step_2a_time_series_analysis",
    "time_series_pids": "steps.step_2b_time_series_analysis_single_pids",
    "dowhy": "steps.step_3a_dowhy_analysis",
    "linear_scm": "steps.step_3b_linear_scm",
    "scm_validation": "steps.step_3c_scm_validation",
    "personas": "steps.step_4a_personas_analysis",
    "influence": "steps.step_4b_influence_analysis",
    "validate": "checks.validate_pipeline_outputs",
}
PIPELINES = {
    "full": [
        "dataset",
        "causal_graph",
        "dowhy",
        "linear_scm",
        "scm_validation",
        "personas",
        "influence",
    ],
    "optional": [
        "eda",
        "time_series",
        "time_series_pids",
    ],
}
SPECIAL_COMMANDS = {"clean"}


def list_steps():
    print("Available steps:", flush=True)
    for step, module_name in STEP_TO_MODULE.items():
        print(f"{step:16} {module_name}", flush=True)
    print("\nPipeline groups:", flush=True)
    for pipeline, steps in PIPELINES.items():
        print(f"{pipeline:16} {' -> '.join(steps)}", flush=True)
    print("\nSpecial commands:", flush=True)
    print("clean            Remove generated artifacts while keeping datasets/health_fitness_dataset.csv and ppt/", flush=True)


def resolve_steps(requested_steps):
    resolved_steps = []
    for step in requested_steps:
        if step in SPECIAL_COMMANDS:
            resolved_steps.append(step)
        if step in PIPELINES:
            resolved_steps.extend(PIPELINES[step])
        elif step in STEP_TO_MODULE:
            resolved_steps.append(step)
        elif step in SPECIAL_COMMANDS:
            continue
        else:
            raise ValueError(f"Unknown step '{step}'. Use --list to see valid options.")
    return resolved_steps


def run_steps(steps, python_executable, run_label, snapshot_outputs):
    for step in steps:
        if step == "clean":
            continue
        module_name = STEP_TO_MODULE[step]
        print(f"\n=== Running {step}: {module_name} ===\n", flush=True)
        env = os.environ.copy()
        env["HEALTH_CEFRIEL_RUN_LABEL"] = run_label
        subprocess.run([python_executable, "-m", module_name], cwd=ROOT, check=True, env=env)
        if snapshot_outputs and step in STEP_OUTPUTS:
            snapshot_step_outputs(step, run_label)
            write_step_manifest(step, f"{module_name.replace('.', '/')}.py", run_label, python_executable)


def build_parser():
    parser = argparse.ArgumentParser(description="Run Health_Cefriel pipeline steps.")
    parser.add_argument(
        "steps",
        nargs="*",
        default=["full"],
        help="Step names or pipeline groups. Defaults to 'full'.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when running the scripts.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available steps and exit.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Label for archived step outputs under artifacts/. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Run steps without archiving outputs under artifacts/<run-label>/.",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="When used with clean, also remove archived runs under artifacts/.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    steps = resolve_steps(args.steps)
    if "clean" in steps:
        print("\n=== Cleaning generated artifacts ===\n", flush=True)
        clean_generated_artifacts(include_archives=args.include_artifacts)

    runnable_steps = [step for step in steps if step != "clean"]
    if not runnable_steps:
        return

    run_label = args.run_label or default_run_label()
    run_steps(runnable_steps, args.python, run_label=run_label, snapshot_outputs=not args.no_snapshot)


if __name__ == "__main__":
    main()
