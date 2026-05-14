from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.config import (
    CAUSAL_DISCOVERY_PC_ALPHA,
    CAUSAL_DISCOVERY_PC_CITS,
    CAUSAL_DISCOVERY_PC_UC_PRIORITY,
    CAUSAL_DISCOVERY_PC_UC_RULE,
    CAUSAL_GRAPH_EDGE_PATH,
    COUNTERFACTUALS_DIR,
    INFLUENCE_PID,
    INFLUENCES_DIR,
    LABELLED_TEST_DATASET,
    LINEAR_SCM_DIR,
    PIDS_PERSONAS,
    TESTS_OUTPUT_DIR,
    TEST_COUNTERFACTUAL_PID,
)


class ValidationReport:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str):
        self.errors.append(message)

    def warn(self, message: str):
        self.warnings.append(message)

    def require_file(self, path: Path):
        if not path.exists():
            self.error(f"Missing file: {path}")
            return False
        if not path.is_file():
            self.error(f"Expected a file but found something else: {path}")
            return False
        if path.stat().st_size == 0:
            self.error(f"Empty file: {path}")
            return False
        return True

    def require_dir(self, path: Path):
        if not path.exists():
            self.error(f"Missing directory: {path}")
            return False
        if not path.is_dir():
            self.error(f"Expected a directory but found something else: {path}")
            return False
        return True


def validate_causal_discovery(report: ValidationReport):
    summary_path = CAUSAL_GRAPH_EDGE_PATH.parents[2] / "labelled_execution_times.json"
    if not report.require_file(summary_path):
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error(f"Invalid JSON in {summary_path}: {exc}")
        return

    results = summary.get("results")
    if not isinstance(results, list) or not results:
        report.error(f"No causal-discovery results listed in {summary_path}")
        return

    primary = results[0]
    for key in ("png_path", "npy_path", "txt_path"):
        if key not in primary:
            report.error(f"Missing '{key}' in {summary_path}")
            continue
        report.require_file(Path(primary[key]))

    if Path(primary.get("npy_path", "")) != CAUSAL_GRAPH_EDGE_PATH:
        report.error(
            "The active causal graph path in pipeline.config does not match the primary result "
            f"listed in {summary_path}"
        )

    if primary.get("method") != "pc":
        report.warn(f"Primary causal-discovery method is '{primary.get('method')}', expected 'pc'.")
    if primary.get("cit") not in CAUSAL_DISCOVERY_PC_CITS:
        report.error(
            f"Primary causal-discovery CIT '{primary.get('cit')}' is not in "
            f"CAUSAL_DISCOVERY_PC_CITS={CAUSAL_DISCOVERY_PC_CITS}"
        )
    if primary.get("alpha") != CAUSAL_DISCOVERY_PC_ALPHA:
        report.error(
            f"Primary causal-discovery alpha is {primary.get('alpha')}, "
            f"expected {CAUSAL_DISCOVERY_PC_ALPHA}"
        )
    if primary.get("uc_rule") != CAUSAL_DISCOVERY_PC_UC_RULE:
        report.error(
            f"Primary causal-discovery uc_rule is {primary.get('uc_rule')}, "
            f"expected {CAUSAL_DISCOVERY_PC_UC_RULE}"
        )
    if primary.get("uc_priority") != CAUSAL_DISCOVERY_PC_UC_PRIORITY:
        report.error(
            f"Primary causal-discovery uc_priority is {primary.get('uc_priority')}, "
            f"expected {CAUSAL_DISCOVERY_PC_UC_PRIORITY}"
        )


def validate_dowhy_outputs(report: ValidationReport):
    if not report.require_dir(TESTS_OUTPUT_DIR):
        return

    expected_files = [
        TESTS_OUTPUT_DIR / "test-intervention.pdf",
        TESTS_OUTPUT_DIR / f"test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}.pdf",
        TESTS_OUTPUT_DIR / f"test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}-percentage_difference.pdf",
        TESTS_OUTPUT_DIR / f"test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}-original_units.pdf",
    ]
    for path in expected_files:
        report.require_file(path)


def validate_persona_outputs(report: ValidationReport):
    if not report.require_dir(COUNTERFACTUALS_DIR):
        return

    for pid in PIDS_PERSONAS:
        expected_files = [
            COUNTERFACTUALS_DIR / f"counterfactual-pid={pid}.pdf",
            COUNTERFACTUALS_DIR / f"counterfactual-pid={pid}-percentage_difference.pdf",
            COUNTERFACTUALS_DIR / f"counterfactual-pid={pid}-original_units.pdf",
        ]
        for path in expected_files:
            report.require_file(path)


def validate_influence_outputs(report: ValidationReport):
    if not report.require_dir(INFLUENCES_DIR):
        return

    expected_files = [
        INFLUENCES_DIR / f"counterfactual-pid={INFLUENCE_PID}_only tennis.pdf",
        INFLUENCES_DIR / f"counterfactual-pid={INFLUENCE_PID}_only tennis and set steps.pdf",
        INFLUENCES_DIR / f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + more training.pdf",
        INFLUENCES_DIR / f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + exact more training.pdf",
        INFLUENCES_DIR / "iccs_perc_calories_burned.pdf",
    ]
    for path in expected_files:
        report.require_file(path)


def validate_linear_scm_outputs(report: ValidationReport):
    if not report.require_dir(LINEAR_SCM_DIR):
        return

    expected_files = [
        LINEAR_SCM_DIR / "scm.txt",
        LINEAR_SCM_DIR / "scm_coefficients.json",
        LINEAR_SCM_DIR / "algebraic_equations.txt",
        LINEAR_SCM_DIR / "cf_results.json",
        LINEAR_SCM_DIR / "cf_results_original_units.json",
        LINEAR_SCM_DIR / "linear_cf_results.json",
        LINEAR_SCM_DIR / "linear_cf_results_original_units.json",
        LINEAR_SCM_DIR / "epsilon_results.json",
        LINEAR_SCM_DIR / "epsilon_means_by_pid_and_var.json",
        LINEAR_SCM_DIR / "linear_vs_gcm_mae_original_units.json",
        LINEAR_SCM_DIR / "linear_vs_gcm_mae_original_units.csv",
        LINEAR_SCM_DIR / "metrics.csv",
        LINEAR_SCM_DIR / "metrics_interpretation.txt",
    ]
    for path in expected_files:
        report.require_file(path)

    metrics_path = LINEAR_SCM_DIR / "metrics.csv"
    if report.require_file(metrics_path):
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "target",
                "rmse",
                "mae",
                "r2",
                "rmse_original_units",
                "mae_original_units",
                "n",
            }
            if reader.fieldnames is None:
                report.error(f"No header row found in {metrics_path}")
            elif not required_columns.issubset(reader.fieldnames):
                report.error(
                    f"{metrics_path} is missing required columns: "
                    f"{sorted(required_columns - set(reader.fieldnames))}"
                )
            rows = list(reader)
            if not rows:
                report.error(f"No metric rows found in {metrics_path}")
            for row in rows:
                try:
                    if int(float(row["n"])) <= 0:
                        report.error(f"Non-positive observation count for target '{row['target']}' in {metrics_path}")
                except ValueError:
                    report.error(f"Invalid observation count for target '{row['target']}' in {metrics_path}")

    equations_path = LINEAR_SCM_DIR / "algebraic_equations.txt"
    metrics_targets: set[str] = set()
    equation_targets: set[str] = set()
    if report.require_file(metrics_path):
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metrics_targets = {row["target"] for row in csv.DictReader(handle)}
    if report.require_file(equations_path):
        for line in equations_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue
            equation_targets.add(stripped.split("=", 1)[0].strip())
    if metrics_targets and equation_targets and metrics_targets != equation_targets:
        report.error(
            "Mismatch between targets in metrics.csv and algebraic_equations.txt: "
            f"{sorted(metrics_targets ^ equation_targets)}"
        )

    cf_original_path = LINEAR_SCM_DIR / "cf_results_original_units.json"
    linear_cf_original_path = LINEAR_SCM_DIR / "linear_cf_results_original_units.json"
    if report.require_file(cf_original_path) and report.require_file(linear_cf_original_path):
        try:
            cf_original = json.loads(cf_original_path.read_text(encoding="utf-8"))
            linear_cf_original = json.loads(linear_cf_original_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error(f"Invalid counterfactual JSON: {exc}")
        else:
            if set(cf_original.keys()) != set(linear_cf_original.keys()):
                report.error(
                    "Mismatch between top-level keys in cf_results_original_units.json and "
                    "linear_cf_results_original_units.json"
                )

    mae_path = LINEAR_SCM_DIR / "linear_vs_gcm_mae_original_units.json"
    if report.require_file(mae_path):
        try:
            mae_data = json.loads(mae_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error(f"Invalid JSON in {mae_path}: {exc}")
        else:
            if not isinstance(mae_data, dict) or not mae_data:
                report.error(f"{mae_path} does not contain a non-empty object.")


def validate_dataset_preconditions(report: ValidationReport):
    if not report.require_file(LABELLED_TEST_DATASET):
        return
    if LABELLED_TEST_DATASET.stat().st_size < 1024:
        report.warn(f"{LABELLED_TEST_DATASET} is unexpectedly small.")


def main():
    report = ValidationReport()
    validate_dataset_preconditions(report)
    validate_causal_discovery(report)
    validate_dowhy_outputs(report)
    validate_linear_scm_outputs(report)
    validate_persona_outputs(report)
    validate_influence_outputs(report)

    print("Validation summary")
    print("------------------")
    print(f"errors:   {len(report.errors)}")
    print(f"warnings: {len(report.warnings)}")

    if report.warnings:
        print("\nWarnings")
        print("--------")
        for message in report.warnings:
            print(f"- {message}")

    if report.errors:
        print("\nErrors")
        print("------")
        for message in report.errors:
            print(f"- {message}")
        raise SystemExit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
