import json
import time
import warnings

import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

from pipeline.config import (
    LABELLED_TEST_DATASET,
    MONTHS,
    SINGLE_PID_TIME_SERIES_CI_TEST,
    SINGLE_PID_TIME_SERIES_DROP_COLUMNS,
    SINGLE_PID_TIME_SERIES_PCS,
    SINGLE_PID_TIME_SERIES_TAUS,
    TIME_SERIES_PIDS_OUTPUT_DIR,
)


pd.options.mode.chained_assignment = None
warnings.filterwarnings(action="ignore", category=UserWarning)

EXPECTED_MONTH_COUNT = len(MONTHS)
EXECUTION_TIMES_PATH = TIME_SERIES_PIDS_OUTPUT_DIR / "labelled_execution_times.json"


def get_ci_test():
    if SINGLE_PID_TIME_SERIES_CI_TEST == "ParCorr":
        return ParCorr()
    raise ValueError(f"Unsupported single-pid time-series CI test: {SINGLE_PID_TIME_SERIES_CI_TEST}")


def build_output_path(pid: int, tau: int, pc_alpha: float, ci_test_name: str):
    safe_ci_test_name = ci_test_name.replace("()", "").replace("/", "_")
    return TIME_SERIES_PIDS_OUTPUT_DIR / (
        f"pid={pid}_TimeSeriesGraph_LPCMCI_tau={tau}_pc={pc_alpha}_cit={safe_ci_test_name}.pdf"
    )


def load_panel_data():
    fit_data = pd.read_csv(LABELLED_TEST_DATASET).copy()
    fit_data["date"] = pd.to_numeric(fit_data["date"], errors="raise")
    participant_ids = sorted(int(pid) for pid in fit_data["participant_id"].unique().tolist())
    return fit_data, participant_ids


def validate_participant_series(fit_data_pid: pd.DataFrame, pid: int):
    month_values = sorted(int(value) for value in fit_data_pid["date"].tolist())
    if len(fit_data_pid) != EXPECTED_MONTH_COUNT:
        raise ValueError(f"pid={pid}: expected {EXPECTED_MONTH_COUNT} rows, found {len(fit_data_pid)}")
    if month_values != list(range(1, EXPECTED_MONTH_COUNT + 1)):
        raise ValueError(
            f"pid={pid}: expected month labels 1..{EXPECTED_MONTH_COUNT}, found {month_values}"
        )


def build_dataframe_for_pid(fit_data: pd.DataFrame, pid: int):
    fit_data_pid = (
        fit_data.loc[fit_data["participant_id"] == pid]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )
    validate_participant_series(fit_data_pid, pid)

    modifiable_fit_data_pid = fit_data_pid.drop(columns=SINGLE_PID_TIME_SERIES_DROP_COLUMNS)
    varying_columns = [
        column_name
        for column_name in modifiable_fit_data_pid.columns
        if modifiable_fit_data_pid[column_name].nunique(dropna=False) > 1
    ]
    modifiable_fit_data_pid = modifiable_fit_data_pid.loc[:, varying_columns].copy()
    if len(varying_columns) < 2:
        return None, None, varying_columns

    var_names = modifiable_fit_data_pid.columns.tolist()
    data_array_pid = modifiable_fit_data_pid.to_numpy()

    dataframe = pp.DataFrame(data_array_pid, var_names=var_names)
    return dataframe, var_names, varying_columns


def record_execution(execution_summary, pid: int, tau: int, pc_alpha: float, ci_test_name: str, elapsed_seconds: float):
    execution_summary.append(
        {
            "participant_id": pid,
            "method": "LPCMCI",
            "tau_max": tau,
            "pc_alpha": pc_alpha,
            "ci_test": ci_test_name,
            "elapsed_seconds": round(elapsed_seconds, 4),
        }
    )


def record_skip(execution_summary, pid: int, reason: str, retained_columns):
    execution_summary.append(
        {
            "participant_id": pid,
            "method": "LPCMCI",
            "status": "skipped",
            "reason": reason,
            "retained_columns": retained_columns,
        }
    )


TIME_SERIES_PIDS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fit_data, participant_ids = load_panel_data()
execution_summary = []
ci_test_name = SINGLE_PID_TIME_SERIES_CI_TEST

for pid in participant_ids:
    dataframe, var_names, varying_columns = build_dataframe_for_pid(fit_data, pid)
    if dataframe is None:
        print(f"Skipping pid={pid}: fewer than two varying columns remain after filtering.\n")
        record_skip(
            execution_summary,
            pid,
            "fewer than two varying columns remain after filtering constant-within-pid features",
            varying_columns,
        )
        continue

    for tau in SINGLE_PID_TIME_SERIES_TAUS:
        for pc_alpha in SINGLE_PID_TIME_SERIES_PCS:
            print(f"Now executing LPCMCI for pid={pid}, tau={tau}, pc={pc_alpha}, cit={ci_test_name}...\n")
            ci_test = get_ci_test()
            start_time = time.time()
            lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
            results = lpcmci.run_lpcmci(pc_alpha=pc_alpha, tau_max=tau)
            elapsed_seconds = time.time() - start_time
            print(f"LPCMCI completed for pid={pid}, tau={tau}, pc={pc_alpha}, cit={ci_test_name}\n")

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=results["val_matrix"],
                graph=results["graph"],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title(f"Causal discovery - LPCMCI for pid={pid} with tau={tau}, pc={pc_alpha}, cit={ci_test_name}")
            plt.savefig(build_output_path(pid, tau, pc_alpha, ci_test_name))
            plt.close()

            record_execution(execution_summary, pid, tau, pc_alpha, ci_test_name, elapsed_seconds)

execution_payload = {
    "dataset": str(LABELLED_TEST_DATASET),
    "participant_count": len(participant_ids),
    "results": execution_summary,
}

with EXECUTION_TIMES_PATH.open("w", encoding="utf-8") as file:
    json.dump(execution_payload, file, indent=2)
