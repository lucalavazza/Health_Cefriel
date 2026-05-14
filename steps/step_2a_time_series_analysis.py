import json
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dowhy.utils.plotting import plot
from dowhy.utils.timeseries import create_graph_from_networkx_array
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.pairwise_CI import PairwiseMultCI
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI

from pipeline.config import (
    LABELLED_TEST_DATASET,
    MONTHS,
    TIME_SERIES_CI_TEST,
    TIME_SERIES_DROP_COLUMNS,
    TIME_SERIES_LPCMCI_PCS,
    TIME_SERIES_LPCMCI_TAUS,
    TIME_SERIES_OUTPUT_DIR,
    TIME_SERIES_PCMCI_PCS,
    TIME_SERIES_PCMCIPLUS_PCS,
    TIME_SERIES_PCMCIPLUS_TAUS,
    TIME_SERIES_PCMCI_TAUS,
)


pd.options.mode.chained_assignment = None

EXPECTED_MONTH_COUNT = len(MONTHS)


def get_ci_test():
    if TIME_SERIES_CI_TEST == "PairwiseMultCI":
        return PairwiseMultCI()
    raise ValueError(f"Unsupported time-series CI test: {TIME_SERIES_CI_TEST}")


def build_output_path(prefix: str, tau: int, pc_alpha: float, ci_test_name: str):
    safe_ci_test_name = ci_test_name.replace("()", "").replace("/", "_")
    return TIME_SERIES_OUTPUT_DIR / f"{prefix}_tau={tau}_pc={pc_alpha}_cit={safe_ci_test_name}.pdf"


def normalize_graph_for_dowhy(graph):
    normalized_graph = np.array(graph, copy=True)
    normalized_graph[normalized_graph == "x-x"] = "<->"
    normalized_graph[normalized_graph == "o-o"] = "<->"
    return normalized_graph


def load_panel_data():
    fit_data = pd.read_csv(LABELLED_TEST_DATASET)
    fit_data = fit_data.copy()
    fit_data["date"] = pd.to_numeric(fit_data["date"], errors="raise")
    participant_ids = sorted(int(pid) for pid in fit_data["participant_id"].unique().tolist())
    return fit_data, participant_ids


def validate_panel_structure(fit_data: pd.DataFrame, participant_ids):
    problems = []
    for pid in participant_ids:
        participant_df = fit_data.loc[fit_data["participant_id"] == pid]
        month_values = sorted(int(value) for value in participant_df["date"].tolist())
        if len(participant_df) != EXPECTED_MONTH_COUNT:
            problems.append(f"pid={pid}: expected {EXPECTED_MONTH_COUNT} rows, found {len(participant_df)}")
        elif month_values != list(range(1, EXPECTED_MONTH_COUNT + 1)):
            problems.append(f"pid={pid}: expected month labels 1..{EXPECTED_MONTH_COUNT}, found {month_values}")

    if problems:
        preview = "; ".join(problems[:10])
        raise ValueError(
            "Time-series preprocessing expects one row per participant-month in the labelled testing dataset. "
            f"Found inconsistencies: {preview}"
        )


def build_multiple_time_series_dataframe(fit_data: pd.DataFrame, participant_ids):
    modifiable_fit_data = fit_data.drop(columns=TIME_SERIES_DROP_COLUMNS)
    var_names = modifiable_fit_data.columns

    data_dict = {}
    for pid in participant_ids:
        fit_data_id = (
            fit_data.loc[fit_data["participant_id"] == pid]
            .sort_values("date")
            .drop(columns=TIME_SERIES_DROP_COLUMNS)
            .reset_index(drop=True)
            .copy()
        )
        data_dict[pid] = fit_data_id.to_numpy()

    dataframe = pp.DataFrame(data_dict, analysis_mode="multiple", var_names=var_names)
    return dataframe, var_names


def record_execution(execution_summary, method_name: str, tau: int, pc_alpha: float, ci_test_name: str, elapsed_seconds: float):
    execution_summary.append(
        {
            "method": method_name,
            "tau_max": tau,
            "pc_alpha": pc_alpha,
            "ci_test": ci_test_name,
            "elapsed_seconds": round(elapsed_seconds, 4),
        }
    )


def run_lpcmci(dataframe, var_names, execution_summary):
    ci_test = get_ci_test()
    ci_test_name = TIME_SERIES_CI_TEST

    for tau in TIME_SERIES_LPCMCI_TAUS:
        for pc_alpha in TIME_SERIES_LPCMCI_PCS:
            print(f"Now executing LPCMCI for tau={tau} pc={pc_alpha} cit={ci_test_name}...\n")
            start_time = time.time()
            lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
            lpcmci_results = lpcmci.run_lpcmci(pc_alpha=pc_alpha, tau_max=tau)
            elapsed_seconds = time.time() - start_time
            print(f"LPCMCI completed for tau={tau} pc={pc_alpha} cit={ci_test_name}\n")

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=lpcmci_results["val_matrix"],
                graph=lpcmci_results["graph"],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title(f"Causal discovery - LPCMCI with tau={tau} pc={pc_alpha} cit={ci_test_name}")
            plt.savefig(build_output_path("TimeSeriesGraph_LPCMCI", tau, pc_alpha, ci_test_name))
            plt.close()

            normalized_graph = normalize_graph_for_dowhy(lpcmci_results["graph"])
            lpcmci_graph = create_graph_from_networkx_array(normalized_graph, var_names)
            plot(
                causal_graph=lpcmci_graph,
                filename=str(build_output_path("TimeSeriesGraph_DoWhy_LPCMCI", tau, pc_alpha, ci_test_name)),
                display_plot=False,
                figure_size=(18, 12),
            )
            record_execution(execution_summary, "LPCMCI", tau, pc_alpha, ci_test_name, elapsed_seconds)


def run_pcmci(dataframe, var_names, execution_summary):
    ci_test = get_ci_test()
    ci_test_name = TIME_SERIES_CI_TEST

    for tau in TIME_SERIES_PCMCI_TAUS:
        for pc_alpha in TIME_SERIES_PCMCI_PCS:
            print(f"Now executing PCMCI for tau={tau} pc={pc_alpha} cit={ci_test_name}...\n")
            start_time = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
            pcmci_results = pcmci.run_pcmci(pc_alpha=pc_alpha, tau_max=tau)
            elapsed_seconds = time.time() - start_time
            print(f"PCMCI completed for tau={tau} pc={pc_alpha} cit={ci_test_name}\n")

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=pcmci_results["val_matrix"],
                graph=pcmci_results["graph"],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title(f"Causal discovery - PCMCI with tau={tau} pc={pc_alpha} cit={ci_test_name}")
            plt.savefig(build_output_path("TimeSeriesGraph_PCMCI", tau, pc_alpha, ci_test_name))
            plt.close()

            normalized_graph = normalize_graph_for_dowhy(pcmci_results["graph"])
            pcmci_graph = create_graph_from_networkx_array(normalized_graph, var_names)
            plot(
                causal_graph=pcmci_graph,
                filename=str(build_output_path("TimeSeriesGraph_DoWhy_PCMCI", tau, pc_alpha, ci_test_name)),
                display_plot=False,
                figure_size=(18, 12),
            )
            record_execution(execution_summary, "PCMCI", tau, pc_alpha, ci_test_name, elapsed_seconds)


def run_pcmciplus(dataframe, var_names, execution_summary):
    ci_test = get_ci_test()
    ci_test_name = TIME_SERIES_CI_TEST

    for tau in TIME_SERIES_PCMCIPLUS_TAUS:
        for pc_alpha in TIME_SERIES_PCMCIPLUS_PCS:
            print(f"Now executing PCMCI+ for tau={tau} pc={pc_alpha} cit={ci_test_name}...\n")
            start_time = time.time()
            pcmciplus = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
            pcmciplus_results = pcmciplus.run_pcmciplus(pc_alpha=pc_alpha, tau_max=tau)
            elapsed_seconds = time.time() - start_time
            print(f"PCMCI+ completed for tau={tau} pc={pc_alpha} cit={ci_test_name}\n")

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=pcmciplus_results["val_matrix"],
                graph=pcmciplus_results["graph"],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title(f"Causal discovery - PCMCI+ with tau={tau} pc={pc_alpha} cit={ci_test_name}")
            plt.savefig(build_output_path("TimeSeriesGraph_PCMCIplus", tau, pc_alpha, ci_test_name))
            plt.close()

            normalized_graph = normalize_graph_for_dowhy(pcmciplus_results["graph"])
            pcmciplus_graph = create_graph_from_networkx_array(normalized_graph, var_names)
            plot(
                causal_graph=pcmciplus_graph,
                filename=str(build_output_path("TimeSeriesGraph_DoWhy_PCMCIplus", tau, pc_alpha, ci_test_name)),
                display_plot=False,
                figure_size=(18, 12),
            )
            record_execution(execution_summary, "PCMCIplus", tau, pc_alpha, ci_test_name, elapsed_seconds)


TIME_SERIES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fit_data, participant_ids = load_panel_data()
validate_panel_structure(fit_data, participant_ids)
dataframe, var_names = build_multiple_time_series_dataframe(fit_data, participant_ids)

execution_summary = []

print("Starting Causal Discovery with LPCMCI, PCMCI, and PCMCI+\n")
run_lpcmci(dataframe, var_names, execution_summary)
run_pcmci(dataframe, var_names, execution_summary)
run_pcmciplus(dataframe, var_names, execution_summary)
print("Causal Discovery with LPCMCI, PCMCI, and PCMCI+ completed\n")

with (TIME_SERIES_OUTPUT_DIR / "labelled_execution_times.json").open("w", encoding="utf-8") as handle:
    json.dump(
        {
            "dataset_path": str(LABELLED_TEST_DATASET),
            "participant_count": len(participant_ids),
            "variables": list(var_names),
            "results": execution_summary,
        },
        handle,
        indent=2,
    )
