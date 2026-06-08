import csv
import json
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from causallearn.graph.GraphClass import GraphUtils
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges

from pipeline.config import (
    CAUSALLEARN_COMPARISON_DIR,
    CAUSALLEARN_EDGE_NPY_DIR,
    CAUSALLEARN_EDGE_TXT_DIR,
    CAUSALLEARN_GRAPHS_DIR,
    CAUSALLEARN_STABILITY_DIR,
    CAUSAL_DISCOVERY_ALT_METHOD,
    CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES,
    CAUSAL_DISCOVERY_DATA_TYPE,
    CAUSAL_DISCOVERY_DROP_COLUMNS,
    CAUSAL_DISCOVERY_PATHWAY_TARGETS,
    CAUSAL_DISCOVERY_PC_ALPHA,
    CAUSAL_DISCOVERY_PC_ALPHAS,
    CAUSAL_DISCOVERY_PC_CITS,
    CAUSAL_DISCOVERY_PC_UC_PRIORITY,
    CAUSAL_DISCOVERY_PC_UC_RULE,
    DATASETS_DIR,
    RANDOM_SEED,
)
from pipeline.visualization import configure_matplotlib, display_label, graph_display_labels


np.random.seed(RANDOM_SEED)
configure_matplotlib()

CAUSALLEARN_ROOT_DIR = CAUSALLEARN_GRAPHS_DIR.parent


def ensure_output_directories():
    CAUSALLEARN_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_EDGE_NPY_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_EDGE_TXT_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data(data_type: str):
    dataset_path = DATASETS_DIR / f"{data_type}_regularised_averaged_health_fitness_dataset_training.csv"
    fit_data = pd.read_csv(dataset_path)
    drop_cols = CAUSAL_DISCOVERY_DROP_COLUMNS[data_type]
    missing_drop_cols = [col for col in drop_cols if col not in fit_data.columns]
    if missing_drop_cols:
        raise ValueError(f"Missing expected columns for causal discovery: {missing_drop_cols}")

    fit_data = fit_data.drop(columns=drop_cols)
    var_names = fit_data.columns
    return dataset_path, fit_data, var_names


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "")


def build_pc_graph_paths(data_type: str, cit: str, alpha: float):
    if data_type == "encoded":
        base_dir = CAUSALLEARN_GRAPHS_DIR / "onehot" / "PC-onehot"
        prefix = "encoding"
    else:
        base_dir = CAUSALLEARN_GRAPHS_DIR / "labelling" / "PC-labelling"
        prefix = "labelling"
    base_dir.mkdir(parents=True, exist_ok=True)

    if np.isclose(alpha, CAUSAL_DISCOVERY_PC_ALPHA):
        stem = f"{prefix}_causal_graph_causal-learn_pc_{cit}"
    else:
        stem = f"{prefix}_causal_graph_causal-learn_pc_{cit}_alpha={alpha:.2f}"
    return stem, base_dir / f"{stem}.png", base_dir / f"{stem}.pdf"


def build_ges_graph_paths(data_type: str):
    if data_type == "encoded":
        base_dir = CAUSALLEARN_GRAPHS_DIR / "onehot" / "GES-onehot"
        prefix = "encoding"
    else:
        base_dir = CAUSALLEARN_GRAPHS_DIR / "labelling" / "GES-labelling"
        prefix = "labelling"
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_causal_graph_causal-learn_ges"
    return stem, base_dir / f"{stem}.png", base_dir / f"{stem}.pdf"


def save_graph(graph, var_names, png_path, pdf_path):
    pydot_graph = GraphUtils.to_pydot(graph, labels=graph_display_labels(var_names))
    pydot_graph.write_png(str(png_path))
    pydot_graph.write_pdf(str(pdf_path))


def save_named_edges(named_edges: np.ndarray, stem: str):
    npy_path = CAUSALLEARN_EDGE_NPY_DIR / f"{stem}.npy"
    txt_path = CAUSALLEARN_EDGE_TXT_DIR / f"{stem}.txt"

    np.save(npy_path, named_edges)
    with txt_path.open("w", encoding="utf-8") as handle:
        for edge in named_edges:
            handle.write(f"{edge[0]} -> {edge[1]}\n")

    return npy_path, txt_path


def named_edges_from_index_pairs(index_edges, var_names):
    named_edges = []
    for source_index, target_index in index_edges:
        named_edges.append([str(var_names[source_index]), str(var_names[target_index])])
    if named_edges:
        return np.array(named_edges, dtype=object).reshape(-1, 2)
    return np.empty((0, 2), dtype=object)


def edge_tuples(named_edges: np.ndarray):
    return {(str(source), str(target)) for source, target in named_edges.tolist()}


def extract_directed_index_edges_from_graph(graph):
    if hasattr(graph, "find_fully_directed"):
        try:
            return list(graph.find_fully_directed())
        except TypeError:
            pass

    matrix = np.asarray(graph.graph)
    directed_edges = []
    for source_index in range(matrix.shape[0]):
        for target_index in range(matrix.shape[1]):
            if source_index == target_index:
                continue
            if matrix[source_index, target_index] == -1 and matrix[target_index, source_index] == 1:
                directed_edges.append((source_index, target_index))
    return sorted(set(directed_edges))


def run_pc_alpha_sweep(fit_data: np.ndarray, var_names, data_type: str):
    execution_summary = []
    retained_named_edges = None

    for cit in CAUSAL_DISCOVERY_PC_CITS:
        for alpha in CAUSAL_DISCOVERY_PC_ALPHAS:
            print(
                f"---> PC, alpha={alpha}, cit={cit}, "
                f"uc_rule={CAUSAL_DISCOVERY_PC_UC_RULE}, uc_priority={CAUSAL_DISCOVERY_PC_UC_PRIORITY}"
            )
            start_time = time.time()
            cg_pc = pc(
                fit_data,
                alpha=alpha,
                indep_test=cit,
                uc_rule=CAUSAL_DISCOVERY_PC_UC_RULE,
                uc_priority=CAUSAL_DISCOVERY_PC_UC_PRIORITY,
            )
            elapsed_seconds = time.time() - start_time
            print(f"PC with alpha={alpha}, cit={cit}: {elapsed_seconds:.2f} seconds\n")

            stem, png_path, pdf_path = build_pc_graph_paths(data_type, cit, alpha)
            save_graph(cg_pc.G, var_names, png_path, pdf_path)

            named_edges = named_edges_from_index_pairs(cg_pc.find_fully_directed(), var_names)
            npy_path, txt_path = save_named_edges(named_edges, stem)
            if np.isclose(alpha, CAUSAL_DISCOVERY_PC_ALPHA):
                retained_named_edges = named_edges

            execution_summary.append(
                {
                    "method": "pc",
                    "cit": cit,
                    "alpha": alpha,
                    "uc_rule": CAUSAL_DISCOVERY_PC_UC_RULE,
                    "uc_priority": CAUSAL_DISCOVERY_PC_UC_PRIORITY,
                    "elapsed_seconds": round(elapsed_seconds, 4),
                    "directed_edge_count": int(len(named_edges)),
                    "png_path": str(png_path),
                    "pdf_path": str(pdf_path),
                    "npy_path": str(npy_path),
                    "txt_path": str(txt_path),
                }
            )

    if retained_named_edges is None:
        raise RuntimeError("Retained PC graph at alpha=0.05 was not generated.")
    return execution_summary, retained_named_edges


def write_pc_stability_summary(pc_results):
    retained_results = [result for result in pc_results if result["method"] == "pc"]
    edge_sets_by_alpha = {}
    for result in retained_results:
        edge_array = np.load(result["npy_path"], allow_pickle=True)
        edge_sets_by_alpha[result["alpha"]] = edge_tuples(edge_array)

    all_edges = sorted(set().union(*edge_sets_by_alpha.values()))
    rows = []
    for source, target in all_edges:
        row = {
            "source": source,
            "target": target,
            "edge": f"{source} -> {target}",
            "edge_display": f"{display_label(source)} -> {display_label(target)}",
            "persistence_count": 0,
        }
        for alpha in CAUSAL_DISCOVERY_PC_ALPHAS:
            present = (source, target) in edge_sets_by_alpha.get(alpha, set())
            row[f"alpha_{alpha:.2f}"] = int(present)
            row["persistence_count"] += int(present)
        rows.append(row)

    csv_path = CAUSALLEARN_STABILITY_DIR / "pc_alpha_sensitivity_summary.csv"
    json_path = CAUSALLEARN_STABILITY_DIR / "pc_alpha_sensitivity_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "edge", "edge_display", "persistence_count"]
            + [f"alpha_{alpha:.2f}" for alpha in CAUSAL_DISCOVERY_PC_ALPHAS],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows, csv_path, json_path


def run_bootstrap_edge_frequencies(fit_data: np.ndarray, var_names):
    rng = np.random.default_rng(RANDOM_SEED)
    edge_counter = Counter()
    n_rows = fit_data.shape[0]

    print(f"Running {CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES} bootstrap PC replicates...")
    for replicate in range(CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES):
        sampled_indices = rng.integers(0, n_rows, size=n_rows)
        sampled_data = fit_data[sampled_indices]
        cg_pc = pc(
            sampled_data,
            alpha=CAUSAL_DISCOVERY_PC_ALPHA,
            indep_test=CAUSAL_DISCOVERY_PC_CITS[0],
            uc_rule=CAUSAL_DISCOVERY_PC_UC_RULE,
            uc_priority=CAUSAL_DISCOVERY_PC_UC_PRIORITY,
        )
        named_edges = named_edges_from_index_pairs(cg_pc.find_fully_directed(), var_names)
        for edge in edge_tuples(named_edges):
            edge_counter[edge] += 1
        if (replicate + 1) % 10 == 0:
            print(f"Completed bootstrap replicate {replicate + 1}/{CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES}")

    rows = []
    for (source, target), count in edge_counter.most_common():
        rows.append(
            {
                "source": source,
                "target": target,
                "edge": f"{source} -> {target}",
                "edge_display": f"{display_label(source)} -> {display_label(target)}",
                "count": count,
                "frequency": round(count / CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES, 4),
            }
        )

    csv_path = CAUSALLEARN_STABILITY_DIR / "pc_bootstrap_edge_frequencies.csv"
    json_path = CAUSALLEARN_STABILITY_DIR / "pc_bootstrap_edge_frequencies.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "edge", "edge_display", "count", "frequency"],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    plot_path = CAUSALLEARN_STABILITY_DIR / "pc_bootstrap_edge_frequencies_top15.pdf"
    plt.figure(figsize=(11, 6))
    top_rows = rows[:15]
    plt.barh(
        [row["edge_display"] for row in reversed(top_rows)],
        [row["frequency"] for row in reversed(top_rows)],
        color="#5072A7",
    )
    plt.xlabel("Bootstrap edge frequency")
    plt.title("PC edge stability under bootstrap resampling")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    return rows, csv_path, json_path, plot_path


def run_ges_discovery(fit_data: np.ndarray, var_names, data_type: str):
    print(f"---> {CAUSAL_DISCOVERY_ALT_METHOD.upper()} discovery on the monthly training data")
    start_time = time.time()
    ges_result = ges(fit_data)
    elapsed_seconds = time.time() - start_time
    graph = ges_result["G"]

    stem, png_path, pdf_path = build_ges_graph_paths(data_type)
    save_graph(graph, var_names, png_path, pdf_path)
    named_edges = named_edges_from_index_pairs(extract_directed_index_edges_from_graph(graph), var_names)
    npy_path, txt_path = save_named_edges(named_edges, stem)

    summary = {
        "method": "ges",
        "elapsed_seconds": round(elapsed_seconds, 4),
        "directed_edge_count": int(len(named_edges)),
        "score": float(ges_result["score"]) if "score" in ges_result else None,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "npy_path": str(npy_path),
        "txt_path": str(txt_path),
    }
    return summary, named_edges


def write_method_comparison(retained_pc_edges, ges_edges):
    retained_pc_edge_set = edge_tuples(retained_pc_edges)
    ges_edge_set = edge_tuples(ges_edges)

    rows = []
    for source, target in CAUSAL_DISCOVERY_PATHWAY_TARGETS:
        rows.append(
            {
                "source": source,
                "target": target,
                "edge": f"{source} -> {target}",
                "edge_display": f"{display_label(source)} -> {display_label(target)}",
                "present_in_pc": int((source, target) in retained_pc_edge_set),
                "present_in_ges": int((source, target) in ges_edge_set),
            }
        )

    csv_path = CAUSALLEARN_COMPARISON_DIR / "pc_vs_ges_pathways.csv"
    json_path = CAUSALLEARN_COMPARISON_DIR / "pc_vs_ges_pathways.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "edge", "edge_display", "present_in_pc", "present_in_ges"],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows, csv_path, json_path


ensure_output_directories()

data_type = CAUSAL_DISCOVERY_DATA_TYPE
dataset_path, fit_data_df, var_names = load_training_data(data_type)
fit_data = fit_data_df.to_numpy()

print(f"Loaded {dataset_path.name} with {len(fit_data_df)} rows and {len(var_names)} causal-discovery variables.\n")

pc_execution_summary, retained_named_edges = run_pc_alpha_sweep(fit_data, var_names, data_type)
pc_stability_rows, pc_stability_csv, pc_stability_json = write_pc_stability_summary(pc_execution_summary)
bootstrap_rows, bootstrap_csv, bootstrap_json, bootstrap_plot = run_bootstrap_edge_frequencies(fit_data, var_names)
ges_summary, ges_named_edges = run_ges_discovery(fit_data, var_names, data_type)
comparison_rows, comparison_csv, comparison_json = write_method_comparison(retained_named_edges, ges_named_edges)

execution_times_path = CAUSALLEARN_ROOT_DIR / f"{data_type}_execution_times.json"
execution_times_path.write_text(
    json.dumps(
        {
            "data_type": data_type,
            "dataset_path": str(dataset_path),
            "row_count": int(len(fit_data_df)),
            "column_count": int(len(var_names)),
            "random_seed": RANDOM_SEED,
            "results": pc_execution_summary + [ges_summary],
            "pc_alpha_sensitivity_summary_csv": str(pc_stability_csv),
            "pc_alpha_sensitivity_summary_json": str(pc_stability_json),
            "pc_bootstrap_edge_frequencies_csv": str(bootstrap_csv),
            "pc_bootstrap_edge_frequencies_json": str(bootstrap_json),
            "pc_bootstrap_edge_plot": str(bootstrap_plot),
            "pc_vs_ges_pathways_csv": str(comparison_csv),
            "pc_vs_ges_pathways_json": str(comparison_json),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Wrote execution summary to {execution_times_path}")
