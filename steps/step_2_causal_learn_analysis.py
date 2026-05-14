import json
import time

import numpy as np
import pandas as pd
from causallearn.graph.GraphClass import GraphUtils
from causallearn.search.ConstraintBased.PC import pc

from pipeline.config import (
    CAUSALLEARN_EDGE_NPY_DIR,
    CAUSALLEARN_EDGE_TXT_DIR,
    CAUSALLEARN_GRAPHS_DIR,
    CAUSAL_DISCOVERY_DATA_TYPE,
    CAUSAL_DISCOVERY_DROP_COLUMNS,
    CAUSAL_DISCOVERY_PC_ALPHA,
    CAUSAL_DISCOVERY_PC_CITS,
    CAUSAL_DISCOVERY_PC_UC_PRIORITY,
    CAUSAL_DISCOVERY_PC_UC_RULE,
    DATASETS_DIR,
    RANDOM_SEED,
)


np.random.seed(RANDOM_SEED)

CAUSALLEARN_ROOT_DIR = CAUSALLEARN_GRAPHS_DIR.parent


def ensure_output_directories():
    CAUSALLEARN_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_EDGE_NPY_DIR.mkdir(parents=True, exist_ok=True)
    CAUSALLEARN_EDGE_TXT_DIR.mkdir(parents=True, exist_ok=True)


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


def build_pc_png_path(data_type: str, cit: str):
    if data_type == "encoded":
        base_dir = CAUSALLEARN_GRAPHS_DIR / "onehot" / "PC-onehot"
        prefix = "encoding"
    else:
        base_dir = CAUSALLEARN_GRAPHS_DIR / "labelling" / "PC-labelling"
        prefix = "labelling"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{prefix}_causal_graph_causal-learn_pc_{cit}.png"


def build_pc_edge_stem(data_type: str, cit: str):
    prefix = "encoding" if data_type == "encoded" else "labelling"
    return f"{prefix}_causal_graph_causal-learn_pc_{cit}"


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


def run_pc(fit_data: np.ndarray, var_names, data_type: str):
    execution_summary = []
    for cit in CAUSAL_DISCOVERY_PC_CITS:
        print(
            f"---> PC, alpha={CAUSAL_DISCOVERY_PC_ALPHA}, cit={cit}, "
            f"uc_rule={CAUSAL_DISCOVERY_PC_UC_RULE}, uc_priority={CAUSAL_DISCOVERY_PC_UC_PRIORITY}"
        )
        start_time = time.time()
        cg_pc = pc(
            fit_data,
            alpha=CAUSAL_DISCOVERY_PC_ALPHA,
            indep_test=cit,
            uc_rule=CAUSAL_DISCOVERY_PC_UC_RULE,
            uc_priority=CAUSAL_DISCOVERY_PC_UC_PRIORITY,
        )
        elapsed_seconds = time.time() - start_time
        print(f"PC with cit={cit}: {elapsed_seconds:.2f} seconds\n")

        png_path = build_pc_png_path(data_type, cit)
        GraphUtils.to_pydot(cg_pc.G, labels=var_names).write_png(str(png_path))

        named_edges = named_edges_from_index_pairs(cg_pc.find_fully_directed(), var_names)
        npy_path, txt_path = save_named_edges(named_edges, build_pc_edge_stem(data_type, cit))
        execution_summary.append(
            {
                "method": "pc",
                "cit": cit,
                "alpha": CAUSAL_DISCOVERY_PC_ALPHA,
                "uc_rule": CAUSAL_DISCOVERY_PC_UC_RULE,
                "uc_priority": CAUSAL_DISCOVERY_PC_UC_PRIORITY,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "directed_edge_count": int(len(named_edges)),
                "png_path": str(png_path),
                "npy_path": str(npy_path),
                "txt_path": str(txt_path),
            }
        )
    return execution_summary


ensure_output_directories()

data_type = CAUSAL_DISCOVERY_DATA_TYPE
dataset_path, fit_data_df, var_names = load_training_data(data_type)
fit_data = fit_data_df.to_numpy()

print(f"Loaded {dataset_path.name} with {len(fit_data_df)} rows and {len(var_names)} causal-discovery variables.\n")

execution_summary = run_pc(fit_data, var_names, data_type)

execution_times_path = CAUSALLEARN_ROOT_DIR / f"{data_type}_execution_times.json"
execution_times_path.write_text(
    json.dumps(
        {
            "data_type": data_type,
            "dataset_path": str(dataset_path),
            "row_count": int(len(fit_data_df)),
            "column_count": int(len(var_names)),
            "random_seed": RANDOM_SEED,
            "results": execution_summary,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"Wrote execution summary to {execution_times_path}")
