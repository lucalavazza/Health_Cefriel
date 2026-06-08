from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATASETS_DIR = ROOT_DIR / "datasets"
GRAPHS_DIR = ROOT_DIR / "graphs"
LINEAR_SCM_DIR = ROOT_DIR / "linear_scm"
PPT_DIR = ROOT_DIR / "ppt"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RAW_DATASET = DATASETS_DIR / "health_fitness_dataset.csv"
PROCESSED_DATASET_FILENAMES = [
    "averaged_health_fitness_dataset_training.csv",
    "averaged_health_fitness_dataset_testing.csv",
    "regularised_averaged_health_fitness_dataset_training.csv",
    "regularised_averaged_health_fitness_dataset_testing.csv",
    "encoded_regularised_averaged_health_fitness_dataset_training.csv",
    "encoded_regularised_averaged_health_fitness_dataset_testing.csv",
    "labelled_regularised_averaged_health_fitness_dataset_training.csv",
    "labelled_regularised_averaged_health_fitness_dataset_testing.csv",
]
PROCESSED_DATASETS = [DATASETS_DIR / name for name in PROCESSED_DATASET_FILENAMES]
PREPROCESSING_METADATA_PATH = DATASETS_DIR / "preprocessing_metadata.json"

LABELLED_TRAIN_DATASET = DATASETS_DIR / "labelled_regularised_averaged_health_fitness_dataset_training.csv"
LABELLED_TEST_DATASET = DATASETS_DIR / "labelled_regularised_averaged_health_fitness_dataset_testing.csv"
CAUSAL_GRAPH_EDGE_PATH = GRAPHS_DIR / "causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy"

CAUSALLEARN_GRAPHS_DIR = GRAPHS_DIR / "causallearn/graphs"
CAUSALLEARN_EDGE_NPY_DIR = GRAPHS_DIR / "causallearn/edges/npy"
CAUSALLEARN_EDGE_TXT_DIR = GRAPHS_DIR / "causallearn/edges/txt"
CAUSALLEARN_STABILITY_DIR = GRAPHS_DIR / "causallearn/stability"
CAUSALLEARN_COMPARISON_DIR = GRAPHS_DIR / "causallearn/comparison"
COUNTERFACTUALS_DIR = GRAPHS_DIR / "counterfactuals"
INFLUENCES_DIR = GRAPHS_DIR / "influences"
TESTS_OUTPUT_DIR = GRAPHS_DIR / "tests"
TIME_SERIES_OUTPUT_DIR = GRAPHS_DIR / "time_series_graphs"
TIME_SERIES_PIDS_OUTPUT_DIR = TIME_SERIES_OUTPUT_DIR / "tsg_pids"

RAW_DATASET_COLUMNS = [
    "participant_id",
    "date",
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "activity_type",
    "duration_minutes",
    "intensity",
    "calories_burned",
    "avg_heart_rate",
    "hours_sleep",
    "stress_level",
    "daily_steps",
    "hydration_level",
    "bmi",
    "resting_heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "health_condition",
    "smoking_status",
    "fitness_level",
]

EXCLUDED_SCALING_COLUMNS = ["date", "participant_id", "height_cm", "weight_kg"]
CATEGORICAL_COLUMNS = ["gender", "activity_type", "intensity", "health_condition", "smoking_status"]
LABEL_ENCODING_SKIP_COLUMNS = ["date", "gender"]

PIDS_PERSONAS = [2, 5, 6, 8, 11, 26, 30, 41, 108, 165, 172, 262]
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
RANDOM_SEED = 7

CAUSAL_DISCOVERY_DATA_TYPE = "labelled"
CAUSAL_DISCOVERY_DROP_COLUMNS = {
    "encoded": ["participant_id", "height_cm", "weight_kg", "gender_M", "gender_F", "gender_Other", "stress_level", "health_condition"],
    "labelled": ["participant_id", "height_cm", "weight_kg", "gender", "stress_level", "health_condition"],
}
CAUSAL_DISCOVERY_PC_CITS = ["fisherz"]
CAUSAL_DISCOVERY_PC_ALPHAS = [0.01, 0.05, 0.10]
CAUSAL_DISCOVERY_PC_ALPHA = 0.05
CAUSAL_DISCOVERY_PC_UC_RULE = 0
CAUSAL_DISCOVERY_PC_UC_PRIORITY = 0
CAUSAL_DISCOVERY_BOOTSTRAP_REPLICATES = 100
CAUSAL_DISCOVERY_ALT_METHOD = "ges"
CAUSAL_DISCOVERY_PATHWAY_TARGETS = [
    ("duration_minutes", "calories_burned"),
    ("duration_minutes", "fitness_level"),
    ("activity_type", "calories_burned"),
    ("daily_steps", "fitness_level"),
    ("age", "avg_heart_rate"),
    ("intensity", "avg_heart_rate"),
]

TIME_SERIES_DROP_COLUMNS = ["participant_id", "height_cm", "weight_kg", "gender", "stress_level", "date"]
TIME_SERIES_LPCMCI_TAUS = [2]
TIME_SERIES_LPCMCI_PCS = [0.05]
TIME_SERIES_PCMCI_TAUS = [2]
TIME_SERIES_PCMCI_PCS = [0.05]
TIME_SERIES_PCMCIPLUS_TAUS = [2]
TIME_SERIES_PCMCIPLUS_PCS = [0.01]
TIME_SERIES_CI_TEST = "PairwiseMultCI"

SINGLE_PID_TIME_SERIES_DROP_COLUMNS = [
    "participant_id",
    "date",
    "height_cm",
    "weight_kg",
    "gender",
    "bmi",
    "resting_heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "smoking_status",
    "health_condition",
    "stress_level",
]
SINGLE_PID_TIME_SERIES_TAUS = [2]
SINGLE_PID_TIME_SERIES_PCS = [0.05]
SINGLE_PID_TIME_SERIES_CI_TEST = "ParCorr"

DOWHY_FALSIFY_PERMUTATIONS = 100
DOWHY_BOOTSTRAP_RESAMPLES = 10

TEST_COUNTERFACTUAL_PID = 42
INFLUENCE_PID = 6
INFLUENCE_TARGET_AVG = -0.2
INFLUENCE_ACTIVITY_TYPE_VALUE = 6
INFLUENCE_DAILY_STEPS_STANDARDIZED_VALUE = 3.0

DISPLAY_LABELS = {
    "activity_type": "activity type",
    "age": "age",
    "avg_heart_rate": "average heart rate",
    "blood_pressure_diastolic": "diastolic blood pressure",
    "blood_pressure_systolic": "systolic blood pressure",
    "bmi": "BMI",
    "calories_burned": "calories burned",
    "daily_steps": "daily steps",
    "date": "month index",
    "duration_minutes": "exercise duration",
    "fitness_level": "fitness level",
    "gender": "gender",
    "health_condition": "health condition",
    "height_cm": "height",
    "hours_sleep": "hours of sleep",
    "hydration_level": "hydration level",
    "intensity": "intensity",
    "participant_id": "participant ID",
    "resting_heart_rate": "resting heart rate",
    "smoking_status": "smoking status",
    "stress_level": "stress level",
    "weight_kg": "weight",
}


def load_causal_graph(edges_path: Path = CAUSAL_GRAPH_EDGE_PATH):
    import networkx as nx
    import numpy as np

    edges = np.load(edges_path, allow_pickle=True)
    nodes = []
    for edge in edges:
        for node in edge:
            if node not in nodes:
                nodes.append(node)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph, edges
