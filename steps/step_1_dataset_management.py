import warnings
from math import trunc
from pathlib import Path
from typing import Dict, List

import pandas as pd

from pipeline.config import CATEGORICAL_COLUMNS, DATASETS_DIR, PIDS_PERSONAS, RAW_DATASET, RAW_DATASET_COLUMNS
from pipeline.preprocessing import (
    build_preprocessing_metadata,
    label_encode_train_test,
    one_hot_encode_train_test,
    save_preprocessing_metadata,
    scale_train_test,
)


warnings.simplefilter(action="ignore", category=FutureWarning)

print("\nNot stuck: it just takes some time to complete!\n")

PARTICIPANTS_DIR = Path("./participants")
EXPECTED_MONTHS = list(range(1, 13))
NO_AVERAGING_COLUMNS = ["date", "gender", "activity_type", "intensity", "health_condition", "smoking_status"]
INTEGER_AVERAGE_COLUMNS = [
    "participant_id",
    "age",
    "duration_minutes",
    "daily_steps",
    "avg_heart_rate",
    "resting_heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "calories_burned",
]


def validate_raw_dataset(df: pd.DataFrame):
    missing_columns = [column for column in RAW_DATASET_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Raw dataset is missing required columns: {missing_columns}")

    df = df[RAW_DATASET_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_rows = df["date"].isna()
    if invalid_date_rows.any():
        raise ValueError(
            "Raw dataset contains invalid dates at rows: "
            f"{df.index[invalid_date_rows].tolist()[:10]}"
        )

    years = sorted(df["date"].dt.year.unique().tolist())
    if len(years) != 1:
        raise ValueError(
            f"Raw dataset spans multiple years {years}. "
            "This preprocessing workflow expects a single calendar year so that one row is created for each month."
        )
    return df


def get_participant_ids(df: pd.DataFrame):
    participant_ids = sorted(int(pid) for pid in df["participant_id"].dropna().unique().tolist())
    if not participant_ids:
        raise ValueError("No participant IDs were found in the raw dataset.")
    return participant_ids


def get_participant_dir(pid: int):
    return PARTICIPANTS_DIR / f"pid-{pid}"


def prepare_output_directories(participant_ids):
    PARTICIPANTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    for pid in participant_ids:
        get_participant_dir(pid).mkdir(parents=True, exist_ok=True)


def serialize_dates(df: pd.DataFrame):
    df_to_write = df.copy()
    if "date" in df_to_write.columns and pd.api.types.is_datetime64_any_dtype(df_to_write["date"]):
        df_to_write["date"] = df_to_write["date"].dt.strftime("%Y-%m-%d")
    return df_to_write


def write_dataframe(df: pd.DataFrame, path: Path):
    serialize_dates(df).to_csv(path, index=False)


def build_month_subset(df: pd.DataFrame, month: int):
    return df[df["date"].dt.month == month].copy()


def build_monthly_average_row(month_df: pd.DataFrame, pid: int, month: int, split_name: str):
    if month_df.empty:
        raise ValueError(
            f"Missing data for participant {pid} in month {month} of the {split_name} split. "
            "The downstream pipeline expects one monthly row for each participant-month, "
            "so the script stops here intentionally."
        )

    column_avgs = month_df.mean(numeric_only=True)
    averaged_row = {}
    for col in month_df.columns:
        if col == "date":
            averaged_row[col] = str(month)
            continue
        if col in NO_AVERAGING_COLUMNS:
            averaged_row[col] = month_df.iloc[0][col]
            continue

        value = column_avgs[col]
        if pd.isna(value):
            raise ValueError(
                f"Unable to compute an averaged value for column '{col}' "
                f"(participant {pid}, month {month}, split {split_name})."
            )

        if col in INTEGER_AVERAGE_COLUMNS:
            averaged_row[col] = trunc(value)
        else:
            averaged_row[col] = round(float(value), 2)

    return averaged_row


def write_month_outputs(
    month_df: pd.DataFrame,
    pid: int,
    month: int,
    split_name: str,
    averaged_rows: List[Dict],
):
    participant_dir = get_participant_dir(pid)
    month_path = participant_dir / f"health_fitness_dataset_pid-{pid}_month-{month}_{split_name}.csv"
    write_dataframe(month_df, month_path)

    averaged_row = build_monthly_average_row(month_df, pid, month, split_name)
    averaged_rows.append(averaged_row)
    averaged_path = participant_dir / f"health_fitness_dataset_pid-{pid}_month-{month}_averaged_{split_name}.csv"
    pd.DataFrame([averaged_row]).to_csv(averaged_path, index=False)


fit_data = validate_raw_dataset(pd.read_csv(RAW_DATASET).convert_dtypes())
participant_ids = get_participant_ids(fit_data)
persona_ids = set(PIDS_PERSONAS)

print("Preparing participant directories...")
prepare_output_directories(participant_ids)

print("Splitting participant data and aggregating monthly outputs...")
training_rows = []
testing_rows = []

for pid in participant_ids:
    participant_df = fit_data[fit_data["participant_id"] == pid].sort_values("date").reset_index(drop=True)
    participant_dir = get_participant_dir(pid)

    testing_path = participant_dir / f"health_fitness_dataset_pid-{pid}_testing.csv"
    write_dataframe(participant_df, testing_path)

    if pid not in persona_ids:
        training_path = participant_dir / f"health_fitness_dataset_pid-{pid}_training.csv"
        write_dataframe(participant_df, training_path)

    for month in EXPECTED_MONTHS:
        month_df = build_month_subset(participant_df, month)
        write_month_outputs(month_df, pid, month, "testing", testing_rows)
        if pid not in persona_ids:
            write_month_outputs(month_df, pid, month, "training", training_rows)

print("Writing combined averaged datasets...")
training_averaged_dataset = pd.DataFrame(training_rows)
testing_averaged_dataset = pd.DataFrame(testing_rows)

training_averaged_path = DATASETS_DIR / "averaged_health_fitness_dataset_training.csv"
testing_averaged_path = DATASETS_DIR / "averaged_health_fitness_dataset_testing.csv"
training_averaged_dataset.to_csv(training_averaged_path, index=False)
testing_averaged_dataset.to_csv(testing_averaged_path, index=False)

print("Scaling numeric columns with training-fitted statistics...")
regularised_fit_data_training = pd.read_csv(training_averaged_path)
regularised_fit_data_testing = pd.read_csv(testing_averaged_path)
regularised_fit_data_training, regularised_fit_data_testing, scaling_metadata = scale_train_test(
    regularised_fit_data_training,
    regularised_fit_data_testing,
    return_metadata=True,
)

regularised_training_path = DATASETS_DIR / "regularised_averaged_health_fitness_dataset_training.csv"
regularised_testing_path = DATASETS_DIR / "regularised_averaged_health_fitness_dataset_testing.csv"
regularised_fit_data_training.to_csv(regularised_training_path, index=False)
regularised_fit_data_testing.to_csv(regularised_testing_path, index=False)

print("Creating aligned one-hot encoded datasets...")
to_be_encoded_training = pd.read_csv(regularised_training_path)
to_be_encoded_testing = pd.read_csv(regularised_testing_path)
encoded_dataset_training, encoded_dataset_testing = one_hot_encode_train_test(
    to_be_encoded_training,
    to_be_encoded_testing,
    CATEGORICAL_COLUMNS,
)
encoded_dataset_training.to_csv(
    DATASETS_DIR / "encoded_regularised_averaged_health_fitness_dataset_training.csv",
    index=False,
)
encoded_dataset_testing.to_csv(
    DATASETS_DIR / "encoded_regularised_averaged_health_fitness_dataset_testing.csv",
    index=False,
)

print("Creating labelled datasets with train-fitted categorical mappings...")
to_be_converted_training = pd.read_csv(regularised_training_path)
to_be_converted_testing = pd.read_csv(regularised_testing_path)
to_be_converted_training, to_be_converted_testing, label_mappings = label_encode_train_test(
    to_be_converted_training,
    to_be_converted_testing,
    return_metadata=True,
)
to_be_converted_training.to_csv(
    DATASETS_DIR / "labelled_regularised_averaged_health_fitness_dataset_training.csv",
    index=False,
)
to_be_converted_testing.to_csv(
    DATASETS_DIR / "labelled_regularised_averaged_health_fitness_dataset_testing.csv",
    index=False,
)

print("Saving preprocessing metadata for downstream inverse transformations...")
preprocessing_metadata = build_preprocessing_metadata(
    scaling_metadata=scaling_metadata,
    label_mappings=label_mappings,
    one_hot_columns=encoded_dataset_training.columns.tolist(),
    categorical_columns=CATEGORICAL_COLUMNS,
)
save_preprocessing_metadata(preprocessing_metadata)

print("Dataset management completed.")
