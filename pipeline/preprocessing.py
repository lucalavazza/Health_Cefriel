import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    CATEGORICAL_COLUMNS,
    EXCLUDED_SCALING_COLUMNS,
    LABEL_ENCODING_SKIP_COLUMNS,
    PREPROCESSING_METADATA_PATH,
)


def get_scaled_numeric_columns(df: pd.DataFrame, excluded_numeric_columns=None):
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    excluded_numeric_columns = excluded_numeric_columns or EXCLUDED_SCALING_COLUMNS
    for col in excluded_numeric_columns:
        if col in numeric_columns:
            numeric_columns.remove(col)
    return numeric_columns


def scale_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, excluded_numeric_columns=None, return_metadata=False):
    numeric_columns = get_scaled_numeric_columns(train_df, excluded_numeric_columns=excluded_numeric_columns)
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.loc[:, numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
    test_df.loc[:, numeric_columns] = scaler.transform(test_df[numeric_columns])
    if not return_metadata:
        return train_df, test_df

    scaling_metadata = {
        column: {
            "mean": float(scaler.mean_[index]),
            "scale": float(scaler.scale_[index]),
        }
        for index, column in enumerate(numeric_columns)
    }
    return train_df, test_df, scaling_metadata


def one_hot_encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns=None):
    categorical_columns = categorical_columns or CATEGORICAL_COLUMNS
    encoded_train = pd.get_dummies(data=train_df, columns=categorical_columns, dtype="int8")
    encoded_test = pd.get_dummies(data=test_df, columns=categorical_columns, dtype="int8")

    extra_test_columns = sorted(set(encoded_test.columns) - set(encoded_train.columns))
    if extra_test_columns:
        raise ValueError(f"Unexpected categories found only in test data: {extra_test_columns}")

    encoded_test = encoded_test.reindex(columns=encoded_train.columns, fill_value=0)
    return encoded_train, encoded_test


def apply_manual_category_mappings(df: pd.DataFrame):
    df = df.copy()
    df.replace(["F", "M", "Other"], [0, 1, 2], inplace=True)
    df.replace(["Never", "Current", "Former"], [0, 1, 2], inplace=True)
    df.replace(["None", "Hypertension", "Diabetes", "Asthma"], [0, 1, 2, 3], inplace=True)
    return df


def label_encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, skip_columns=None, return_metadata=False):
    skip_columns = skip_columns or LABEL_ENCODING_SKIP_COLUMNS
    train_df = apply_manual_category_mappings(train_df)
    test_df = apply_manual_category_mappings(test_df)
    label_mappings = {}

    non_numeric_columns = list(train_df.select_dtypes(exclude=[np.number]).columns)
    for col in non_numeric_columns:
        if col in skip_columns:
            continue

        train_values = train_df[col].dropna().astype(str)
        test_values = test_df[col].dropna().astype(str)
        categories = sorted(train_values.unique().tolist())
        unseen_test_values = sorted(set(test_values.unique()) - set(categories))
        if unseen_test_values:
            raise ValueError(f"Unexpected labels found only in test data for '{col}': {unseen_test_values}")

        mapping = {value: index for index, value in enumerate(categories)}
        label_mappings[col] = mapping
        train_df.loc[:, col] = train_df[col].astype(str).map(mapping)
        test_df.loc[:, col] = test_df[col].astype(str).map(mapping)

    if return_metadata:
        return train_df, test_df, label_mappings
    return train_df, test_df


def build_preprocessing_metadata(
    scaling_metadata,
    label_mappings,
    one_hot_columns,
    categorical_columns=None,
    excluded_numeric_columns=None,
):
    return {
        "scaled_numeric_columns": scaling_metadata,
        "label_mappings": label_mappings,
        "one_hot_columns": list(one_hot_columns),
        "categorical_columns": list(categorical_columns or CATEGORICAL_COLUMNS),
        "excluded_numeric_columns": list(excluded_numeric_columns or EXCLUDED_SCALING_COLUMNS),
    }


def save_preprocessing_metadata(metadata, metadata_path=PREPROCESSING_METADATA_PATH):
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_preprocessing_metadata(metadata_path=PREPROCESSING_METADATA_PATH):
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def inverse_transform_values(values, column_name: str, metadata) -> np.ndarray:
    scaling_metadata = metadata.get("scaled_numeric_columns", {})
    if column_name not in scaling_metadata:
        return np.asarray(values, dtype=float)

    mean = float(scaling_metadata[column_name]["mean"])
    scale = float(scaling_metadata[column_name]["scale"])
    return np.asarray(values, dtype=float) * scale + mean


def standardize_values(values, column_name: str, metadata) -> np.ndarray:
    scaling_metadata = metadata.get("scaled_numeric_columns", {})
    if column_name not in scaling_metadata:
        return np.asarray(values, dtype=float)

    mean = float(scaling_metadata[column_name]["mean"])
    scale = float(scaling_metadata[column_name]["scale"])
    if np.isclose(scale, 0.0):
        return np.asarray(values, dtype=float)
    return (np.asarray(values, dtype=float) - mean) / scale


def standardize_delta(delta, column_name: str, metadata) -> float:
    scaling_metadata = metadata.get("scaled_numeric_columns", {})
    if column_name not in scaling_metadata:
        return float(delta)
    scale = float(scaling_metadata[column_name]["scale"])
    if np.isclose(scale, 0.0):
        return float(delta)
    return float(delta) / scale


def compat_standardized_delta_to_original(delta_standardized, column_name: str, metadata) -> float:
    scaling_metadata = metadata.get("scaled_numeric_columns", {})
    if column_name not in scaling_metadata:
        return float(delta_standardized)
    scale = float(scaling_metadata[column_name]["scale"])
    return float(delta_standardized) * scale


def compat_standardized_value_to_original(value_standardized, column_name: str, metadata) -> float:
    return float(inverse_transform_values([value_standardized], column_name, metadata)[0])


def inverse_transform_indexed_frame(df: pd.DataFrame, row_variable_names, metadata) -> pd.DataFrame:
    if len(row_variable_names) != len(df.index):
        raise ValueError("row_variable_names must match the number of rows in the dataframe.")

    df_out = df.copy()
    for row_label, variable_name in zip(df.index, row_variable_names):
        df_out.loc[row_label] = inverse_transform_values(df.loc[row_label].to_numpy(dtype=float), variable_name, metadata)
    return df_out
