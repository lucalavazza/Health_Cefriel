import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
import warnings
from pathlib import Path

from pipeline.config import INFLUENCE_PID, INFLUENCE_TARGET_AVG, INFLUENCES_DIR, LABELLED_TEST_DATASET, LABELLED_TRAIN_DATASET, MONTHS, PREPROCESSING_METADATA_PATH, RANDOM_SEED, load_causal_graph
from pipeline.preprocessing import (
    inverse_transform_indexed_frame,
    inverse_transform_values,
    compat_standardized_delta_to_original,
    compat_standardized_value_to_original,
    load_preprocessing_metadata,
    standardize_delta,
    standardize_values,
)

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(RANDOM_SEED)
INFLUENCES_DIR.mkdir(parents=True, exist_ok=True)


def try_load_preprocessing_metadata():
    metadata_path = Path(PREPROCESSING_METADATA_PATH)
    if not metadata_path.exists():
        return None
    return load_preprocessing_metadata(metadata_path)


PREPROCESSING_METADATA = try_load_preprocessing_metadata()


def to_standardized_value(value_original, variable_name: str) -> float:
    if PREPROCESSING_METADATA is None:
        return float(value_original)
    return float(standardize_values([value_original], variable_name, PREPROCESSING_METADATA)[0])


def to_standardized_delta(delta_original, variable_name: str) -> float:
    if PREPROCESSING_METADATA is None:
        return float(delta_original)
    return float(standardize_delta(delta_original, variable_name, PREPROCESSING_METADATA))


def add_delta_intervention(variable_name: str, delta_original):
    delta_standardized = to_standardized_delta(delta_original, variable_name)
    return {variable_name: lambda x, delta=delta_standardized: x + delta}


def set_value_intervention(variable_name: str, value_original):
    value_standardized = to_standardized_value(value_original, variable_name)
    return {variable_name: lambda x, value=value_standardized: value}


def compat_delta_original(variable_name: str, standardized_delta: float) -> float:
    if PREPROCESSING_METADATA is None:
        return float(standardized_delta)
    return compat_standardized_delta_to_original(standardized_delta, variable_name, PREPROCESSING_METADATA)


def compat_value_original(variable_name: str, standardized_value: float) -> float:
    if PREPROCESSING_METADATA is None:
        return float(standardized_value)
    return compat_standardized_value_to_original(standardized_value, variable_name, PREPROCESSING_METADATA)


def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


def maybe_inverse_transform_values(values, column_name: str):
    if PREPROCESSING_METADATA is None:
        return np.asarray(values, dtype=float)
    return inverse_transform_values(values, column_name, PREPROCESSING_METADATA)


def maybe_inverse_transform_frame(df: pd.DataFrame, row_variable_names):
    if PREPROCESSING_METADATA is None:
        return df
    return inverse_transform_indexed_frame(df, row_variable_names, PREPROCESSING_METADATA)


def save_counterfactual_bar_plot(array_plot, index_labels, title: str, output_name: str):
    df_plot = pd.DataFrame(array_plot, columns=MONTHS, index=index_labels)
    df_plot = maybe_inverse_transform_frame(df_plot, ["calories_burned"] * len(index_labels))
    chart = df_plot.plot.bar(title=title, figsize=(20, 20))
    chart.get_figure().savefig(INFLUENCES_DIR / output_name)
    plt.close(chart.get_figure())


def average_calories_original(values):
    calories = maybe_inverse_transform_values(np.array(values), "calories_burned")
    return float(np.average(calories))


def main():
    fitness_data_training = pd.read_csv(LABELLED_TRAIN_DATASET)
    fitness_data_testing = pd.read_csv(LABELLED_TEST_DATASET)
    G, _ = load_causal_graph()

    causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(
        causal_model=causal_model_for_counterfactual_analysis,
        based_on=fitness_data_training,
        quality=AssignmentQuality.GOOD,
    )
    gcm.fit(
        causal_model=causal_model_for_counterfactual_analysis,
        data=fitness_data_training,
        return_evaluation_summary=True,
    )

    fitness_data_pid = fitness_data_testing[fitness_data_testing["participant_id"] == INFLUENCE_PID]
    calories_avg_baseline = average_calories_original(fitness_data_pid["calories_burned"])
    print(f"\nCalories burned (baseline): {calories_avg_baseline:.2f}")

    counterfactual_data1 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("activity_type", 6),
        observed_data=fitness_data_pid,
    )
    save_counterfactual_bar_plot(
        np.array([fitness_data_pid["calories_burned"], counterfactual_data1["calories_burned"]]),
        ["calories_burned when doing multiple sports", "calories_burned if playing only tennis"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, only tennis",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis.pdf",
    )

    counterfactual_data2 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("daily_steps", compat_value_original("daily_steps", 3.0)),
        observed_data=counterfactual_data1,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data1["calories_burned"], counterfactual_data2["calories_burned"]]),
        [
            "calories_burned when doing preferred sport and usual steps",
            "calories_burned if playing tennis and setting a steps limit/goal",
        ],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, only tennis and set steps",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set steps.pdf",
    )

    calories_avg = average_calories_original(counterfactual_data2["calories_burned"])
    print(f"\nCalories burned on average when only playing tennis and walking a set amount of steps daily: {calories_avg:.2f}")

    scm_calories = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm_calories, fitness_data_testing)
    gcm.fit(scm_calories, fitness_data_testing)
    perc_iccs_calories = convert_to_percentage(gcm.intrinsic_causal_influence(scm_calories, target_node="calories_burned"))
    plt.figure(figsize=(20, 20))
    plt.bar(range(len(perc_iccs_calories)), list(perc_iccs_calories.values()))
    plt.xticks(range(len(perc_iccs_calories)), list(perc_iccs_calories.keys()))
    plt.savefig(INFLUENCES_DIR / "iccs_perc_calories_burned.pdf")
    plt.close()

    counterfactual_data3 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 1.0)),
        observed_data=counterfactual_data2,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data2["calories_burned"], counterfactual_data3["calories_burned"]]),
        ["calories_burned with no training increase", "calories_burned with more training"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, only tennis and set steps + more training",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + more training.pdf",
    )
    calories_avg = average_calories_original(counterfactual_data3["calories_burned"])
    print(f"\nCalories burned on average when setting a comfortable increase in duration: {calories_avg:.2f}")

    current_avg = float(np.average(np.array(counterfactual_data2["calories_burned"])))
    target_avg = INFLUENCE_TARGET_AVG
    target_avg_report = maybe_inverse_transform_values([target_avg], "calories_burned")[0]
    delta = 0.0
    step = compat_delta_original("duration_minutes", 0.01)
    max_iters = 5000
    tolerance = 1e-9
    iterations = 0
    while current_avg < target_avg - tolerance and iterations < max_iters:
        counterfactual_data_aim = gcm.counterfactual_samples(
            causal_model_for_counterfactual_analysis,
            add_delta_intervention("duration_minutes", delta),
            observed_data=counterfactual_data2,
        )
        current_avg = float(np.average(np.array(counterfactual_data_aim["calories_burned"])))
        if current_avg < target_avg:
            delta += step
        iterations += 1

    if iterations >= max_iters and current_avg < target_avg - tolerance:
        print(
            "\nTarget not reached within search budget; "
            + f"best standardized average found={current_avg:.6f}, requested={target_avg:.6f}."
        )

    print(f"\nTraining duration increase necessary for reaching at least {target_avg_report:.2f} calories burned on average: {delta:.2f}")

    counterfactual_data4 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        add_delta_intervention("duration_minutes", delta),
        observed_data=counterfactual_data2,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data2["calories_burned"], counterfactual_data4["calories_burned"]]),
        ["calories_burned with no training increase", "calories_burned with exactly more training"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, only tennis and set steps + exact more training",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + exact more training.pdf",
    )
    calories_avg = average_calories_original(counterfactual_data4["calories_burned"])
    print(f"\nCalories burned on average with predicted duration: {calories_avg:.2f}")
    print(f"Influence-analysis outputs written to: {INFLUENCES_DIR}")


if __name__ == "__main__":
    main()
