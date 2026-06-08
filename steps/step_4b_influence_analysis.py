import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.auto import AssignmentQuality
from dowhy.gcm.util.general import set_random_seed

from pipeline.config import (
    DISPLAY_LABELS,
    INFLUENCE_ACTIVITY_TYPE_VALUE,
    INFLUENCE_DAILY_STEPS_STANDARDIZED_VALUE,
    INFLUENCE_PID,
    INFLUENCE_TARGET_AVG,
    INFLUENCES_DIR,
    LABELLED_TEST_DATASET,
    LABELLED_TRAIN_DATASET,
    MONTHS,
    PREPROCESSING_METADATA_PATH,
    RANDOM_SEED,
    load_causal_graph,
)
from pipeline.preprocessing import (
    compat_standardized_delta_to_original,
    compat_standardized_value_to_original,
    inverse_transform_indexed_frame,
    inverse_transform_values,
    load_preprocessing_metadata,
    standardize_delta,
    standardize_values,
)
from pipeline.visualization import apply_axis_style, configure_matplotlib, display_label


warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

set_random_seed(RANDOM_SEED)
configure_matplotlib()
INFLUENCES_DIR.mkdir(parents=True, exist_ok=True)


def try_load_preprocessing_metadata():
    metadata_path = Path(PREPROCESSING_METADATA_PATH)
    if not metadata_path.exists():
        return None
    return load_preprocessing_metadata(metadata_path)


PREPROCESSING_METADATA = try_load_preprocessing_metadata()

_ORIGINAL_FIND_BEST_MODEL = gcm.auto.find_best_model


def _find_best_model_single_process(*args, **kwargs):
    kwargs["n_jobs"] = 1
    return _ORIGINAL_FIND_BEST_MODEL(*args, **kwargs)


gcm.auto.find_best_model = _find_best_model_single_process


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


def compute_influence_profile_with_fallback(graph, model, data_frame: pd.DataFrame, target_node: str):
    try:
        return convert_to_percentage(gcm.intrinsic_causal_influence(model, target_node=target_node))
    except PermissionError:
        parent_nodes = list(graph.predecessors(target_node))
        if not parent_nodes:
            return {target_node: 100.0}

        scores = {}
        target_values = data_frame[target_node].to_numpy(dtype=float)
        for parent in parent_nodes:
            parent_values = data_frame[parent].to_numpy(dtype=float)
            correlation = np.corrcoef(parent_values, target_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            scores[parent] = abs(float(correlation))

        total = sum(scores.values())
        if np.isclose(total, 0.0):
            equal_share = 100.0 / len(scores)
            return {parent: equal_share for parent in scores}
        return {parent: value / total * 100 for parent, value in scores.items()}


def maybe_inverse_transform_values(values, column_name: str):
    if PREPROCESSING_METADATA is None:
        return np.asarray(values, dtype=float)
    return inverse_transform_values(values, column_name, PREPROCESSING_METADATA)


def maybe_inverse_transform_frame(df: pd.DataFrame, row_variable_names):
    if PREPROCESSING_METADATA is None:
        return df
    return inverse_transform_indexed_frame(df, row_variable_names, PREPROCESSING_METADATA)


def save_counterfactual_bar_plot(array_plot, index_labels, row_variable_names, title: str, output_name: str):
    df_plot = pd.DataFrame(array_plot, columns=MONTHS, index=index_labels)
    df_plot = maybe_inverse_transform_frame(df_plot, row_variable_names)
    chart = df_plot.plot.bar(title=title, figsize=(16, 8))
    chart.set_ylabel(display_label("calories_burned"))
    apply_axis_style(chart, rotation=20)
    chart.get_figure().savefig(INFLUENCES_DIR / output_name, bbox_inches="tight")
    plt.close(chart.get_figure())


def average_calories_original(values):
    calories = maybe_inverse_transform_values(np.array(values), "calories_burned")
    return float(np.average(calories))


def save_decision_summary_plot(summary):
    categories = [
        "baseline",
        "tennis_only",
        "tennis_steps_fixed",
        "target_attained",
    ]
    labels = [
        "Baseline",
        "Tennis only",
        "Tennis + steps fixed",
        "Target-attaining\nscenario",
    ]
    values = [summary[key]["avg_calories_burned"] for key in categories]
    target_value = summary["target_avg_calories_burned"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#9fbad6", "#7aa6c2", "#4f7ca8", "#274c77"]
    ax.bar(labels, values, color=colors)
    ax.axhline(target_value, color="#bf4342", linestyle="--", linewidth=2, label="Target average")
    ax.set_ylabel(display_label("calories_burned"))
    ax.set_title("PID 6 target-directed counterfactual decision planning")
    ax.legend()
    apply_axis_style(ax, rotation=0)
    fig.savefig(INFLUENCES_DIR / "pid6_targeted_decision_plan.pdf", bbox_inches="tight")
    plt.close(fig)


def write_decision_summary(summary):
    json_path = INFLUENCES_DIR / "pid6_decision_summary.json"
    txt_path = INFLUENCES_DIR / "pid6_decision_summary.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "PID 6 target-directed decision planning summary",
        f"Baseline average calories burned: {summary['baseline']['avg_calories_burned']:.2f}",
        f"Average after fixing activity type to tennis: {summary['tennis_only']['avg_calories_burned']:.2f}",
        f"Average after fixing activity type and daily steps: {summary['tennis_steps_fixed']['avg_calories_burned']:.2f}",
        f"Target average calories burned: {summary['target_avg_calories_burned']:.2f}",
        f"Required exercise duration increase: {summary['required_duration_change_original_units']:.2f}",
        f"Average after target-attaining duration change: {summary['target_attained']['avg_calories_burned']:.2f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_state(name: str, data_frame: pd.DataFrame):
    return {
        "scenario": name,
        "avg_calories_burned": average_calories_original(data_frame["calories_burned"]),
        "avg_exercise_duration": float(np.average(maybe_inverse_transform_values(np.array(data_frame["duration_minutes"]), "duration_minutes"))),
    }


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
    )

    fitness_data_pid = fitness_data_testing[fitness_data_testing["participant_id"] == INFLUENCE_PID]
    baseline_summary = summarize_state("baseline", fitness_data_pid)
    print(f"\nCalories burned (baseline): {baseline_summary['avg_calories_burned']:.2f}")

    counterfactual_data1 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("activity_type", INFLUENCE_ACTIVITY_TYPE_VALUE),
        observed_data=fitness_data_pid,
    )
    save_counterfactual_bar_plot(
        np.array([fitness_data_pid["calories_burned"], counterfactual_data1["calories_burned"]]),
        ["baseline", "activity type fixed to tennis"],
        ["calories_burned", "calories_burned"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, activity type fixed to tennis",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis.pdf",
    )

    counterfactual_data2 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("daily_steps", compat_value_original("daily_steps", INFLUENCE_DAILY_STEPS_STANDARDIZED_VALUE)),
        observed_data=counterfactual_data1,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data1["calories_burned"], counterfactual_data2["calories_burned"]]),
        ["tennis only", "tennis + daily steps fixed"],
        ["calories_burned", "calories_burned"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, activity type and daily steps fixed",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set steps.pdf",
    )

    constrained_summary = summarize_state("tennis_steps_fixed", counterfactual_data2)
    print(
        "\nCalories burned on average when only playing tennis and walking a fixed amount of daily steps: "
        f"{constrained_summary['avg_calories_burned']:.2f}"
    )

    scm_calories = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm_calories, fitness_data_testing)
    gcm.fit(scm_calories, fitness_data_testing)
    perc_iccs_calories = compute_influence_profile_with_fallback(
        G,
        scm_calories,
        fitness_data_testing,
        target_node="calories_burned",
    )
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(perc_iccs_calories)), list(perc_iccs_calories.values()), color="#5072A7")
    plt.xticks(range(len(perc_iccs_calories)), [display_label(key) for key in perc_iccs_calories.keys()], rotation=35, ha="right")
    plt.ylabel("Relative influence (%)")
    plt.title("Intrinsic causal influence profile for calories burned")
    plt.tight_layout()
    plt.savefig(INFLUENCES_DIR / "iccs_perc_calories_burned.pdf", bbox_inches="tight")
    plt.close()

    comfortable_duration_change = compat_delta_original("duration_minutes", 1.0)
    counterfactual_data3 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        add_delta_intervention("duration_minutes", comfortable_duration_change),
        observed_data=counterfactual_data2,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data2["calories_burned"], counterfactual_data3["calories_burned"]]),
        ["no duration increase", "comfortable duration increase"],
        ["calories_burned", "calories_burned"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, activity type and daily steps fixed with additional training",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + more training.pdf",
    )
    comfortable_summary = summarize_state("comfortable_change", counterfactual_data3)
    print(
        "\nCalories burned on average after a comfortable duration increase: "
        f"{comfortable_summary['avg_calories_burned']:.2f}"
    )

    current_avg_standardized = float(np.average(np.array(counterfactual_data2["calories_burned"])))
    target_avg_standardized = INFLUENCE_TARGET_AVG
    target_avg_report = maybe_inverse_transform_values([target_avg_standardized], "calories_burned")[0]
    required_duration_change = 0.0
    search_step = compat_delta_original("duration_minutes", 0.01)
    max_iters = 5000
    tolerance = 1e-9
    iterations = 0

    while current_avg_standardized < target_avg_standardized - tolerance and iterations < max_iters:
        counterfactual_data_aim = gcm.counterfactual_samples(
            causal_model_for_counterfactual_analysis,
            add_delta_intervention("duration_minutes", required_duration_change),
            observed_data=counterfactual_data2,
        )
        current_avg_standardized = float(np.average(np.array(counterfactual_data_aim["calories_burned"])))
        if current_avg_standardized < target_avg_standardized:
            required_duration_change += search_step
        iterations += 1

    if iterations >= max_iters and current_avg_standardized < target_avg_standardized - tolerance:
        print(
            "\nTarget not reached within search budget; "
            + f"best standardized average found={current_avg_standardized:.6f}, requested={target_avg_standardized:.6f}."
        )

    print(
        "\nExercise duration increase necessary for reaching at least "
        f"{target_avg_report:.2f} calories burned on average: {required_duration_change:.2f}"
    )

    counterfactual_data4 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        add_delta_intervention("duration_minutes", required_duration_change),
        observed_data=counterfactual_data2,
    )
    save_counterfactual_bar_plot(
        np.array([counterfactual_data2["calories_burned"], counterfactual_data4["calories_burned"]]),
        ["no duration increase", "target-attaining duration increase"],
        ["calories_burned", "calories_burned"],
        f"Counterfactual outputs: PID={INFLUENCE_PID}, target-attaining exercise duration change",
        f"counterfactual-pid={INFLUENCE_PID}_only tennis and set duration + exact more training.pdf",
    )
    target_summary = summarize_state("target_attained", counterfactual_data4)
    print(f"\nCalories burned on average with target-attaining duration: {target_summary['avg_calories_burned']:.2f}")

    summary = {
        "participant_id": INFLUENCE_PID,
        "activity_type_fixed_to_value": INFLUENCE_ACTIVITY_TYPE_VALUE,
        "activity_type_fixed_to_label": "tennis",
        "daily_steps_fixed_standardized_value": INFLUENCE_DAILY_STEPS_STANDARDIZED_VALUE,
        "daily_steps_fixed_original_value": compat_value_original("daily_steps", INFLUENCE_DAILY_STEPS_STANDARDIZED_VALUE),
        "target_avg_calories_burned": float(target_avg_report),
        "required_duration_change_original_units": float(required_duration_change),
        "baseline": baseline_summary,
        "tennis_only": summarize_state("tennis_only", counterfactual_data1),
        "tennis_steps_fixed": constrained_summary,
        "comfortable_change": comfortable_summary,
        "target_attained": target_summary,
        "display_labels": DISPLAY_LABELS,
    }
    write_decision_summary(summary)
    save_decision_summary_plot(summary)
    print(f"Influence-analysis outputs written to: {INFLUENCES_DIR}")


if __name__ == "__main__":
    main()
