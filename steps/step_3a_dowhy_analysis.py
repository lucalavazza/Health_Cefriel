import dowhy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.utils import bar_plot
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.auto import AssignmentQuality
import warnings
from pathlib import Path

from pipeline.config import (
    DOWHY_BOOTSTRAP_RESAMPLES,
    DOWHY_FALSIFY_PERMUTATIONS,
    LABELLED_TEST_DATASET,
    LABELLED_TRAIN_DATASET,
    MONTHS,
    PREPROCESSING_METADATA_PATH,
    RANDOM_SEED,
    TEST_COUNTERFACTUAL_PID,
    TESTS_OUTPUT_DIR,
    load_causal_graph,
)
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
TESTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def try_load_preprocessing_metadata():
    metadata_path = Path(PREPROCESSING_METADATA_PATH)
    if not metadata_path.exists():
        return None
    return load_preprocessing_metadata(metadata_path)


PREPROCESSING_METADATA = try_load_preprocessing_metadata()


# Define independence test based on the generalised covariance measure with gradient boosted decision trees as models
def create_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))


def gcm_fal(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)


def validate_counterfactual_pid_data(pid_data: pd.DataFrame, pid: int):
    if pid_data.shape[0] != len(MONTHS):
        raise ValueError(
            f"Counterfactual PID {pid} expected {len(MONTHS)} monthly rows, found {pid_data.shape[0]}."
        )


def compute_percentage_differences(counterfactual_values, observed_values):
    percentages = []
    for counterfactual_value, observed_value in zip(counterfactual_values, observed_values):
        baseline = abs(observed_value)
        if np.isclose(baseline, 0.0):
            percentages.append(0.0)
        else:
            percentages.append(round(((counterfactual_value - observed_value) * 100) / baseline, 2))
    return percentages


def maybe_inverse_transform_values(values, column_name: str):
    if PREPROCESSING_METADATA is None:
        return np.asarray(values, dtype=float)
    return inverse_transform_values(values, column_name, PREPROCESSING_METADATA)


def maybe_inverse_transform_frame(df: pd.DataFrame, row_variable_names):
    if PREPROCESSING_METADATA is None:
        return None
    return inverse_transform_indexed_frame(df, row_variable_names, PREPROCESSING_METADATA)


def to_standardized_value(value_original, column_name: str) -> float:
    if PREPROCESSING_METADATA is None:
        return float(value_original)
    return float(standardize_values([value_original], column_name, PREPROCESSING_METADATA)[0])


def to_standardized_delta(delta_original, column_name: str) -> float:
    if PREPROCESSING_METADATA is None:
        return float(delta_original)
    return float(standardize_delta(delta_original, column_name, PREPROCESSING_METADATA))


def add_delta_intervention(column_name: str, delta_original):
    delta_standardized = to_standardized_delta(delta_original, column_name)
    return {column_name: lambda x, delta=delta_standardized: x + delta}


def set_value_intervention(column_name: str, value_original):
    value_standardized = to_standardized_value(value_original, column_name)
    return {column_name: lambda x, value=value_standardized: value}


DURATION_INTERVENTION_DELTA_ORIGINAL = compat_standardized_delta_to_original(
    1.0,
    "duration_minutes",
    PREPROCESSING_METADATA,
) if PREPROCESSING_METADATA is not None else 1.0

LOW_DURATION_VALUE_ORIGINAL = compat_standardized_value_to_original(
    -3.0,
    "duration_minutes",
    PREPROCESSING_METADATA,
) if PREPROCESSING_METADATA is not None else -3.0

HIGH_DURATION_VALUE_ORIGINAL = compat_standardized_value_to_original(
    3.0,
    "duration_minutes",
    PREPROCESSING_METADATA,
) if PREPROCESSING_METADATA is not None else 3.0

def save_percentage_difference_plot(differences):
    x = np.arange(len(MONTHS))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(figsize=(20, 10))

    for value, diffs in differences.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, diffs, width, label=value)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('%')
    ax.set_title('Percentage difference in calories_burned depending on duration_minutes')
    ax.set_xticks(x + width, MONTHS)
    ax.legend(loc='upper left', ncols=2)
    plt.savefig(TESTS_OUTPUT_DIR / f'test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}-percentage_difference.pdf')
    plt.close(fig)


def main():
    fitness_data_training = pd.read_csv(LABELLED_TRAIN_DATASET)
    G, _ = load_causal_graph()

    print("Running DoWhy analysis and example counterfactual outputs...")
    result = falsify_graph(
        G,
        fitness_data_training,
        n_permutations=DOWHY_FALSIFY_PERMUTATIONS,
        independence_test=gcm_fal,
        conditional_independence_test=gcm_fal,
        plot_histogram=False,
        suggestions=True,
    )
    print(result)

    model = dowhy.CausalModel(
        data=fitness_data_training,
        graph=G,
        treatment="duration_minutes",
        outcome="calories_burned",
    )
    print("\nIdentification\n")
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    print("\nEstimation\n")
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(estimate)

    refute1_results = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        show_progress_bar=True,
        placebo_type="permute",
    )
    print("\nRefutation 1\n")
    print(refute1_results)

    refute2_results = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        show_progress_bar=True,
    )
    print("\nRefutation 2\n")
    print(refute2_results)

    refute3_results = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        show_progress_bar=True,
        subset_fraction=0.8,
    )
    print("\nRefutation 3\n")
    print(refute3_results)

    causal_model = gcm.ProbabilisticCausalModel(G)
    gcm.auto.assign_causal_mechanisms(causal_model, fitness_data_training)
    gcm.fit(causal_model, fitness_data_training)

    median_mean_latencies, uncertainty_mean_latencies = gcm.confidence_intervals(
        lambda: gcm.fit_and_compute(
            gcm.interventional_samples,
            causal_model,
            fitness_data_training,
            interventions=add_delta_intervention("duration_minutes", DURATION_INTERVENTION_DELTA_ORIGINAL),
            observed_data=fitness_data_training,
        )().mean().to_dict(),
        num_bootstrap_resamples=DOWHY_BOOTSTRAP_RESAMPLES,
    )
    avg_calories_burned_before = fitness_data_training.mean().to_dict()["calories_burned"]
    avg_calories_burned_after = median_mean_latencies["calories_burned"]
    avg_calories_burned_before_plot = maybe_inverse_transform_values([avg_calories_burned_before], "calories_burned")[0]
    avg_calories_burned_after_plot = maybe_inverse_transform_values([avg_calories_burned_after], "calories_burned")[0]

    bar_plot(
        dict(before=avg_calories_burned_before_plot, after=avg_calories_burned_after_plot),
        dict(
            before=np.array([avg_calories_burned_before_plot, avg_calories_burned_before_plot]),
            after=maybe_inverse_transform_values(uncertainty_mean_latencies["calories_burned"], "calories_burned"),
        ),
        filename=str(TESTS_OUTPUT_DIR / "test-intervention.pdf"),
        ylabel="Avg. Calories Burned",
        display_plot=False,
        figure_size=(15, 15),
        bar_width=0.4,
        xticks=["Before", "After"],
        xticks_rotation=45,
    )

    causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(
        causal_model=causal_model_for_counterfactual_analysis,
        based_on=fitness_data_training,
        quality=AssignmentQuality.BEST,
    )
    gcm.fit(
        causal_model=causal_model_for_counterfactual_analysis,
        data=fitness_data_training,
        return_evaluation_summary=True,
    )

    fitness_data_42 = fitness_data_training[fitness_data_training["participant_id"] == TEST_COUNTERFACTUAL_PID]
    validate_counterfactual_pid_data(fitness_data_42, TEST_COUNTERFACTUAL_PID)
    counterfactual_data1 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("duration_minutes", LOW_DURATION_VALUE_ORIGINAL),
        observed_data=fitness_data_42,
    )
    counterfactual_data2 = gcm.counterfactual_samples(
        causal_model_for_counterfactual_analysis,
        set_value_intervention("duration_minutes", HIGH_DURATION_VALUE_ORIGINAL),
        observed_data=fitness_data_42,
    )

    array_plot = np.array(
        [
            fitness_data_42["calories_burned"],
            counterfactual_data1["calories_burned"],
            counterfactual_data2["calories_burned"],
        ]
    )
    df_plot = pd.DataFrame(array_plot, columns=MONTHS, index=["regular", "lack_of", "too_much"])
    chart = df_plot.plot.bar(title="Counterfactual outputs", figsize=(17, 17))
    plt.ylabel("Calories Burned")
    chart.get_figure().savefig(TESTS_OUTPUT_DIR / f"test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}.pdf")
    plt.close(chart.get_figure())

    df_original_units = maybe_inverse_transform_frame(df_plot, ["calories_burned", "calories_burned", "calories_burned"])
    if df_original_units is not None:
        original_units_plot = df_original_units.plot.bar(title="Counterfactual outputs - original units", figsize=(17, 17))
        original_units_plot.get_figure().savefig(
            TESTS_OUTPUT_DIR / f"test-counterfactual-pid={TEST_COUNTERFACTUAL_PID}-original_units.pdf"
        )
        plt.close(original_units_plot.get_figure())

    observed_calories_burned = maybe_inverse_transform_values(np.array(fitness_data_42["calories_burned"]), "calories_burned")
    counterfactual_calories_burned_1 = maybe_inverse_transform_values(np.array(counterfactual_data1["calories_burned"]), "calories_burned")
    counterfactual_calories_burned_2 = maybe_inverse_transform_values(np.array(counterfactual_data2["calories_burned"]), "calories_burned")
    differences = {
        "less": compute_percentage_differences(counterfactual_calories_burned_1, observed_calories_burned),
        "more": compute_percentage_differences(counterfactual_calories_burned_2, observed_calories_burned),
    }
    save_percentage_difference_plot(differences)
    print(f"Example DoWhy outputs written to: {TESTS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
