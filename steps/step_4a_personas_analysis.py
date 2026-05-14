import pandas as pd
import numpy as np
from dowhy import gcm
import matplotlib.pyplot as plt
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
import warnings
from pathlib import Path

from pipeline.config import COUNTERFACTUALS_DIR, LABELLED_TEST_DATASET, LABELLED_TRAIN_DATASET, MONTHS, PIDS_PERSONAS, PREPROCESSING_METADATA_PATH, RANDOM_SEED, load_causal_graph
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
COUNTERFACTUALS_DIR.mkdir(parents=True, exist_ok=True)


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


def combine_interventions(*intervention_dicts):
    merged = {}
    for intervention in intervention_dicts:
        merged.update(intervention)
    return merged


def compat_delta_original(variable_name: str, standardized_delta: float) -> float:
    if PREPROCESSING_METADATA is None:
        return float(standardized_delta)
    return compat_standardized_delta_to_original(standardized_delta, variable_name, PREPROCESSING_METADATA)


def compat_value_original(variable_name: str, standardized_value: float) -> float:
    if PREPROCESSING_METADATA is None:
        return float(standardized_value)
    return compat_standardized_value_to_original(standardized_value, variable_name, PREPROCESSING_METADATA)


def save_original_units_plot(df_plot: pd.DataFrame, row_variable_names, output_path: str, title: str):
    if PREPROCESSING_METADATA is None:
        return
    df_original_units = inverse_transform_indexed_frame(df_plot, row_variable_names, PREPROCESSING_METADATA)
    original_units_plot = df_original_units.plot.bar(title=title, figsize=(20, 20))
    original_units_figure = original_units_plot.get_figure()
    original_units_figure.savefig(output_path)


def compute_percentage_differences(observed_values, counterfactual_values, variable_name: str):
    observed_array = np.asarray(observed_values, dtype=float)
    counterfactual_array = np.asarray(counterfactual_values, dtype=float)
    if PREPROCESSING_METADATA is not None:
        observed_array = inverse_transform_values(observed_array, variable_name, PREPROCESSING_METADATA)
        counterfactual_array = inverse_transform_values(counterfactual_array, variable_name, PREPROCESSING_METADATA)

    percentages = []
    for observed_value, counterfactual_value in zip(observed_array, counterfactual_array):
        baseline = abs(observed_value)
        if np.isclose(baseline, 0.0):
            percentages.append(0.0)
        else:
            percentages.append(round(((counterfactual_value - observed_value) * 100) / baseline, 2))
    return percentages

fitness_data_training = pd.read_csv(LABELLED_TRAIN_DATASET)
fitness_data_testing = pd.read_csv(LABELLED_TEST_DATASET)
G, _ = load_causal_graph()

causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis, based_on=fitness_data_training,
                                  quality=AssignmentQuality.GOOD)
fitting = gcm.fit(causal_model=causal_model_for_counterfactual_analysis, data=fitness_data_training,
                  return_evaluation_summary=True)

pids_personas = PIDS_PERSONAS
months = MONTHS

for pid in pids_personas:
    if pid == 2:
        # PID=2: reduce daily_steps
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 2]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("daily_steps", compat_delta_original("daily_steps", -2.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'],
             fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
             fitness_data_pid['bmi'], counterfactual_data1['bmi']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories before', 'calories after',
                                                                  'fit level before', 'fit level after',
                                                                  'bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=2, reduce daily_steps", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={2}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned', 'fitness_level', 'fitness_level', 'bmi', 'bmi'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={2}-original_units.pdf',
            "Counterfactual outputs: PID=2, reduce daily_steps - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        perc1 = compute_percentage_differences(fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'], 'fitness_level')
        perc2 = compute_percentage_differences(fitness_data_pid['bmi'], counterfactual_data1['bmi'], 'bmi')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(3, figsize=(20, 20))
        p0 = ax[0].bar(months, perc0, width, color='tab:orange')
        ax[0].bar_label(p0, fmt=lambda x: x)
        ax[0].axhline(np.average(perc0), color='pink', linewidth=3)
        ax[0].set_ylabel('%')
        ax[0].set_title('Percentage difference in calories_burned when reducing daily_steps for pid=2')
        ax[0].set_xticks(x, months)
        p1 = ax[1].bar(months, perc1, width, color='tab:red')
        ax[1].bar_label(p1, fmt=lambda x: x)
        ax[1].axhline(np.average(perc1), color='pink', linewidth=3)
        ax[1].set_ylabel('%')
        ax[1].set_title('Percentage difference in fitness_level when reducing daily_steps for pid=2')
        ax[1].set_xticks(x, months)
        p2 = ax[2].bar(months, perc2, width, color='tab:blue')
        ax[2].bar_label(p2, fmt=lambda x: x)
        ax[2].axhline(np.average(perc2), color='pink', linewidth=3)
        ax[2].set_ylabel('%')
        ax[2].set_title('Percentage difference in bmi when reducing daily_steps for pid=2')
        ax[2].set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={2}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 5:
        # PID=5: hours_sleep ==> duration_minutes
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 5]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("hours_sleep", compat_delta_original("hours_sleep", 3.0)),
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("hours_sleep", compat_delta_original("hours_sleep", -3.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['duration_minutes'],
             counterfactual_data1['duration_minutes'],
             counterfactual_data2['duration_minutes']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['duration_minutes with usual sleep', 'duration_minutes with more sleep',
                                      'duration_minutes with less sleep'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=5, hours_sleep ==> duration_minutes",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={5}.pdf')
        save_original_units_plot(
            df_plot,
            ['duration_minutes', 'duration_minutes', 'duration_minutes'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={5}-original_units.pdf',
            "Counterfactual outputs: PID=5, hours_sleep ==> duration_minutes - original units",
        )
        perc1 = compute_percentage_differences(fitness_data_pid['duration_minutes'], counterfactual_data1['duration_minutes'], 'duration_minutes')
        perc2 = compute_percentage_differences(fitness_data_pid['duration_minutes'], counterfactual_data2['duration_minutes'], 'duration_minutes')
        differences = {
            'more sleep': perc1,
            'less sleep': perc2,
        }
        x = np.arange(len(months))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(20, 15))
        for value, diffs in differences.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, diffs, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.axhline(np.average(perc1), color='blue', linewidth=3)
        ax.axhline(np.average(perc2), color='orange', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in duration_minutes depending on hours_sleep for pid=5')
        ax.set_xticks(x + width, months)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={5}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 6:
        # PID=6: duration_minutes ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 6]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          set_value_intervention("duration_minutes", compat_value_original("duration_minutes", 3.0)),
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          set_value_intervention("duration_minutes", compat_value_original("duration_minutes", -3.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['regular', 'more', 'less'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, duration_minutes ==> calories_burned",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={6}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned', 'calories_burned'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={6}-original_units.pdf',
            "Counterfactual outputs: PID=6, duration_minutes ==> calories_burned - original units",
        )
        perc1 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        perc2 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data2['calories_burned'], 'calories_burned')
        differences = {
            'more time': perc1,
            'less time': perc2,
        }
        x = np.arange(len(months))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(20, 15))
        for value, diffs in differences.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, diffs, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.axhline(np.average(perc1), color='blue', linewidth=3)
        ax.axhline(np.average(perc2), color='orange', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in calories_burned depending on duration_minutes for pid=6')
        ax.set_xticks(x + width, months)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={6}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 8:
        # PID=8: increase duration_minutes
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 8]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 4.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'],
             fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
             fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate'],
             fitness_data_pid['bmi'], counterfactual_data1['bmi']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories before', 'calories after', 'fit level before', 'fit level after',
                                      'heart before', 'heart after', 'bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=8, increase duration_minutes", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={8}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned', 'fitness_level', 'fitness_level',
             'resting_heart_rate', 'resting_heart_rate', 'bmi', 'bmi'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={8}-original_units.pdf',
            "Counterfactual outputs: PID=8, increase duration_minutes - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        perc1 = compute_percentage_differences(fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'], 'fitness_level')
        perc2 = compute_percentage_differences(fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate'], 'resting_heart_rate')
        perc3 = compute_percentage_differences(fitness_data_pid['bmi'], counterfactual_data1['bmi'], 'bmi')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(4, figsize=(20, 20))
        p0 = ax[0].bar(months, perc0, width, color='tab:orange')
        ax[0].bar_label(p0, fmt=lambda x: x)
        ax[0].axhline(np.average(perc0), color='pink', linewidth=3)
        ax[0].set_ylabel('%')
        ax[0].set_title('Percentage difference in calories_burned when increasing duration_minutes for pid=8')
        ax[0].set_xticks(x, months)
        p1 = ax[1].bar(months, perc1, width, color='tab:red')
        ax[1].bar_label(p1, fmt=lambda x: x)
        ax[1].axhline(np.average(perc1), color='pink', linewidth=3)
        ax[1].set_ylabel('%')
        ax[1].set_title('Percentage difference in fitness_level when increasing duration_minutes for pid=8')
        ax[1].set_xticks(x, months)
        p2 = ax[2].bar(months, perc2, width, color='tab:blue')
        ax[2].bar_label(p2, fmt=lambda x: x)
        ax[2].axhline(np.average(perc2), color='pink', linewidth=3)
        ax[2].set_ylabel('%')
        ax[2].set_title('Percentage difference in resting_heart_rate when increasing duration_minutes for pid=8')
        ax[2].set_xticks(x, months)
        p3 = ax[3].bar(months, perc3, width, color='tab:pink')
        ax[3].bar_label(p3, fmt=lambda x: x)
        ax[3].axhline(np.average(perc3), color='pink', linewidth=3)
        ax[3].set_ylabel('%')
        ax[3].set_title('Percentage difference in bmi when increasing duration_minutes for pid=8')
        ax[3].set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={8}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 11:
        # PID=11: duration_minutes/daily_steps ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 11]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          combine_interventions(
                                                              add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 3.0)),
                                                              add_delta_intervention("daily_steps", compat_delta_original("daily_steps", 2.0)),
                                                          ),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories_burned before', 'calories_burned after'])
        bar_plot = df_plot.plot.bar(
            title="Counterfactual outputs: PID=11, daily_steps/duration_minutes ==> calories_burned", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={11}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={11}-original_units.pdf',
            "Counterfactual outputs: PID=11, daily_steps/duration_minutes ==> calories_burned - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(figsize=(20, 20))
        p0 = ax.bar(months, perc0, width)
        ax.bar_label(p0, fmt=lambda x: x)
        ax.axhline(np.average(perc0), color='pink', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title(
            'Percentage difference in calories_burned when increasing daily steps and duration_minutes for pid=11')
        ax.set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={11}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 26:
        # PID=26: fitness_level => calories burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 26]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("fitness_level", compat_delta_original("fitness_level", 2.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories_burned regularly', 'calories_burned when more fit'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=26, fitness_level ==> calories_burned",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={26}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={26}-original_units.pdf',
            "Counterfactual outputs: PID=26, fitness_level ==> calories_burned - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(figsize=(20, 20))
        p0 = ax.bar(months, perc0, width)
        ax.bar_label(p0, fmt=lambda x: x)
        ax.axhline(np.average(perc0), color='pink', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in calories_burned when increasing fitness_level for pid=26')
        ax.set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={26}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 30:
        # PID=30: activity_type ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 30]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          set_value_intervention("activity_type", 6),  # tennis
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          set_value_intervention("activity_type", 9),  # yoga
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories_burned when doing preferred sport', 'calories_burned if playing tennis',
                                      'calories_burned if doing yoga'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=30, activity_type ==> calories_burned",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={30}.pdf')
        save_original_units_plot(
            df_plot,
            ['calories_burned', 'calories_burned', 'calories_burned'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={30}-original_units.pdf',
            "Counterfactual outputs: PID=30, activity_type ==> calories_burned - original units",
        )
        perc1 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'], 'calories_burned')
        perc2 = compute_percentage_differences(fitness_data_pid['calories_burned'], counterfactual_data2['calories_burned'], 'calories_burned')
        differences = {
            'tennis': perc1,
            'yoga': perc2,
        }
        x = np.arange(len(months))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(20, 15))
        for value, diffs in differences.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, diffs, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.axhline(np.average(perc1), color='blue', linewidth=3)
        ax.axhline(np.average(perc2), color='orange', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in calories_burned only doing one sport for pid=30')
        ax.set_xticks(x + width, months)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={30}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 41:
        # PID=41: calories_burned ==> bmi
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 41]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("calories_burned", compat_delta_original("calories_burned", -2.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['bmi'], counterfactual_data1['bmi']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=41, calories_burned ==> bmi", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={41}.pdf')
        save_original_units_plot(
            df_plot,
            ['bmi', 'bmi'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={41}-original_units.pdf',
            "Counterfactual outputs: PID=41, calories_burned ==> bmi - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['bmi'], counterfactual_data1['bmi'], 'bmi')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(figsize=(20, 20))
        p0 = ax.bar(months, perc0, width)
        ax.bar_label(p0, fmt=lambda x: x)
        ax.axhline(np.average(perc0), color='pink', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in bmi when lowering calories_burned for pid=41')
        ax.set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={41}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 108:
        # PID=108: duration_minutes => blood_pressure/heart_rate
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 108]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 3.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['blood_pressure_systolic'], counterfactual_data1['blood_pressure_systolic'],
             fitness_data_pid['blood_pressure_diastolic'], counterfactual_data1['blood_pressure_diastolic'],
             fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['blood_pressure_systolic before', 'blood_pressure_systolic after',
                                      'blood_pressure_diastolic before', 'blood_pressure_diastolic after',
                                      'resting_heart_rate before', 'resting_heart_rate after'])
        bar_plot = df_plot.plot.bar(
            title="Counterfactual outputs: PID=108, duration_minutes => blood_pressure/heart_rate",
            figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={108}.pdf')
        save_original_units_plot(
            df_plot,
            ['blood_pressure_systolic', 'blood_pressure_systolic', 'blood_pressure_diastolic',
             'blood_pressure_diastolic', 'resting_heart_rate', 'resting_heart_rate'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={108}-original_units.pdf',
            "Counterfactual outputs: PID=108, duration_minutes => blood_pressure/heart_rate - original units",
        )
        perc0 = compute_percentage_differences(
            fitness_data_pid['blood_pressure_systolic'],
            counterfactual_data1['blood_pressure_systolic'],
            'blood_pressure_systolic',
        )
        perc1 = compute_percentage_differences(
            fitness_data_pid['blood_pressure_diastolic'],
            counterfactual_data1['blood_pressure_diastolic'],
            'blood_pressure_diastolic',
        )
        perc2 = compute_percentage_differences(
            fitness_data_pid['resting_heart_rate'],
            counterfactual_data1['resting_heart_rate'],
            'resting_heart_rate',
        )
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(3, figsize=(20, 20))
        p0 = ax[0].bar(months, perc0, width, color='tab:orange')
        ax[0].bar_label(p0, fmt=lambda x: x)
        ax[0].axhline(np.average(perc0), color='pink', linewidth=3)
        ax[0].set_ylabel('%')
        ax[0].set_title('Percentage difference in blood_pressure_systolic when increasing duration_minutes for pid=108')
        ax[0].set_xticks(x, months)
        p1 = ax[1].bar(months, perc1, width, color='tab:red')
        ax[1].bar_label(p1, fmt=lambda x: x)
        ax[1].axhline(np.average(perc1), color='pink', linewidth=3)
        ax[1].set_ylabel('%')
        ax[1].set_title(
            'Percentage difference in blood_pressure_diastolic when increasing duration_minutes for pid=108')
        ax[1].set_xticks(x, months)
        p2 = ax[2].bar(months, perc2, width, color='tab:blue')
        ax[2].bar_label(p2, fmt=lambda x: x)
        ax[2].axhline(np.average(perc2), color='pink', linewidth=3)
        ax[2].set_ylabel('%')
        ax[2].set_title('Percentage difference in resting_heart_rate when increasing duration_minutes for pid=108')
        ax[2].set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={108}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 165:
        # PID=165: duration_minutes ==> fitness_level/bmi
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 165]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 3.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
                               fitness_data_pid['bmi'], counterfactual_data1['bmi']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['fitness_level before', 'fitness_level after', 'bmi_before', 'bmi_after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=165, duration_minutes ==> fitness_level/bmi",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={165}.pdf')
        save_original_units_plot(
            df_plot,
            ['fitness_level', 'fitness_level', 'bmi', 'bmi'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={165}-original_units.pdf',
            "Counterfactual outputs: PID=165, duration_minutes ==> fitness_level/bmi - original units",
        )
        perc0 = compute_percentage_differences(fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'], 'fitness_level')
        perc1 = compute_percentage_differences(fitness_data_pid['bmi'], counterfactual_data1['bmi'], 'bmi')
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(2, figsize=(20, 20))
        p0 = ax[0].bar(months, perc0, width, color='tab:orange')
        ax[0].bar_label(p0, fmt=lambda x: x)
        ax[0].axhline(np.average(perc0), color='pink', linewidth=3)
        ax[0].set_ylabel('%')
        ax[0].set_title('Percentage difference in fitness_level when increasing duration_minutes for pid=165')
        ax[0].set_xticks(x, months)
        p1 = ax[1].bar(months, perc1, width, color='tab:red')
        ax[1].bar_label(p1, fmt=lambda x: x)
        ax[1].axhline(np.average(perc1), color='pink', linewidth=3)
        ax[1].set_ylabel('%')
        ax[1].set_title('Percentage difference in bmi when increasing duration_minutes for pid=165')
        ax[1].set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={165}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 172:
        # PID=172: duration_minutes ==> resting_heart_rate
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 172]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 3.0)),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['resting_heart_rate before', 'resting_heart_rate after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=172, duration_minutes ==> resting_heart_rate",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={172}.pdf')
        save_original_units_plot(
            df_plot,
            ['resting_heart_rate', 'resting_heart_rate'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={172}-original_units.pdf',
            "Counterfactual outputs: PID=172, duration_minutes ==> resting_heart_rate - original units",
        )
        perc0 = compute_percentage_differences(
            fitness_data_pid['resting_heart_rate'],
            counterfactual_data1['resting_heart_rate'],
            'resting_heart_rate',
        )
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(figsize=(20, 20))
        p0 = ax.bar(months, perc0, width)
        ax.bar_label(p0, fmt=lambda x: x)
        ax.axhline(np.average(perc0), color='pink', linewidth=3)
        ax.set_ylabel('%')
        ax.set_title('Percentage difference in resting_heart_rate when increasing duration_minutes for pid=172')
        ax.set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={172}-percentage_difference.pdf')
        plt.close('all')
    elif pid == 262:
        # PID=262: daily_steps ==> fitness_level + duration_minutes ==> fitness_level
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 262]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          combine_interventions(
                                                              add_delta_intervention("duration_minutes", compat_delta_original("duration_minutes", 3.0)),
                                                              add_delta_intervention("daily_steps", compat_delta_original("daily_steps", 2.0)),
                                                          ),
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['fitness_level before', 'fitness_level after'])
        bar_plot = df_plot.plot.bar(
            title="Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> fitness_level", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={262}.pdf')
        save_original_units_plot(
            df_plot,
            ['fitness_level', 'fitness_level'],
            COUNTERFACTUALS_DIR / f'counterfactual-pid={262}-original_units.pdf',
            "Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> fitness_level - original units",
        )
        perc0 = compute_percentage_differences(
            fitness_data_pid['fitness_level'],
            counterfactual_data1['fitness_level'],
            'fitness_level',
        )
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(figsize=(20, 20))
        p0 = ax.bar(months, perc0, width)
        ax.bar_label(p0, fmt=lambda x: x)
        ax.axhline(np.average(perc0), color='pink', linewidth=3)
        # ax[0].grid(True, linestyle='-.')
        ax.set_ylabel('%')
        ax.set_title(
            'Percentage difference in fitness_level when increasing duration_minutes and daily_steps for pid=262')
        ax.set_xticks(x, months)
        plt.savefig(COUNTERFACTUALS_DIR / f'counterfactual-pid={262}-percentage_difference.pdf')
        plt.close('all')
    else:
        print('PID ' + str(pid) + ' not found.')
