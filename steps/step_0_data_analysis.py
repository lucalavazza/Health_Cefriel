from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from pipeline.config import RAW_DATASET, ROOT_DIR


warnings.simplefilter(action="ignore", category=FutureWarning)

ANALYSIS_DIR = ROOT_DIR / "data_analysis"
VIOLIN_PLOTS_DIR = ANALYSIS_DIR / "violin_plots"
VIOLIN_COLUMNS = [
    "age",
    "height_cm",
    "weight_kg",
    "duration_minutes",
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
    "fitness_level",
]
NON_NUMERIC_COLUMNS = {
    "participant_id",
    "date",
    "gender",
    "activity_type",
    "intensity",
    "health_condition",
    "smoking_status",
}
HEALTH_METRICS = [
    "bmi",
    "avg_heart_rate",
    "stress_level",
    "hours_sleep",
    "daily_steps",
    "calories_burned",
    "hydration_level",
]


def ensure_output_directories():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIOLIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATASET).convert_dtypes()


def save_violin_and_correlation_plots(data: pd.DataFrame):
    for column_name in VIOLIN_COLUMNS:
        plt.figure(figsize=(15, 17))
        sns.violinplot(x=column_name, data=data)
        plt.savefig(VIOLIN_PLOTS_DIR / f"violinplot_{column_name}.pdf")
        plt.close()

    numerical_columns = sorted(set(data.columns) - NON_NUMERIC_COLUMNS)
    corr_matrix = data[numerical_columns].corr()
    plt.figure(figsize=(15, 17))
    sns.heatmap(corr_matrix)
    plt.savefig(VIOLIN_PLOTS_DIR / "corr_matrix.pdf")
    plt.close()


def save_demographic_plots(data: pd.DataFrame):
    plt.figure(figsize=(25, 10))

    plt.subplot(1, 3, 1)
    sns.histplot(data=data, x="age", bins=30, color="skyblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    gender_counts = data["gender"].value_counts()
    plt.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct="%1.1f%%",
        colors=["lightblue", "lightpink", "lightgreen"],
    )
    plt.title("Gender Distribution")

    plt.subplot(1, 3, 3)
    sns.boxplot(data=data, x="gender", y="bmi")
    plt.title("BMI Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("BMI")

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "demographic_analysis.pdf")
    plt.close()


def save_activity_plots(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    activity_counts = data["activity_type"].value_counts()
    sns.barplot(x=activity_counts.values, y=activity_counts.index, palette="viridis")
    plt.title("Distribution of Activities")
    plt.xlabel("Number of Sessions")
    plt.savefig(ANALYSIS_DIR / "activity_distribution_analysis.pdf")
    plt.close()

    plt.figure(figsize=(12, 6))
    activity_intensity = pd.crosstab(data["activity_type"], data["intensity"])
    sns.heatmap(activity_intensity, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Activity Intensity Distribution")
    plt.savefig(ANALYSIS_DIR / "activity_intensity_distribution.pdf")
    plt.close()

    plt.figure(figsize=(12, 6))
    avg_calories = data.groupby("activity_type")["calories_burned"].mean().sort_values(ascending=False)
    sns.barplot(x=avg_calories.values, y=avg_calories.index, palette="rocket")
    plt.title("Average Calories Burned by Activity Type")
    plt.xlabel("Calories Burned")
    plt.savefig(ANALYSIS_DIR / "calories_analysis.pdf")
    plt.close()


def save_health_plots(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    correlation = data[HEALTH_METRICS].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Between Health Metrics")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "health_correlation.pdf")
    plt.close()

    plt.figure(figsize=(10, 8))
    health_condition_counts = data["health_condition"].value_counts()
    sns.barplot(x=health_condition_counts.index, y=health_condition_counts.values, palette="Set2")
    plt.title("Distribution of Health Conditions")
    plt.xticks(rotation=45)
    plt.savefig(ANALYSIS_DIR / "health_distribution_analysis.pdf")
    plt.close()


def save_time_series_plot(data: pd.DataFrame):
    daily_activities = data.groupby("date")["activity_type"].count().reset_index()
    plt.figure(figsize=(15, 8))
    plt.plot(daily_activities["date"], daily_activities["activity_type"])
    plt.title("Activity Frequency Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Activities")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(20))
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "activity_frequency_over_time.pdf")
    plt.close()


def save_fitness_plot(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=data, x="health_condition", y="fitness_level")
    plt.title("Fitness Level Distribution by Health Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "fitness_level_analysis.pdf")
    plt.close()


def summarize_dataset(data: pd.DataFrame):
    insights = {
        "rows": len(data),
        "participants": data["participant_id"].nunique(),
        "date_min": data["date"].min(),
        "date_max": data["date"].max(),
        "most_popular_activity": data["activity_type"].mode().iloc[0],
        "avg_calories_per_session": data["calories_burned"].mean(),
        "avg_daily_steps": data["daily_steps"].mean(),
        "avg_sleep_hours": data["hours_sleep"].mean(),
    }
    print("EDA input summary")
    print("-----------------")
    print(f"rows:                 {insights['rows']:,}")
    print(f"participants:         {insights['participants']:,}")
    print(f"date range:           {insights['date_min']} -> {insights['date_max']}")
    print(f"most popular activity:{insights['most_popular_activity']}")
    print(f"avg calories/session: {insights['avg_calories_per_session']:.1f}")
    print(f"avg daily steps:      {insights['avg_daily_steps']:,.0f}")
    print(f"avg sleep hours:      {insights['avg_sleep_hours']:.1f}")


def main():
    ensure_output_directories()
    fit_data = load_raw_data()
    summarize_dataset(fit_data)
    save_violin_and_correlation_plots(fit_data)
    save_demographic_plots(fit_data)
    save_activity_plots(fit_data)
    save_health_plots(fit_data)
    save_time_series_plot(fit_data)
    save_fitness_plot(fit_data)
    print(f"\nEDA outputs written to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
