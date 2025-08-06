import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# Remove annoying warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Create a directory for saving the analyses
analysis_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/data_analysis/'
viol_plots_dir = analysis_dir + 'violin_plots/'
try:
    os.mkdir(analysis_dir)
    os.mkdir(viol_plots_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create '" + dir_name + "'.")
except Exception as e:
    print(f"An error occurred: {e}")

# General Data Analysis
print("\n1. General Data Analysis")
print("-" * 50)
fit_data = pd.read_csv('../../TechnicalDeepDiveCefriel/datasets/health_fitness_dataset.csv').convert_dtypes()
print("Infos on the dataset:\n")
print(fit_data.info())
print("\nDescription of the dataset:\n")
print(fit_data.describe())
print("\nMissing values in the dataset:\n")
print(fit_data.isna().sum())
box_columns = ['age', 'height_cm', 'weight_kg', 'duration_minutes', 'calories_burned', 'avg_heart_rate', 'hours_sleep',
               'stress_level', 'daily_steps', 'hydration_level', 'bmi', 'resting_heart_rate', 'blood_pressure_systolic',
               'blood_pressure_diastolic', 'fitness_level']
for c in box_columns:
    plt.figure(figsize=(15, 17))
    sns.violinplot(x=c, data=fit_data)
    plt.savefig(viol_plots_dir + 'violinplot_' + str(c) + '.png')
    plt.close()
numerical_columns = list(
    set(fit_data.columns) - {'participant_id', 'date', 'gender', 'activity_type', 'intensity', 'health_condition',
                             'smoking_status'})
corr_matrix = fit_data[numerical_columns].corr()
plt.figure(figsize=(15, 17))
sns.heatmap(corr_matrix)
plt.savefig(viol_plots_dir + 'corr_matrix.png')
plt.close()

# Participant Demographics Analysis
print("\n2. Demographics Analysis")
print("-" * 50)
# Create a figure for demographics
fig = plt.figure(figsize=(20, 6))
# Age Distribution
plt.subplot(1, 3, 1)
sns.histplot(data=fit_data, x='age', bins=30, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
# Gender Distribution
plt.subplot(1, 3, 2)
gender_counts = fit_data['gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        colors=['lightblue', 'lightpink', 'lightgreen'])
plt.title('Gender Distribution')
# BMI Distribution by Gender
plt.subplot(1, 3, 3)
sns.boxplot(data=fit_data, x='gender', y='bmi')
plt.title('BMI Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('BMI')
plt.tight_layout()
plt.savefig(analysis_dir + 'demographic_analysis.png')

# Activity Analysis
print("\n3. Activity Patterns")
print("-" * 50)
# Activity Distribution
plt.figure(figsize=(12, 6))
activity_counts = fit_data['activity_type'].value_counts()
sns.barplot(x=activity_counts.values, y=activity_counts.index, palette='viridis')
plt.title('Distribution of Activities')
plt.xlabel('Number of Sessions')
plt.savefig(analysis_dir + 'activity_distribution_analysis.png')
# Average Calories Burned by Activity
plt.figure(figsize=(12, 6))
avg_calories = fit_data.groupby('activity_type')['calories_burned'].mean().sort_values(ascending=False)
sns.barplot(x=avg_calories.values, y=avg_calories.index, palette='rocket')
plt.title('Average Calories Burned by Activity Type')
plt.xlabel('Calories Burned')
plt.savefig(analysis_dir + 'calories_analysis.png')

# Health Metrics Analysis
print("\n4. Health Metrics")
print("-" * 50)
# Correlation Matrix
health_metrics = ['bmi', 'avg_heart_rate', 'stress_level', 'hours_sleep',
                  'daily_steps', 'calories_burned', 'hydration_level']
plt.figure(figsize=(12, 8))
correlation = fit_data[health_metrics].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Health Metrics')
plt.tight_layout()
plt.savefig(analysis_dir + 'health_correlation.png')
# Health Conditions Distribution
plt.figure(figsize=(10, 6))
health_condition_counts = fit_data['health_condition'].value_counts()
sns.barplot(x=health_condition_counts.index, y=health_condition_counts.values, palette='Set2')
plt.title('Distribution of Health Conditions')
plt.xticks(rotation=45)
plt.savefig(analysis_dir + 'health_distribution_analysis.png')

# Fitness Progress
print("\n6. Fitness Progress Analysis")
print("-" * 50)
# Fitness Level Distribution
plt.figure(figsize=(12, 6))
sns.violinplot(data=fit_data, x='health_condition', y='fitness_level')
plt.title('Fitness Level Distribution by Health Condition')
plt.xticks(rotation=45)
plt.savefig(analysis_dir + 'fitness_level_analysis.png')

# Key Insights
print("\n7. Key Dataset Insights")
print("-" * 50)
insights = {
    'total_participants': fit_data['participant_id'].nunique(),
    'total_activities': len(fit_data),
    'avg_activities_per_person': len(fit_data) / fit_data['participant_id'].nunique(),
    'most_popular_activity': fit_data['activity_type'].mode().iloc[0],
    'avg_calories_per_session': fit_data['calories_burned'].mean(),
    'avg_daily_steps': fit_data['daily_steps'].mean(),
    'avg_sleep_hours': fit_data['hours_sleep'].mean()
}
print("\nKey Insights from the Dataset:")
print(f"Total Participants: {insights['total_participants']:,}")
print(f"Total Activities Recorded: {insights['total_activities']:,}")
print(f"Average Activities per Person: {insights['avg_activities_per_person']:.1f}")
print(f"Most Popular Activity: {insights['most_popular_activity']}")
print(f"Average Calories Burned per Session: {insights['avg_calories_per_session']:.1f}")
print(f"Average Daily Steps: {insights['avg_daily_steps']:,.0f}")
print(f"Average Sleep Hours: {insights['avg_sleep_hours']:.1f}")