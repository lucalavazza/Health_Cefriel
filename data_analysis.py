import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fit_data = pd.read_csv('health_fitness_dataset.csv').convert_dtypes()

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
    plt.savefig('violinplot_' + str(c) + '.png')
    plt.close()

numerical_columns = list(set(fit_data.columns) - {'participant_id', 'date', 'gender', 'activity_type', 'intensity', 'health_condition', 'smoking_status'})
corr_matrix = fit_data[numerical_columns].corr()
plt.figure(figsize=(15, 17))
sns.heatmap(corr_matrix)
plt.savefig('corr_matrix.png')


