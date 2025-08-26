import csv
import os
from math import trunc
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# remove annoying warning
warnings.simplefilter(action='ignore', category=FutureWarning)

pids_personas = [2, 5, 6, 8, 11, 26, 30, 41, 108, 165, 202, 262]

# import the dataset
fit_data = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/health_fitness_dataset.csv').convert_dtypes()
ids = len(pd.unique(fit_data['participant_id']))

# create the directories for the new separate datasets
all_participants_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants'
single_participant_dir = all_participants_dir + '/pid-'
datasets_dir = "/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets"

try:
    os.mkdir(all_participants_dir)
    os.mkdir(datasets_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create '" + dir_name + "'.")
except Exception as e:
    print(f"An error occurred: {e}")
for pid in range(1, ids+1):
    try:
        os.mkdir(single_participant_dir+str(pid))
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Permission denied: Unable to create '" + dir_name + "'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# create a separate training dataset for each participant_id
for (pid), group in fit_data.groupby(['participant_id']):
    if pid not in pids_personas:
        group.to_csv(f'{single_participant_dir}{pid}/health_fitness_dataset_pid-{pid}_training.csv', index=False)
# create a separate testing dataset for each participant_id
for (pid), group in fit_data.groupby(['participant_id']):
    group.to_csv(f'{single_participant_dir}{pid}/health_fitness_dataset_pid-{pid}_testing.csv', index=False)

# for each participant, create training a separate dataset for each month
for pid in range(1, ids+1):
    if pid not in pids_personas:
        fit_data_pid = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-'+str(pid)+'/health_fitness_dataset_pid-'+str(pid)+'_training.csv').convert_dtypes()
        for m in range(1, 13):
            if m <= 9:
                month = fit_data_pid[fit_data_pid['date'].str.contains('2024-0'+str(m))]
            else:
                month = fit_data_pid[fit_data_pid['date'].str.contains('2024-' + str(m))]
            month.to_csv(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-'+str(pid)+'_month-'+str(m)+'_training.csv', index=False)
# for each participant, create testing a separate dataset for each month
for pid in range(1, ids+1):
    fit_data_pid = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-'+str(pid)+'/health_fitness_dataset_pid-'+str(pid)+'_testing.csv').convert_dtypes()
    for m in range(1, 13):
        if m <= 9:
            month = fit_data_pid[fit_data_pid['date'].str.contains('2024-0'+str(m))]
        else:
            month = fit_data_pid[fit_data_pid['date'].str.contains('2024-' + str(m))]
        month.to_csv(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-'+str(pid)+'_month-'+str(m)+'_testing.csv', index=False)

# convert string values to numerical values and combine previous reworked training datasets by averaging each month
for pid in range(1, ids+1):
    if pid not in pids_personas:
        for m in range(1, 13):
            fit_data_pid_averaging = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-' + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_training.csv')

            # average the values for each month
            column_avgs = fit_data_pid_averaging.mean()
            # fill the averaged dataset
            averaged_fit_data_pid_test = {}
            no_averaging_cols = ['date', 'gender', 'activity_type', 'intensity', 'health_condition', 'smoking_status']
            int_columns = ['participant_id', 'age', 'duration_minutes', 'daily_steps', 'avg_heart_rate',
                           'resting_heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'calories_burned']
            for col in fit_data_pid_averaging.columns:
                if col not in no_averaging_cols:
                    if col in int_columns:
                        averaged_fit_data_pid_test[col] = trunc(column_avgs[col])
                    else:
                        averaged_fit_data_pid_test[col] = round(column_avgs[col], 2)
                else:
                    if col == 'date':
                        averaged_fit_data_pid_test[col] = (str(m))
                    else:
                        averaged_fit_data_pid_test[col] = fit_data_pid_averaging[col][0]

            csv_filename = single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_averaged_training.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(averaged_fit_data_pid_test.keys())
                writer.writerow(averaged_fit_data_pid_test.values())
# convert string values to numerical values and combine previous reworked testing datasets by averaging each month
for pid in range(1, ids + 1):
    for m in range(1, 13):
        fit_data_pid_averaging = pd.read_csv(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-' + str(
                pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_testing.csv')

        # average the values for each month
        column_avgs = fit_data_pid_averaging.mean()
        # fill the averaged dataset
        averaged_fit_data_pid_test = {}
        no_averaging_cols = ['date', 'gender', 'activity_type', 'intensity', 'health_condition', 'smoking_status']
        int_columns = ['participant_id', 'age', 'duration_minutes', 'daily_steps', 'avg_heart_rate',
                       'resting_heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                       'calories_burned']
        for col in fit_data_pid_averaging.columns:
            if col not in no_averaging_cols:
                if col in int_columns:
                    averaged_fit_data_pid_test[col] = trunc(column_avgs[col])
                else:
                    averaged_fit_data_pid_test[col] = round(column_avgs[col], 2)
            else:
                if col == 'date':
                    averaged_fit_data_pid_test[col] = (str(m))
                else:
                    averaged_fit_data_pid_test[col] = fit_data_pid_averaging[col][0]

        csv_filename = single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(
            pid) + '_month-' + str(m) + '_averaged_testing.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(averaged_fit_data_pid_test.keys())
            writer.writerow(averaged_fit_data_pid_test.values())

# combine the datasets for training
data_names_training = []
for pid in range(1, ids+1):
    if pid not in pids_personas:
        for m in range(1, 13):
            data_names_training.append(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_averaged_training.csv')
averaged_dataset = pd.concat(map(pd.read_csv, data_names_training), ignore_index=True)
averaged_dataset.to_csv(datasets_dir + '/averaged_health_fitness_dataset_training.csv', index=False)
# combine the datasets for testing
data_names_testing = []
for pid in range(1, ids+1):
    for m in range(1, 13):
        data_names_testing.append(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_averaged_testing.csv')
averaged_dataset = pd.concat(map(pd.read_csv, data_names_testing), ignore_index=True)
averaged_dataset.to_csv(datasets_dir + '/averaged_health_fitness_dataset_testing.csv', index=False)

# regularisation with standard scaler for training
regularised_fit_data = pd.read_csv(datasets_dir + '/averaged_health_fitness_dataset_training.csv')
numeric_columns = list(regularised_fit_data.select_dtypes(include=[np.number]).columns)
numeric_columns.remove('date')
to_be_removed = ['participant_id', 'height_cm', 'weight_kg']
for col in to_be_removed:
    numeric_columns.remove(col)
numeric_columns = np.asarray(numeric_columns)
scaler = StandardScaler()
numeric_columns = numeric_columns.reshape(-1, 1)
for col in numeric_columns:
    regularised_fit_data[col] = scaler.fit_transform(regularised_fit_data[col])
regularised_fit_data.to_csv(datasets_dir + '/regularised_averaged_health_fitness_dataset_training.csv', index=False)
# regularisation with standard scaler for testing
regularised_fit_data = pd.read_csv(datasets_dir + '/averaged_health_fitness_dataset_testing.csv')
numeric_columns = list(regularised_fit_data.select_dtypes(include=[np.number]).columns)
numeric_columns.remove('date')
to_be_removed = ['participant_id', 'height_cm', 'weight_kg']
for col in to_be_removed:
    numeric_columns.remove(col)
numeric_columns = np.asarray(numeric_columns)
scaler = StandardScaler()
numeric_columns = numeric_columns.reshape(-1, 1)
for col in numeric_columns:
    regularised_fit_data[col] = scaler.fit_transform(regularised_fit_data[col])
regularised_fit_data.to_csv(datasets_dir + '/regularised_averaged_health_fitness_dataset_testing.csv', index=False)

# one-hot encoding for training
to_be_encoded = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/regularised_averaged_health_fitness_dataset_training.csv')
encoded_dataset = pd.get_dummies(data=to_be_encoded, columns=['gender', 'activity_type', 'intensity',
                                                              'health_condition', 'smoking_status'], dtype='int8')
encoded_dataset.to_csv(datasets_dir + '/encoded_regularised_averaged_health_fitness_dataset_training.csv', index=False)
# one-hot encoding for testing
to_be_encoded = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/regularised_averaged_health_fitness_dataset_testing.csv')
encoded_dataset = pd.get_dummies(data=to_be_encoded, columns=['gender', 'activity_type', 'intensity',
                                                              'health_condition', 'smoking_status'], dtype='int8')
encoded_dataset.to_csv(datasets_dir + '/encoded_regularised_averaged_health_fitness_dataset_testing.csv', index=False)

# numerical encoding for training
to_be_converted = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/regularised_averaged_health_fitness_dataset_training.csv')
to_be_converted.replace(['F', 'M', 'Other'], [0, 1, 2], inplace=True)
to_be_converted.replace(['Never', 'Current', 'Former'], [0, 1, 2], inplace=True)
to_be_converted.replace(['None', 'Hypertension', 'Diabetes', 'Asthma'], [0, 1, 2, 3], inplace=True)
non_numeric_columns = list(to_be_converted.select_dtypes(exclude=[np.number]).columns)
label = LabelEncoder()
for col in non_numeric_columns:
    if col not in ['date', 'gender']:
        to_be_converted[col] = label.fit_transform(to_be_converted[col])
to_be_converted.to_csv(datasets_dir + '/labelled_regularised_averaged_health_fitness_dataset_training.csv', index=False)
# numerical encoding for testing
to_be_converted = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/regularised_averaged_health_fitness_dataset_testing.csv')
to_be_converted.replace(['F', 'M', 'Other'], [0, 1, 2], inplace=True)
to_be_converted.replace(['Never', 'Current', 'Former'], [0, 1, 2], inplace=True)
to_be_converted.replace(['None', 'Hypertension', 'Diabetes', 'Asthma'], [0, 1, 2, 3], inplace=True)
non_numeric_columns = list(to_be_converted.select_dtypes(exclude=[np.number]).columns)
label = LabelEncoder()
for col in non_numeric_columns:
    if col not in ['date', 'gender']:
        to_be_converted[col] = label.fit_transform(to_be_converted[col])
to_be_converted.to_csv(datasets_dir + '/labelled_regularised_averaged_health_fitness_dataset_testing.csv', index=False)
