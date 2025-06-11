import csv
import os
from math import trunc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder

# remove annoying warning
warnings.simplefilter(action='ignore', category=FutureWarning)

# import the dataset
fit_data = pd.read_csv('health_fitness_dataset.csv').convert_dtypes()
ids = len(pd.unique(fit_data['participant_id']))

# create the directories for the new separate datasets
all_participants_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants'
single_participant_dir = all_participants_dir + '/pid-'
try:
    os.mkdir(all_participants_dir)
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

# create a separate dataset for each participant_id
for (pid), group in fit_data.groupby(['participant_id']):
    group.to_csv(f'{single_participant_dir}{pid}/health_fitness_dataset_pid-{pid}.csv', index=False)

# for each participant, create a separate dataset for each month
for pid in range(1, ids+1):
    fit_data_pid = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-'+str(pid)+'/health_fitness_dataset_pid-'+str(pid)+'.csv').convert_dtypes()
    for m in range(1, 13):
        if m <= 9:
            month = fit_data_pid[fit_data_pid['date'].str.contains('2024-0'+str(m))]
        else:
            month = fit_data_pid[fit_data_pid['date'].str.contains('2024-' + str(m))]
        month.to_csv(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-'+str(pid)+'_month-'+str(m)+'.csv', index=False)

# convert string values to numerical values + combine previous reworked datasets by averaging each month
for pid in range(1, ids+1):
    for m in range(1, 13):
        fit_data_pid_averaging = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-' + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '.csv')
        activity_type_col = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-' + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '.csv', usecols=['activity_type'])

        # convert string values to numerical values
        fit_data_pid_averaging.replace(['Never', 'Current', 'Former'], [0, 1, 2], inplace=True)
        fit_data_pid_averaging.replace(['None', 'Hypertension', 'Diabetes', 'Asthma'], [0, 1, 2, 3], inplace=True)
        non_numeric_columns = list(fit_data_pid_averaging.select_dtypes(exclude=[np.number]).columns)
        label = LabelEncoder()
        for col in non_numeric_columns:
            fit_data_pid_averaging[col] = label.fit_transform(fit_data_pid_averaging[col])


        # average the values for each month
        column_avgs = fit_data_pid_averaging.mean()
        list_indexes = column_avgs.index.tolist()
        list_values = column_avgs.to_list()

        # fill the averaged dataset
        averaged_fit_data_pid_test = {}
        special_treatment_cols = ['date', 'gender', 'activity_type']
        int_columns = ['participant_id', 'age', 'duration_minutes', 'daily_steps', 'smoking_status', 'health_condition',
                       'avg_heart_rate', 'resting_heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic']
        # the og values are irrealistic, let's bump'em up
        fix_column = ['calories_burned']
        for col in fit_data_pid_averaging.columns:
            if col not in special_treatment_cols:
                if col in int_columns:
                    averaged_fit_data_pid_test[col] = trunc(column_avgs[col])
                elif col in fix_column:
                    averaged_fit_data_pid_test[col] = round(10*column_avgs[col], 2)
                else:
                    averaged_fit_data_pid_test[col] = round(column_avgs[col], 2)
            else:
                if col == 'gender':
                    averaged_fit_data_pid_test[col] = fit_data_pid_averaging['gender'][0]
                elif col == 'date':
                    if m <= 9:
                        averaged_fit_data_pid_test[col] = ('2024-0' + str(m))
                    else:
                        averaged_fit_data_pid_test[col] = ('2024-' + str(m))
                else:
                    averaged_fit_data_pid_test[col] = '#TODO: fix'
        csv_filename = single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_averaged.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(averaged_fit_data_pid_test.keys())
            writer.writerow(averaged_fit_data_pid_test.values())

# combine the datasets
data_names = []
for pid in range(1, ids+1):
    for m in range(1, 13):
        data_names.append(single_participant_dir + str(pid) + '/health_fitness_dataset_pid-' + str(pid) + '_month-' + str(m) + '_averaged.csv')
averaged_dataset = pd.concat(map(pd.read_csv, data_names), ignore_index=True)
averaged_dataset.to_csv('averaged_health_fitness_dataset.csv', index=False)

# TODO: take care of activity_type... or do?
#  Do I actually need it for the causal graph? Isn't it sufficient to just considered the calories burned?
