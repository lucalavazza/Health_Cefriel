import networkx as nx
import pandas as pd
import numpy as np
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(7)

fitness_data_training = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
fitness_data_testing = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv')
edges = np.load(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')

nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis, based_on=fitness_data_training,
                                  quality=AssignmentQuality.GOOD)
fitting = gcm.fit(causal_model=causal_model_for_counterfactual_analysis, data=fitness_data_training,
                  return_evaluation_summary=True)

pids_personas = [2, 5, 6, 8, 11, 26, 30, 41, 108, 165, 172, 262]

for pid in pids_personas:
    if pid == 2:
        # PID=2: reduce daily_steps
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 2]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'daily_steps': lambda x: x - 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'],
             fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
             fitness_data_pid['bmi'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories before', 'calories after',
                                                                  'fit level before', 'fit level after',
                                                                  'bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=2, reduce daily_steps", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(2))
    elif pid == 5:
        # PID=5: hours_sleep ==> duration_minutes
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 5]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'hours_sleep': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'hours_sleep': lambda x: x - 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['duration_minutes'],
             counterfactual_data1['duration_minutes'],
             counterfactual_data2['duration_minutes']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['duration_minutes with usual sleep', 'duration_minutes with more sleep',
                                      'duration_minutes with less sleep'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=5, hours_sleep ==> duration_minutes", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(5))
    elif pid == 6:
        # PID=6: duration_minutes ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 6]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: -3},
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['regular', 'lack_of', 'too_much'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, duration_minutes ==> calories_burned", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(6))
    elif pid == 8:
        # PID=8: increase duration_minutes
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 8]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 4},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'],
             fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
             fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate'],
             fitness_data_pid['bmi'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories before', 'calories after', 'fit level before', 'fit level after',
                                      'heart before', 'heart after', 'bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=8, increase duration_minutes", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(8))
    elif pid == 11:
        # PID=11: duration_minutes/daily_steps ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 11]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3,
                                                           'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories_burned before', 'calories_burned after'])
        bar_plot = df_plot.plot.bar(
            title="Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> calories_burned", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(11))
    elif pid == 26:
        # PID=26: fitness_level => calories burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 26]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'fitness_level': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories_burned regularly', 'calories_burned when more fit'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=26, fitness_level ==> calories_burned",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(26))
    elif pid == 30:
        # PID=30: activity_type ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 30]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'activity_type': lambda x: 6},  # tennis
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'activity_type': lambda x: 9},  # yoga
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories_burned when doing preferred sport', 'calories_burned if playing tennis',
                                      'calories_burned if doing yoga'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=30, activity_type ==> calories_burned", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(30))
    elif pid == 41:
        # PID=41: calories_burned ==> bmi
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 41]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'calories_burned': lambda x: x - 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['bmi'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=41, calories_burned ==> bmi", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(41))
    elif pid == 108:
        # PID=108: duration_minutes => blood_pressure/heart_rate
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 165]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['blood_pressure_systolic'], counterfactual_data1['blood_pressure_systolic'],
                               fitness_data_pid['blood_pressure_diastolic'], counterfactual_data1['blood_pressure_diastolic'],
                               fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['blood_pressure_systolic before', 'blood_pressure_systolic after',
                                      'blood_pressure_diastolic before', 'blood_pressure_diastolic after',
                                      'resting_heart_rate before', 'resting_heart_rate after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=108, duration_minutes => blood_pressure/heart_rate",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(108))
    elif pid == 165:
        # PID=165: duration_minutes ==> fitness_level/bmi
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 165]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
                               fitness_data_pid['bmi'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['fitness_level before', 'fitness_level after', 'bmi_before', 'bmi_after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=165, duration_minutes ==> fitness_level/bmi", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(165))
    elif pid == 172:
        # PID=172: duration_minutes ==> resting_heart_rate
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 202]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['resting_heart_rate before', 'resting_heart_rate after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=172, duration_minutes ==> resting_heart_rate", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(172))
    elif pid == 262:
        # PID=262: daily_steps ==> fitness_level + duration_minutes ==> fitness_level
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 262]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3,
                                                           'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['fitness_level before', 'fitness_level after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> fitness_level", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-pid=' + str(262))
    else:
        print('PID ' + str(pid) + ' not found.')

