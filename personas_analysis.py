import dowhy
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import CausalModel
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.utils import plot, bar_plot
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.auto import AssignmentQuality
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(7)

fitness_data = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset.csv')
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
gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis, based_on=fitness_data,
                                  quality=AssignmentQuality.GOOD)
fitting = gcm.fit(causal_model=causal_model_for_counterfactual_analysis, data=fitness_data,
                  return_evaluation_summary=True)

pids_personas = [2, 5, 6, 8, 11, 26, 30, 41, 108, 165, 202, 262]

for pid in pids_personas:
    if pid == 2:
        # PID=2: reduce daily_steps
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 2]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'daily_steps': lambda x: x - 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], fitness_data_pid['fitness_level'], fitness_data_pid['bmi'],
             counterfactual_data1['calories_burned'], counterfactual_data1['fitness_level'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories before', 'fit level before', 'bmi before',
                                                                  'calories after', 'fit level after', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(2))
    elif pid == 5:
        # PID=5: smoking_status => resting_heart_rate
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 5]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'smoking_status': lambda x: x + 1},  # from current to former
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'smoking_status': lambda x: x - 1},  # from current to never
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['resting_heart_rate'],
             counterfactual_data1['resting_heart_rate'],
             counterfactual_data2['resting_heart_rate']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['resting_heart_rate when smoking', 'resting_heart_rate when quitting',
                                      'resting_heart_rate if never smoked'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(5))
    elif pid == 6:
        # PID=6: duration_minutes => calories_burned
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 6]
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
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(6))
    elif pid == 8:
        # PID=8: increase duration_minutes
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 8]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 4},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array(
            [fitness_data_pid['calories_burned'], fitness_data_pid['fitness_level'],
             fitness_data_pid['resting_heart_rate'], fitness_data_pid['bmi'],
             counterfactual_data1['calories_burned'], counterfactual_data1['fitness_level'],
             counterfactual_data1['resting_heart_rate'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories before', 'fit level before', 'heart before', 'bmi before',
                                      'calories after', 'fit level after', 'heart after', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(8))
    elif pid == 11:
        # PID=11: daily_steps => bmi
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 11]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['bmi'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(11))
    elif pid == 26:
        # PID=26: intensity ==> avg_heart_rate
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 26]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'intensity': lambda x: 0},  # low intensity
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'intensity': lambda x: 1},  # average intensity
                                                          observed_data=fitness_data_pid)
        counterfactual_data3 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'intensity': lambda x: 2},  # high intensity
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['avg_heart_rate'], counterfactual_data1['avg_heart_rate'],
                               counterfactual_data2['avg_heart_rate'], counterfactual_data3['avg_heart_rate']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['regular', 'low', 'average', 'high'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(26))
    elif pid == 30:
        # PID=30: smoking_status ==> calories_burned
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 30]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'smoking_status': lambda x: x + 1},  # from current to former
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'smoking_status': lambda x: x - 1},  # from current to never
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['calories_burned when smoking', 'calories_burned when quitting',
                                      'calories_burned if never smoked'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(30))
    elif pid == 41:
        # PID=41: daily_steps ==> calories_burned
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 41]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories_burned before', 'calories_burned after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(41))
    elif pid == 108:
        # PID=108: daily_steps ==> fitness_level
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 108]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['fitness_level before', 'fitness_level after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(108))
    elif pid == 165:
        # PID=165: duration_minutes ==> fitness_level/bmi
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 165]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], fitness_data_pid['bmi'],
                               counterfactual_data1['fitness_level'], counterfactual_data1['bmi']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['fitness_level before', 'bmi_before', 'fitness_level after', 'bmi_after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(165))
    elif pid == 202:
        # PID=202: duration_minutes ==> resting_heart_rate
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 202]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['resting_heart_rate before', 'resting_heart_rate after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(202))
    elif pid == 262:
        # PID=262: daily_steps ==> fitness_level + duration_minutes ==> fitness_level
        fitness_data_pid = fitness_data[fitness_data['participant_id'] == 262]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: x + 3,
                                                           'daily_steps': lambda x: x + 2},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level']])
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        df_plot = pd.DataFrame(array_plot, columns=months, index=['fitness_level before', 'fitness_level after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=' + str(262))
    else:
        print('PID ' + str(pid) + ' not found.')

