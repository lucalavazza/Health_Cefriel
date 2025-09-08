import networkx as nx
import pandas as pd
import numpy as np
from dowhy import gcm
import matplotlib.pyplot as plt
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
import warnings
from sklearn.preprocessing import StandardScaler

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
# alternative_personas = [180, 191, 609, 614, 918, 1323, 2022, 2047, 2207, 2457, 2476, 2720]
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
          'december']

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
        
        df_plot = pd.DataFrame(array_plot, columns=months, index=['calories before', 'calories after',
                                                                  'fit level before', 'fit level after',
                                                                  'bmi before', 'bmi after'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=2, reduce daily_steps", figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(2))
        # re-scaled data
        scaler = StandardScaler().fit(df_plot)  # this
        non_scaled_data = scaler.inverse_transform(df_plot)  # this
        df_ns_plot = pd.DataFrame(non_scaled_data, columns=months, index=['calories before', 'calories after',
                                                                          'fit level before', 'fit level after',
                                                                          'bmi before', 'bmi after'])  # this
        ns_bar_plot = df_ns_plot.plot.bar(title="Counterfactual outputs: PID=2, reduce daily_steps - non scaled",
                                          figsize=(20, 20))
        ns_fig = ns_bar_plot.get_figure()
        ns_fig.savefig(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(2) + '-non_scaled')
        diff0 = np.subtract(np.array(counterfactual_data1['calories_burned']), np.array(fitness_data_pid['calories_burned']))
        diff1 = np.subtract(np.array(counterfactual_data1['fitness_level']), np.array(fitness_data_pid['fitness_level']))
        diff2 = np.subtract(np.array(counterfactual_data1['bmi']), np.array(fitness_data_pid['bmi']))
        perc0 = []
        perc1 = []
        perc2 = []
        for i in range(len(months)):
            diff_0 = diff0[i] * 100
            diff_1 = diff1[i] * 100
            diff_2 = diff2[i] * 100
            max_v0 = np.array(fitness_data_pid['calories_burned'])[i]
            max_v1 = np.array(fitness_data_pid['fitness_level'])[i]
            max_v2 = np.array(fitness_data_pid['bmi'])[i]
            perc_0 = diff_0 / abs(max_v0)
            perc_1 = diff_1 / abs(max_v1)
            perc_2 = diff_2 / abs(max_v2)
            perc0.append(int(perc_0))
            perc1.append(int(perc_1))
            perc2.append(int(perc_2))
        x = np.arange(len(months))  # the label locations
        width = 0.65  # the width of the bars
        fig, ax = plt.subplots(3, figsize=(15, 15))
        p0 = ax[0].bar(months, perc0, width, color='tab:orange')
        ax[0].bar_label(p0, fmt=lambda x: x)
        # ax[0].grid(True, linestyle='-.')
        ax[0].set_ylabel('%')
        ax[0].set_title('Percentual difference in calories_burned when reducing daily_steps')
        ax[0].set_xticks(x, months)
        p1 = ax[1].bar(months, perc1, width, color='tab:red')
        ax[1].bar_label(p1, fmt=lambda x: x)
        # ax[1].grid(True, linestyle='-.')
        ax[1].set_ylabel('%')
        ax[1].set_title('Percentual difference in fitness_level when reducing daily_steps')
        ax[1].set_xticks(x, months)
        p2 = ax[2].bar(months, perc2, width, color='tab:blue')
        ax[2].bar_label(p2, fmt=lambda x: x)
        # ax[2].grid(True, linestyle='-.')
        ax[2].set_ylabel('%')
        ax[2].set_title('Percentual difference in bmi when reducing daily_steps')
        ax[2].set_xticks(x, months)
        plt.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(2) + '-percentual_difference')
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

        df_plot = pd.DataFrame(array_plot, columns=months,
                               index=['duration_minutes with usual sleep', 'duration_minutes with more sleep',
                                      'duration_minutes with less sleep'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=5, hours_sleep ==> duration_minutes",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(5))
        # re-scaled data
        scaler = StandardScaler().fit(df_plot)  # this
        non_scaled_data = scaler.inverse_transform(df_plot)  # this
        df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
                                  index=['duration_minutes with usual sleep', 'duration_minutes with more sleep',
                                         'duration_minutes with less sleep'])  # this
        ns_bar_plot = df_ns_plot.plot.bar(title="Counterfactual outputs: PID=5, hours_sleep ==> duration_minutes - non scaled",
                                          figsize=(20, 20))
        ns_fig = ns_bar_plot.get_figure()
        ns_fig.savefig(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(5) + '-non_scaled')
        diff1 = np.subtract(np.array(counterfactual_data1['duration_minutes']),
                            np.array(fitness_data_pid['duration_minutes']))
        diff2 = np.subtract(np.array(counterfactual_data2['duration_minutes']),
                            np.array(fitness_data_pid['duration_minutes']))
        perc1 = []
        perc2 = []
        for i in range(len(fitness_data_pid['duration_minutes'])):
            diff_1 = diff1[i] * 100
            diff_2 = diff2[i] * 100
            max_v = np.array(fitness_data_pid['duration_minutes'])[i]
            perc_1 = diff_1 / abs(max_v)
            perc_2 = diff_2 / abs(max_v)
            perc1.append(int(perc_1))
            perc2.append(int(perc_2))
        differences = {
            'more sleep': perc1,
            'less sleep': perc2,
        }
        x = np.arange(len(months))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(20, 10))
        for value, diffs in differences.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, diffs, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_ylabel('%')
        ax.set_title('Percentual difference in duration_minutes depending on hours_sleep')
        ax.set_xticks(x + width, months)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(5) + '-percentual_difference')
    elif pid == 6:
        # PID=6: duration_minutes ==> calories_burned
        fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 6]
        counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: 3},
                                                          observed_data=fitness_data_pid)
        counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                          {'duration_minutes': lambda x: -3},
                                                          observed_data=fitness_data_pid)
        array_plot = np.array([fitness_data_pid['calories_burned'],
                               counterfactual_data1['calories_burned'],
                               counterfactual_data2['calories_burned']])

        df_plot = pd.DataFrame(array_plot, columns=months, index=['regular', 'more', 'less'])
        bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, duration_minutes ==> calories_burned",
                                    figsize=(20, 20))
        fig = bar_plot.get_figure()
        fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(6))
        # re-scaled data
        scaler = StandardScaler().fit(df_plot)  # this
        non_scaled_data = scaler.inverse_transform(df_plot)  # this
        df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
                                  index=['regular', 'more', 'less'])  # this
        ns_bar_plot = df_ns_plot.plot.bar(
            title="Counterfactual outputs: PID=6, duration_minutes ==> calories_burned - non scaled",
            figsize=(20, 20))
        ns_fig = ns_bar_plot.get_figure()
        ns_fig.savefig(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(6) + '-non_scaled')
        diff1 = np.subtract(np.array(counterfactual_data1['calories_burned']),
                            np.array(fitness_data_pid['calories_burned']))
        diff2 = np.subtract(np.array(counterfactual_data2['calories_burned']),
                            np.array(fitness_data_pid['calories_burned']))
        perc1 = []
        perc2 = []
        for i in range(len(fitness_data_pid['calories_burned'])):
            diff_1 = diff1[i] * 100
            diff_2 = diff2[i] * 100
            max_v = np.array(fitness_data_pid['calories_burned'])[i]
            perc_1 = diff_1 / abs(max_v)
            perc_2 = diff_2 / abs(max_v)
            perc1.append(int(perc_1))
            perc2.append(int(perc_2))
        differences = {
            'more time': perc1,
            'less time': perc2,
        }
        x = np.arange(len(months))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(figsize=(20, 10))
        for value, diffs in differences.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, diffs, width, label=value)
            ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_ylabel('%')
        ax.set_title('Percentual difference in calories_burned depending on duration_minutes')
        ax.set_xticks(x + width, months)
        ax.legend(loc='upper left', ncols=2)
        plt.savefig(
            '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(6) + '-percentual_difference')
    # elif pid == 8:
    #     # PID=8: increase duration_minutes
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 8]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 4},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array(
    #         [fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned'],
    #          fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
    #          fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate'],
    #          fitness_data_pid['bmi'], counterfactual_data1['bmi']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['calories before', 'calories after', 'fit level before', 'fit level after',
    #                                   'heart before', 'heart after', 'bmi before', 'bmi after'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=8, increase duration_minutes", figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(8))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['calories before', 'calories after', 'fit level before', 'fit level after',
    #                                      'heart before', 'heart after', 'bmi before', 'bmi after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=8, increase duration_minutes - non scaled",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(8) + '-non_scaled')
    # elif pid == 11:
    #     # PID=11: duration_minutes/daily_steps ==> calories_burned
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 11]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 3,
    #                                                        'daily_steps': lambda x: x + 2},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months, index=['calories_burned before', 'calories_burned after'])
    #     bar_plot = df_plot.plot.bar(
    #         title="Counterfactual outputs: PID=11, daily_steps/duration_minutes ==> calories_burned", figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(11))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['calories_burned before', 'calories_burned after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=11, daily_steps/duration_minutes ==> calories_burned - non scaled",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(11) + '-non_scaled')
    # elif pid == 26:
    #     # PID=26: fitness_level => calories burned
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 26]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'fitness_level': lambda x: x + 2},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['calories_burned'],
    #                            counterfactual_data1['calories_burned']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['calories_burned regularly', 'calories_burned when more fit'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=26, fitness_level ==> calories_burned",
    #                                 figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(26))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['calories_burned regularly', 'calories_burned when more fit'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=26, fitness_level ==> calories_burned - non scaled",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(26) + '-non_scaled')
    # elif pid == 30:
    #     # PID=30: activity_type ==> calories_burned
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 30]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'activity_type': lambda x: 6},  # tennis
    #                                                       observed_data=fitness_data_pid)
    #     counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'activity_type': lambda x: 9},  # yoga
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['calories_burned'],
    #                            counterfactual_data1['calories_burned'],
    #                            counterfactual_data2['calories_burned']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['calories_burned when doing preferred sport', 'calories_burned if playing tennis',
    #                                   'calories_burned if doing yoga'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=30, activity_type ==> calories_burned",
    #                                 figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(30))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['calories_burned when doing preferred sport',
    #                                      'calories_burned if playing tennis',
    #                                      'calories_burned if doing yoga'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=30, activity_type ==> calories_burned",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(30) + '-non_scaled')
    # elif pid == 41:
    #     # PID=41: calories_burned ==> bmi
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 41]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'calories_burned': lambda x: x - 2},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['bmi'], counterfactual_data1['bmi']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months, index=['bmi before', 'bmi after'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=41, calories_burned ==> bmi", figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(41))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['bmi before', 'bmi after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=41, calories_burned ==> bmi",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(41) + '-non_scaled')
    # elif pid == 108:
    #     # PID=108: duration_minutes => blood_pressure/heart_rate
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 165]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 3},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array(
    #         [fitness_data_pid['blood_pressure_systolic'], counterfactual_data1['blood_pressure_systolic'],
    #          fitness_data_pid['blood_pressure_diastolic'], counterfactual_data1['blood_pressure_diastolic'],
    #          fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['blood_pressure_systolic before', 'blood_pressure_systolic after',
    #                                   'blood_pressure_diastolic before', 'blood_pressure_diastolic after',
    #                                   'resting_heart_rate before', 'resting_heart_rate after'])
    #     bar_plot = df_plot.plot.bar(
    #         title="Counterfactual outputs: PID=108, duration_minutes => blood_pressure/heart_rate",
    #         figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(108))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['blood_pressure_systolic before', 'blood_pressure_systolic after',
    #                                      'blood_pressure_diastolic before', 'blood_pressure_diastolic after',
    #                                      'resting_heart_rate before', 'resting_heart_rate after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=108, duration_minutes => blood_pressure/heart_rate",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(108) + '-non_scaled')
    # elif pid == 165:
    #     # PID=165: duration_minutes ==> fitness_level/bmi
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 165]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 3},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level'],
    #                            fitness_data_pid['bmi'], counterfactual_data1['bmi']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['fitness_level before', 'fitness_level after', 'bmi_before', 'bmi_after'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=165, duration_minutes ==> fitness_level/bmi",
    #                                 figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(165))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['fitness_level before', 'fitness_level after', 'bmi_before', 'bmi_after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=165, duration_minutes ==> fitness_level/bmi",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(165) + '-non_scaled')
    # elif pid == 172:
    #     # PID=172: duration_minutes ==> resting_heart_rate
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 172]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 3},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['resting_heart_rate'], counterfactual_data1['resting_heart_rate']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months,
    #                            index=['resting_heart_rate before', 'resting_heart_rate after'])
    #     bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=172, duration_minutes ==> resting_heart_rate",
    #                                 figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(172))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['resting_heart_rate before', 'resting_heart_rate after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=172, duration_minutes ==> resting_heart_rate",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(172) + '-non_scaled')
    # elif pid == 262:
    #     # PID=262: daily_steps ==> fitness_level + duration_minutes ==> fitness_level
    #     fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 262]
    #     counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
    #                                                       {'duration_minutes': lambda x: x + 3,
    #                                                        'daily_steps': lambda x: x + 2},
    #                                                       observed_data=fitness_data_pid)
    #     array_plot = np.array([fitness_data_pid['fitness_level'], counterfactual_data1['fitness_level']])
    #
    #     df_plot = pd.DataFrame(array_plot, columns=months, index=['fitness_level before', 'fitness_level after'])
    #     bar_plot = df_plot.plot.bar(
    #         title="Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> fitness_level", figsize=(20, 20))
    #     fig = bar_plot.get_figure()
    #     fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(262))
    #     # re-scaled data
    #     scaler = StandardScaler().fit(df_plot)  # this
    #     non_scaled_data = scaler.inverse_transform(df_plot)  # this
    #     df_ns_plot = pd.DataFrame(non_scaled_data, columns=months,
    #                               index=['fitness_level before', 'fitness_level after'])  # this
    #     ns_bar_plot = df_ns_plot.plot.bar(
    #         title="Counterfactual outputs: PID=262, daily_steps/duration_minutes ==> fitness_level",
    #         figsize=(20, 20))
    #     ns_fig = ns_bar_plot.get_figure()
    #     ns_fig.savefig(
    #         '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/counterfactual-pid=' + str(262) + '-non_scaled')
    else:
        print('PID ' + str(pid) + ' not found.')
