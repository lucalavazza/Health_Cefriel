import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(7)

fitness_data_training = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
fitness_data_testing = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv')
edges = np.load('./graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')

nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)


def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis, based_on=fitness_data_training,
                                  quality=AssignmentQuality.GOOD)
gcm.fit(causal_model=causal_model_for_counterfactual_analysis, data=fitness_data_training,
        return_evaluation_summary=True)
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
          'december']

# We want to perform a questioning of the type:
# I want to achieve this, while doing/not doing this and that. How?
# Example: Individual #6 wants to burn more calories while walking a set number of steps daily and only playing tennis.

# Let's set the first desired constraint and see how the data changes.
fitness_data_pid = fitness_data_testing[fitness_data_testing['participant_id'] == 6]
calories = fitness_data_pid['calories_burned']
calories_avg_baseline = np.average(calories)
print('\nCalories burned (baseline): ' + f"{calories_avg_baseline:.2f}")
counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'activity_type': lambda x: 6},  # only play tennis
                                                  observed_data=fitness_data_pid)
array_plot = np.array([fitness_data_pid['calories_burned'], counterfactual_data1['calories_burned']])
df_plot = pd.DataFrame(array_plot, columns=months,
                       index=['calories_burned when doing multiple sports', 'calories_burned if playing only tennis'])
bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, only tennis", figsize=(20, 20))
fig = bar_plot.get_figure()
fig.savefig('./graphs/influences/counterfactual-pid=' + str(6) + '_only tennis.pdf')

# Now we know the baseline from only playing tennis. How does it change when I set the daily steps?
fitness_data_pid = counterfactual_data1
counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'daily_steps': lambda x: 3},
                                                  observed_data=fitness_data_pid)
array_plot = np.array([counterfactual_data1['calories_burned'], counterfactual_data2['calories_burned']])
df_plot = pd.DataFrame(array_plot, columns=months,
                       index=['calories_burned when doing preferred sport and usual steps',
                              'calories_burned if playing tennis and setting a steps limit/goal'])
bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, only tennis and set steps", figsize=(20, 20))
fig = bar_plot.get_figure()
fig.savefig('./graphs/influences/counterfactual-pid=' + str(6) + '_only tennis and set steps.pdf')

# The two constraints above could have been set simultaneously with no difference in the results.
# Done separately for clarity.
# The difference in the code would have been the following:
# counterfactual_data_combined = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
#                                                           {'activity_type': lambda x: 6,  # only play tennis
#                                                            'daily_steps': lambda x: 3},  # set number of steps
#                                                           observed_data=fitness_data_pid)

# Now we know the baseline from only playing tennis and walking a set amount of steps.
# What can we do to get to the objective calorie burn?
# First, let's compute the average calorie burn at this state
calories = np.array(counterfactual_data2['calories_burned'])
calories_avg = np.average(calories)
print('\nCalories burned on average when only playing tennis and walking a set amount of steps daily: '  + f"{calories_avg:.2f}")  # -0.5237279115845814
# Let's say that our objective is to reach -0.2 calories burned on average.
# Next, let's compute the intrinsic causal influence (ICC) of the nodes on the calories_burned node
scm_calories = gcm.StructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(scm_calories, fitness_data_testing)
gcm.fit(scm_calories, fitness_data_testing)
iccs_calories = gcm.intrinsic_causal_influence(scm_calories, target_node='calories_burned')
perc_iccs_calories = convert_to_percentage(iccs_calories)
arr_perc_iccs_calories = np.array(perc_iccs_calories)
plt.figure(figsize=(20, 20))
plt.bar(range(len(perc_iccs_calories)), list(perc_iccs_calories.values()))
plt.xticks(range(len(perc_iccs_calories)), list(perc_iccs_calories.keys()))
plt.savefig('./graphs/influences/iccs_perc_calories_burned.pdf')
plt.close('all')

# Since we cannot manipulate the date, the fitness level and the calories themselves, and since the hours of sleep
# have close to no effect on the calories burned, what's left to do is manipulate the duration minutes.

# 1) We can either set a level we are comfortable with, and find out how close we can get to our objective.
fitness_data_pid = counterfactual_data2
counterfactual_data3 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'duration_minutes': lambda x: x + 1},
                                                  observed_data=fitness_data_pid)
array_plot = np.array([counterfactual_data2['calories_burned'],
                       counterfactual_data3['calories_burned']])
df_plot = pd.DataFrame(array_plot, columns=months,
                       index=['calories_burned with no training increase', 'calories_burned with more training'])
bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, only tennis and set steps + more training",
                            figsize=(20, 20))
fig = bar_plot.get_figure()
fig.savefig('./graphs/influences/counterfactual-pid=' + str(6) +
            '_only tennis and set duration + more training.pdf')
calories = np.array(counterfactual_data3['calories_burned'])
calories_avg = np.average(calories)
print('\nCalories burned on average when setting a comfortable increase in duration:' + f"{calories_avg:.2f}")  # -0.2197595152197581
# We see that the results is close to the objective. We can then either push toward a bit more training, or be satisfied with the results.

# 2) Or we can discover exactly how much more we'd need to train to reach at least the desired level of calories burn.
calories = np.array(counterfactual_data2['calories_burned'])
current_avg = np.average(calories)
target_avg = -0.2
delta = 0
while current_avg < target_avg:
    fitness_data_pid = counterfactual_data2
    counterfactual_data_aim = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                         {'duration_minutes': lambda x: x + delta},
                                                         observed_data=fitness_data_pid)
    current_avg = np.average(np.array(counterfactual_data_aim['calories_burned']))
    if current_avg < target_avg:
        delta += 0.01
print('\nTraining duration increase necessary for reaching at least ' + str(target_avg) + ' calories burned: ' + f"{delta:.2f}")
fitness_data_pid = counterfactual_data2
counterfactual_data4 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'duration_minutes': lambda x: x + delta},
                                                  observed_data=fitness_data_pid)
array_plot = np.array([counterfactual_data2['calories_burned'], counterfactual_data4['calories_burned']])
df_plot = pd.DataFrame(array_plot, columns=months,
                       index=['calories_burned with no training increase', 'calories_burned with exactly more training'])
bar_plot = df_plot.plot.bar(title="Counterfactual outputs: PID=6, only tennis and set steps + exact more training",
                            figsize=(20, 20))
fig = bar_plot.get_figure()
fig.savefig('./graphs/influences/counterfactual-pid=' + str(6) +
            '_only tennis and set duration + exact more training.pdf')
calories = np.array(counterfactual_data4['calories_burned'])
calories_avg = np.average(calories)
print('\nCalories burned on average with predicted duration:' + f"{calories_avg:.2f}")  # -0.16504377569442524
