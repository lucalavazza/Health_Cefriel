import networkx as nx
import pandas as pd
import numpy as np
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
from statistics import mean
import warnings
import time

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(7)


def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: round(abs(v) / total_absolute_sum * 100, 2) for k, v in value_dictionary.items()}


start_time = time.time()

fitness_data_training = pd.read_csv(
    './datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
edges = np.load(
    './graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')

tbr = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level']
var_names = list(fitness_data_training.columns)
for r in tbr:
    var_names.remove(r)

nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Let's compute the SCM
print('*** SCM with AssignmentQuality = GOOD\n')
causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
model_perf = gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis,
                                               based_on=fitness_data_training,
                                               quality=AssignmentQuality.GOOD)
print('\n')
# print('\n\n', '-'*50, 'MODEL PERFORMANCE -- from gcm.auto.assign_causal_mechanisms\n\n', model_perf)
fitting = gcm.fit(causal_model=causal_model_for_counterfactual_analysis, data=fitness_data_training,
                  return_evaluation_summary=True)
# print('\n\n', '-'*50, 'EVALUATION SUMMARY -- from gcm.fit\n\n', fitting)

print('\n*** SCM computed.')
print('*** Elapsed Time: ', round(time.time() - start_time, 2), 'seconds')
print('\n', '-'*50, '\n')

# Let's compute the intrinsic causal influence (ICC) of the nodes
total_iccs_mean = {}
for var in var_names:
    iccs_mean = []
    iccs_mean_dict = {}

    print('\n*** Computing ICCs for variable:', var)
    print('*** Elapsed Time: ', round(time.time() - start_time, 2), 'seconds')

    for n_iter in range(10):
        iccs_calories = gcm.intrinsic_causal_influence(causal_model_for_counterfactual_analysis, target_node=var)
        perc_iccs_calories = convert_to_percentage(iccs_calories)
        del_key = []
        iccs_mean.append(perc_iccs_calories)

        for key in perc_iccs_calories.keys():
            iccs_mean_dict[key] = round(mean([d[key] for d in iccs_mean]), 2)
            if iccs_mean_dict[key] <= 0:
                del_key.append(key)
        for key in del_key:
            del iccs_mean_dict[key]

    total_iccs_mean[var] = iccs_mean_dict

for key in total_iccs_mean.keys():
    print(key, ': ', total_iccs_mean[key])

end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print('\n', '-'*50, '\n')
print('*** Computation completed.')
print(f"*** Elapsed Time: {elapsed_time} seconds")
