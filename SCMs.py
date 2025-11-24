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

fitness_data_training = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
edges = np.load('./graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')
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
causal_model = InvertibleStructuralCausalModel(G)
model_perf = gcm.auto.assign_causal_mechanisms(causal_model=causal_model,
                                               based_on=fitness_data_training,
                                               quality=AssignmentQuality.GOOD)
fitting = gcm.fit(causal_model=causal_model, data=fitness_data_training,
                  return_evaluation_summary=True)
with open('./linear_scm/scm.txt', 'a') as f:
    f.write('}')
print('*** SCM computed.')

# # Let's compute the intrinsic causal influence (ICC) of the nodes
# total_iccs_mean = {}
# for var in var_names:
#     iccs_mean = []
#     iccs_mean_dict = {}
#
#     print('\n*** Computing ICCs for variable:', var)
#     print('*** Elapsed Time: ', round(time.time() - start_time, 2), 'seconds')
#
#     for n_iter in range(10):
#         iccs_calories = gcm.intrinsic_causal_influence(causal_model, target_node=var)
#         perc_iccs_calories = convert_to_percentage(iccs_calories)
#         del_key = []
#         iccs_mean.append(perc_iccs_calories)
#
#         for key in perc_iccs_calories.keys():
#             iccs_mean_dict[key] = round(mean([d[key] for d in iccs_mean]), 2)
#             if iccs_mean_dict[key] <= 0:
#                 del_key.append(key)
#         for key in del_key:
#             del iccs_mean_dict[key]
#
#     total_iccs_mean[var] = iccs_mean_dict
#
# for key in total_iccs_mean.keys():
#     print(key, ': ', total_iccs_mean[key])

# # Let's compute the arrow strengths
# print('\n*** Computing arrow strengths')
# print('*** Elapsed Time: ', round(time.time() - start_time, 2), 'seconds\n')
# strengths = {}
# for var in var_names:
#     shorter_strength = {}
#     if not is_root_node(G, var):
#         strength = gcm.arrow_strength(causal_model, var)
#         for s in range(len(strength)):
#             shorter_strength[list(strength.keys())[s][0]] = round(list(strength.values())[s], 4)
#         strengths[var] = shorter_strength
#
# for key in strengths.keys():
#
#     my_func = str(key) + ' = '
#
#     for l in range(len(strengths[key])):
#         if l == 0:
#             my_func = my_func + str(list(strengths[key].values())[l]) + '*' + str(list(strengths[key])[l])
#         else:
#             my_func = my_func + ' + ' + str(list(strengths[key].values())[l]) + '*' + str(list(strengths[key])[l])
#
#     my_func += ' + ε'
#
#     # print(key, ': ', strengths[key])
#     print(my_func)
