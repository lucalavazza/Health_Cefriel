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


def convert_to_percentage(value_dictionary):
    total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
    return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}


fitness_data = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
edges = np.load('./graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')

tbr = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level']
var_names = list(fitness_data.columns)
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

scm_causal_model = InvertibleStructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model=scm_causal_model, based_on=fitness_data, quality=AssignmentQuality.BEST)
gcm.fit(causal_model=scm_causal_model, data=fitness_data, return_evaluation_summary=True)

# let's compute the intrinsic causal influence (ICC) of the nodes
for var in var_names:
    iccs_calories = gcm.intrinsic_causal_influence(scm_causal_model, target_node=var)
    perc_iccs_calories = convert_to_percentage(iccs_calories)
    print(var, '=', perc_iccs_calories)
