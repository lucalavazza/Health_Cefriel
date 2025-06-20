import pandas as pd
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.utils import cit
from causallearn.utils.GraphUtils import *
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils import BackgroundKnowledge, SkeletonDiscovery
from causallearn.graph import Graph
from causallearn.utils.cit import CIT, fisherz, chisq, gsq, kci
import time

np.random.seed(42)

# create the directories for the causal graphs
graphs_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/causallearn/graphs'
npy_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/causallearn/edges/npy'
txt_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/causallearn/edges/txt'
try:
    os.mkdir(graphs_dir)
    os.mkdir(npy_dir)
    os.mkdir(txt_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create the directory.")
except Exception as e:
    print(f"An error occurred: {e}")

fit_data = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/regularised_averaged_health_fitness_dataset.csv')
drop_cols = ['participant_id', 'date', 'height_cm', 'weight_kg', 'activity_type']
fit_data = fit_data.drop(drop_cols, axis=1)
var_names = fit_data.columns
fit_data = fit_data.to_numpy()

# Constraint-based Methods
# PC
cits = ['fisherz', 'gsq', 'chisq', 'kci']
for cit in cits:
    if cit not in ['chisq', 'kci']:  # chisq fails after a lot, kci haven't been able to complete it yet (time-wise)
        print('---> PC, alpha=0.05, cit = '+str(cit)+', uc_rule: 0, uc_priority: 0, no bg_knowledge')
        start_pc = time.time()
        cg_pc = pc(fit_data, alpha=0.05, indep_test=cit, uc_rule=0, uc_priority=0)
        print("PC with cit = " + str(cit) + ": " + str((time.time() - start_pc)) + " seconds\n\n")
        pcg_pc = GraphUtils.to_pydot(cg_pc.G, labels=var_names)
        pcg_pc.write_png(graphs_dir + '/causal_graph_causal-learn_pc_' + str(cit) + '.png')

        np_pc_edges = np.array(cg_pc.find_fully_directed())
        np_pc_nodes = []
        for edge in np_pc_edges:
            for node in edge:
                if node not in np_pc_nodes:
                    np_pc_nodes.append(node)
        np_pc_names = var_names.to_numpy()
        np_pc_edges_names = []
        np_pc_nodes_names = []
        for edge in np_pc_edges:
            new_edge = []
            for node in edge:
                new_node = var_names[node-1]
                new_edge.append(new_node)
            np_pc_edges_names.append(new_edge)
        for node in np_pc_nodes:
            np_pc_nodes_names.append(var_names[node-1])
        np_pc_edges_names = np.array(np_pc_edges_names)

        np.save(npy_dir + '/causal_graph_causal-learn_pc_' + str(cit) + '.npy', np_pc_edges_names)
        with open(txt_dir + '/causal_graph_causal-learn_pc_' + str(cit) + '.txt', 'w') as f:
            for edge in np_pc_edges_names:
                f.write(f"{edge}\n")
    else:
        print('---> PC, alpha=0.05, cit = '+str(cit)+' skipped')

    print('---> FCI, alpha=0.05, cit = ' + str(cit) + ', no bg_knowledge')
    start_fci = time.time()
    cg_fci, fci_edges = fci(fit_data, alpha=0.05, indep_test=cit, node_names=var_names)
    print("FCI with cit = " + str(cit) + ": " + str((time.time() - start_fci)) + " seconds\n\n")
    pcg_fci = GraphUtils.to_pydot(cg_fci, labels=var_names)
    pcg_fci.write_png(graphs_dir + '/causal_graph_causal-learn_fci_' + str(cit) + '.png')

    np_fci_edges = []
    for i in range(len(fci_edges)):
        new_edge = [str(fci_edges[i].get_node1()), str(fci_edges[i].get_node2())]
        np_fci_edges.append(new_edge)

    np.save(npy_dir + '/causal_graph_causal-learn_fci_' + str(cit) + '.npy', np_fci_edges)
    with open(txt_dir + '/causal_graph_causal-learn_fci_' + str(cit) + '.txt', 'w') as f:
        for edge in np_fci_edges:
            f.write(f"{edge}\n")

# list_edges = []
# for i in range(len(edges)):
#     edge_list = list(edges[i])
#     edge_list[0] = edge_list[0]+1
#     edge_list[1] = edge_list[1]+1
#     list_edges.append(edge_list)
#
# # # add bmi --> blood_pressure_diastolic
# # list_edges.append([7, 10])
# # # add resting_heart_rate --> blood_pressure_systolic
# # list_edges.append([8, 9])
# # # add smoking_status --> blood_pressure_systolic
# # list_edges.append([12, 9])
# # # remove duration_minutes --> resting_heart_rate
# # list_edges.remove([2, 8])
#
# np_edges = np.array(list_edges)
# np_nodes = []
# for edge in np_edges:
#     for node in edge:
#         if node not in np_nodes:
#             np_nodes.append(node)
# np_names = var_names.to_numpy()
# np_edges_names = []
# np_nodes_names = []
# for edge in np_edges:
#     new_edge = []
#     for node in edge:
#         new_node = var_names[node-1]
#         new_edge.append(new_node)
#     np_edges_names.append(new_edge)
# for node in np_nodes:
#     np_nodes_names.append(var_names[node-1])
# np_edges_names = np.array(np_edges_names)
#
# G = nx.DiGraph()
# G.add_nodes_from(np_nodes_names)
# G.add_edges_from(np_edges_names)
# pcg_pc_bgk = nx.nx_pydot.to_pydot(G)
# pcg_pc_bgk.write_png(graphs_dir + 'causal_graph_causal-learn_pc_alpha05_fisherz_uc0_ucp0_bg_regularised.png')
