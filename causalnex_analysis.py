import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dowhy import CausalModel
import numpy as np
import csv
import os
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork

fit_data = pd.read_csv('averaged_health_fitness_dataset.csv')

# create the directories for the causal graphs
graphs_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/'
try:
    os.mkdir(graphs_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create '" + dir_name + "'.")
except Exception as e:
    print(f"An error occurred: {e}")

drop_cols = ['participant_id', 'date', 'height_cm', 'weight_kg', 'activity_type']
participants_considered = 75
drop_rows = range(participants_considered*12+1, len(fit_data))
fit_data = fit_data.drop(drop_rows, axis=0)
fit_data = fit_data.drop(drop_cols, axis=1)

struct_fit_data = fit_data.copy()
non_numeric_columns = list(struct_fit_data.select_dtypes(exclude=[np.number]).columns)
label = LabelEncoder()
for col in non_numeric_columns:
    struct_fit_data[col] = label.fit_transform(struct_fit_data[col])

# this takes a long time to run
structure_model = from_pandas(struct_fit_data, w_threshold=0.8)

sub_structural_model = structure_model.get_largest_subgraph()

# TODO: add/remove pc_edges and/or np_nodes
# sub_structural_model.remove_edge('from_edge', 'to_edge')
# sub_structural_model.add_edge('from_edge', 'to_edge')

plot = plot_structure(sub_structural_model, all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
plot.toggle_physics(False)
plot.save_graph(graphs_dir+'causal_graph_75-rows_causalnex.html')

edges = np.array(list(sub_structural_model.edges()))
np.save(graphs_dir+'causal_graph_75-rows_causalnex_edges.npy', edges)
with open(graphs_dir+'causal_graph_75-rows_causalnex_edges.txt', 'w') as f:
    for edge in edges:
        f.write(f"{edge}\n")
