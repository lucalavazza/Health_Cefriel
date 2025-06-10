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

drop_cols = ['participant_id', 'date', 'gender', 'height_cm', 'weight_kg', 'activity_type']
participants_considered = 30
drop_rows = range(participants_considered*365+1, len(fit_data))
fit_data = fit_data.drop(drop_rows, axis=0)
fit_data = fit_data.drop(drop_cols, axis=1)

struct_fit_data = fit_data.copy()
non_numeric_columns = list(struct_fit_data.select_dtypes(exclude=[np.number]).columns)
label = LabelEncoder()
for col in non_numeric_columns:
    struct_fit_data[col] = label.fit_transform(struct_fit_data[col])

# this takes a long time to run
structure_model = from_pandas(struct_fit_data)

structure_model.remove_edges_below_threshold(1)

sub_structural_model = structure_model.get_largest_subgraph()

# TODO: add/remove edges and/or nodes
# sub_structural_model.remove_edge('from_edge', 'to_edge')
# sub_structural_model.add_edge('from_edge', 'to_edge')


plot = plot_structure(sub_structural_model, all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
plot.toggle_physics(False)
plot.save_graph('causal_graph.html')

edges = np.array(list(sub_structural_model.edges()))
np.save('edges.npy', edges)
with open('edges.txt', 'w') as f:
    for edge in edges:
        f.write(f"{edge}\n")
