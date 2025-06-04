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

fit_data = pd.read_csv('health_fitness_dataset.csv')

drop_cols = ['participant_id', 'date', 'gender', 'height_cm', 'weight_kg', 'activity_type', 'health_condition', 'stress_level']
participants_considered = 3
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
sub_structural_model.remove_edge('duration_minutes', 'daily_steps')
sub_structural_model.remove_edge('intensity', 'daily_steps')
sub_structural_model.remove_edge('hours_sleep', 'duration_minutes')
sub_structural_model.remove_edge('hydration_level', 'daily_steps')
sub_structural_model.remove_edge('blood_pressure_diastolic', 'duration_minutes')
sub_structural_model.remove_edge('smoking_status', 'calories_burned')
sub_structural_model.remove_edge('smoking_status', 'daily_steps')

sub_structural_model.remove_edge('calories_burned', 'duration_minutes')
sub_structural_model.add_edge('duration_minutes', 'calories_burned')
sub_structural_model.remove_edge('calories_burned', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'calories_burned')
sub_structural_model.remove_edge('avg_heart_rate', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'avg_heart_rate')
sub_structural_model.remove_edge('hours_sleep', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'hours_sleep')
sub_structural_model.remove_edge('bmi', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'bmi')
sub_structural_model.remove_edge('resting_heart_rate', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'resting_heart_rate')
sub_structural_model.remove_edge('blood_pressure_systolic', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'blood_pressure_systolic')
sub_structural_model.add_edge('age', 'blood_pressure_systolic')
sub_structural_model.remove_edge('blood_pressure_diastolic', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'blood_pressure_diastolic')
sub_structural_model.remove_edge('blood_pressure_diastolic', 'age')
sub_structural_model.add_edge('age', 'blood_pressure_diastolic')
sub_structural_model.remove_edge('fitness_level', 'daily_steps')
sub_structural_model.add_edge('daily_steps', 'fitness_level')

plot = plot_structure(sub_structural_model, all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
plot.toggle_physics(False)
plot.save_graph('causal_graph.html')

edges = np.array(list(sub_structural_model.edges()))
np.save('edges.npy', edges)
with open('edges.txt', 'w') as f:
    for edge in edges:
        f.write(f"{edge}\n")
