import dowhy
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import CausalModel
from dowhy.causal_refuters import refute_placebo_treatment

fit_data = pd.read_csv('health_fitness_dataset.csv')

edges = np.load('edges.npy')
nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

treatment_col = 'daily_steps'
outcome_col = 'fitness_level'
model = CausalModel(
    data=fit_data,
    treatment=treatment_col,
    outcome=outcome_col,
    graph=G)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

method = "backdoor.linear_regression"

desired_effect = "ate"

estimate = model.estimate_effect(
    identified_estimand,
    method_name=method,
    target_units=desired_effect,
    method_params={"weighting_scheme": "ips_weight"})

print("Causal Estimate is " + str(estimate.value))

refute_placebo_treatment = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name='placebo_treatment_refuter',
    placebo_type='permute')

print(refute_placebo_treatment)


