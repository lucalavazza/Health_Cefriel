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

# Choice of the treatment
treatment_col = 'daily_steps'  # TODO: change this, it's temporary for testing purposes
# Choice of the outcome
outcome_col = 'avg_heart_rate'  # TODO: change this, it's temporary for testing purposes

# 1. Create a causal model from the data and given graph
model = CausalModel(
    data=fit_data,
    treatment=treatment_col,
    outcome=outcome_col,
    graph=G)
# 2. Identify the causal effect to be estimated, using properties of the causal graph and return target estimands
identified_estimand = model.identify_effect()
# 3. Estimate the target estimand using a statistical method
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
refute_placebo_treatment = model.refute_estimate(identified_estimand,
                                                 estimate,
                                                 method_name="random_common_cause")

print(model.summary())
print(identified_estimand)
print("Causal Estimate is " + str(estimate.value))
print(refute_placebo_treatment)


