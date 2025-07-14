import dowhy
import networkx as nx
import pandas as pd
import numpy as np
from dowhy import CausalModel
from dowhy import gcm
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel

set_random_seed(42)

fit_data = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset.csv')
edges = np.load(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')

# modification suggested by the falsification step
to_be_removed = np.array(['calories_burned', 'health_condition'])
list_edges = edges.tolist()
list_edges.remove(['calories_burned', 'health_condition'])
edges = np.array(list_edges)

nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)


# Define independence test based on the generalised covariance measure with gradient boosted decision trees as models
def create_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))


def gcm_fal(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)


# STEP 0: Falsification of the Causal Graph: is it informative? Is it rejected?
# Done already and successful: no need to run it every time
# Run evaluation for consensus graph and data.
result = falsify_graph(G, fit_data, n_permutations=100,
                       independence_test=gcm_fal,
                       conditional_independence_test=gcm_fal,
                       plot_histogram=False,
                       suggestions=True)
print(result)

# STEP 1: Causal Effects Estimation: If we change X, how much will it cause Y to change?
# STEP 1.1: Model a causal inference problem using assumptions (i.e., provide data + cg + select Treatment and Outcome)
# STEP 1.2: Identify the causal effect (i.e., the estimand)
# STEP 1.3: Estimate the causal effect
# STEP 1.4: Refute the estimate

model = dowhy.CausalModel(
    data=fit_data,
    graph=G,
    treatment="duration_minutes",
    outcome='calories_burned')

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\nIDENTIFICATION\n")
print(identified_estimand)

estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print("\nESTIMATION\n")
print(estimate)

refute1_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print("\nREFUTATION #1\n")
print(refute1_results)

refute2_results = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter",
                                        subset_fraction=0.8)
print("\nREFUTATION #2\n")
print(refute2_results)

refute3_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                                        placebo_type="permute")
print("\nREFUTATION #3\n")
print(refute3_results)

# STEP 2: What-if questions: What if X had been changed to a different value than its observed value? What would have been the values of other variables?
# STEP 2.1: Simulating the Impact of Interventions: What will happen to the variable Z if I intervene on Y?
causal_model = gcm.ProbabilisticCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model, fit_data)
gcm.fit(causal_model, fit_data)
samples = gcm.interventional_samples(causal_model,
                                     {'duration_minutes': lambda x: x + 1},
                                     num_samples_to_draw=1000)
# At this point, I can inspect the samples for the result

# STEP 2.2: Computing Counterfactuals: I observed a certain outcome z for a variable Z where variable X was set to a
#           value x. What would have happened to the value of Z, had I intervened on X to assign it a different value x'?
