import dowhy
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import CausalModel
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.utils import plot, bar_plot
from sklearn.ensemble import GradientBoostingRegressor
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel
from dowhy.gcm.auto import AssignmentQuality
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

set_random_seed(42)

fitness_data = pd.read_csv(
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
# Run evaluation for graph and data.
result = falsify_graph(G, fitness_data, n_permutations=100,
                       independence_test=gcm_fal,
                       conditional_independence_test=gcm_fal,
                       plot_histogram=False,
                       suggestions=True)
print(result)


# STEP 1: Causal Effects Estimation: If we change X, how much will it cause Y to change?

# STEP 1.1: Model a causal inference problem using assumptions (i.e., provide data + cg + select Treatment and Outcome)
model = dowhy.CausalModel(
    data=fitness_data,
    graph=G,
    treatment="duration_minutes",
    outcome='calories_burned')
# STEP 1.2: Identify the causal effect (i.e., the estimand)
print("\nIDENTIFICATION\n")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
# STEP 1.3: Estimate the causal effect
print("\nESTIMATION - backdoor\n")
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print(estimate)
# STEP 1.4: Refute the estimate
refute1_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                                        show_progress_bar=True, placebo_type="permute")
print("\nREFUTATION #1: placebo treatment (effect should go to zero)\n")
print(refute1_results)
refute2_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause",
                                        show_progress_bar=True)
print("\nREFUTATION #2: random common causa (effect should be the same)\n")
print(refute2_results)
refute3_results = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter",
                                        show_progress_bar=True, subset_fraction=0.8)
print("\nREFUTATION #3: random common causa (effect should be the same)\n")
print(refute3_results)


# STEP 2: What-if questions: What if X had been changed to a different value than its observed value? What would have been the values of other variables?
# STEP 2.1: Simulating the Impact of Interventions: What will happen to the variable Z if I intervene on Y?
causal_model = gcm.ProbabilisticCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model, fitness_data)
gcm.fit(causal_model, fitness_data)

median_mean_latencies, uncertainty_mean_latencies = gcm.confidence_intervals(
    lambda: gcm.fit_and_compute(gcm.interventional_samples,
                                causal_model,
                                fitness_data,
                                interventions={
                                    'duration_minutes': lambda x: x + 1},
                                observed_data=fitness_data)().mean().to_dict(),
    num_bootstrap_resamples=10)
avg_calories_burned_before = fitness_data.mean().to_dict()['calories_burned']

bar_plot(dict(before=avg_calories_burned_before, after=median_mean_latencies['calories_burned']),
         dict(before=np.array([avg_calories_burned_before, avg_calories_burned_before]),
              after=uncertainty_mean_latencies['calories_burned']),
         filename='/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Intervention-duration_minutes->calories_burned',
         ylabel='Avg. Calories Burned',
         display_plot=False,
         figure_size=(15, 15),
         bar_width=0.4,
         xticks=['Before', 'After'],
         xticks_rotation=45)


# STEP 3: Computing counterfactuals: I observed a certain outcome z for a variable Z where variable X was set to a value x.
# What would have happened to the value of Z, had I intervened on X to assign it a different value x'?
# As an example, I want to check what happens to the calories_burned of participant_id=42 if they do not train enough or too much.

causal_model_for_counterfactual_analysis = InvertibleStructuralCausalModel(G)
gcm.auto.assign_causal_mechanisms(causal_model=causal_model_for_counterfactual_analysis, based_on=fitness_data,
                                  quality=AssignmentQuality.BEST)
gcm.fit(causal_model_for_counterfactual_analysis, fitness_data)

fitness_data_42 = fitness_data[fitness_data['participant_id'] == 42]
counterfactual_data1 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'duration_minutes': lambda x: -3},
                                                  observed_data=fitness_data_42)
counterfactual_data2 = gcm.counterfactual_samples(causal_model_for_counterfactual_analysis,
                                                  {'duration_minutes': lambda x: 3},
                                                  observed_data=fitness_data_42)


array_plot = np.array([fitness_data_42['calories_burned'], counterfactual_data1['calories_burned'], counterfactual_data2['calories_burned']])
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
df_plot = pd.DataFrame(array_plot, columns=months, index=['regular', 'lack_of', 'too_much'])

bar_plot = df_plot.plot.bar(title="Counterfactual outputs", figsize=(17, 17))
plt.ylabel('Calories Burned')
fig = bar_plot.get_figure()
fig.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/Counterfactual-duration_minutes->calories_burned-pid=42')
