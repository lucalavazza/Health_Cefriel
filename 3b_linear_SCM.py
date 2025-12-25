import statsmodels.api as sm
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from dowhy import gcm
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.auto import AssignmentQuality
from collections import deque
import warnings
import time
import json
import ast
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

start_time = time.time()
set_random_seed(7)

data = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')
data_testing = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv')
edges = np.load('./graphs/causallearn/edges/npy/labelling_causal_graph_causal-learn_pc_fisherz.npy')
nodes = []
for edge in edges:
    for node in edge:
        if node not in nodes:
            nodes.append(node)
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
# Make the directory for the results (if not there already)
OUTPUT_PATH = Path("./linear_scm")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
with open('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/linear_scm/scm_coefficients.json', 'w') as file:
    pass
scm_dict = {}
pids_personas = [2, 5, 6, 8, 11, 26, 30, 41, 108, 165, 172, 262]
CATEGORICAL_VARS = {"activity_type", "intensity", "smoking_status", "gender", "date"}


# Let's compute the SCM
start_time_1 = time.time()
# Cannot use AssignmentQuality = BEST due to compatibility issues with numpy and python 3.8
print('\n*** SCM with AssignmentQuality = BETTER\n')
causal_model = InvertibleStructuralCausalModel(G)
model_perf = gcm.auto.assign_causal_mechanisms(causal_model=causal_model,
                                               based_on=data,
                                               quality=AssignmentQuality.BETTER)
print('\n')
fitting = gcm.fit(causal_model=causal_model, data=data,
                  return_evaluation_summary=True)
with open('./linear_scm/scm.txt', 'a') as f:
    f.write('}')
# Import the SCM computed above and define the categorical variables
with open('linear_scm/scm.txt', 'r') as f:
    SCM = eval(f.read())
print('\n*** SCM computed.\n')
print('*** Elapsed time for SCM computation: ', round(time.time() - start_time_1, 2), 'seconds.\n')
print(50*'-')


# Estimation of linear structural equations (with Ordinary Least Squares (OLS) / Linear Probability Model (LPM))
# for a given structural causal model (SCM) using a given tabular dataset.
start_time_2 = time.time()
print('\n*** Linear SCM computation')
"""
*** OLS REGRESSION ***
    OLS regression estimates the relationship between one or more independent variables (predictors) and a dependent
    variable (response). It accomplishes this by fitting a linear equation to observed data.
    
    Here is what that equation looks like: 
    y = β_0 + β_1*x_1 + ... + β_n*x_n + ε, where
        - y is the dependent variable;
        - x1, x2,…, are independent variables;
        - β_0 is the intercept;
        - β_i are the coefficients, with n>=1;
        - ε represents the error term.
        
    At the core of OLS regression lies an optimization challenge: finding the line (or hyperplane in higher dimensions)
    that best fits the data. But what does "best fit" mean? "Best fit" here means minimizing the sum of
    squared residuals.

*** LPM REGRESSION ***
    The Linear Probability Model (LPM) is a regression model used when the outcome variable Y is binary (i.e. Y∈{0,1}),
    which still uses OLS.
"""
# Detects whether a variable should be modeled as a binary categorical variable rather than a continuous numeric one.
# When estimating linear equations, all dependent variables (and ideally independent ones) must be numeric.
# The function’s job is to identify which of these categorical variables are binary.
# If so, the script treats them specially by encoding them as 0 and 1 — converting them into a Linear Probability Model
# because Linear regression (OLS) cannot handle text labels.
def is_binary_non_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    unique_vals = series.dropna().astype(str).unique()
    return len(unique_vals) == 2


# Creates the matrix of explanatory (independent) variables that will be used in each linear regression model.
# Transforms the list of parent (predictor) variables for each structural equation into a numerical, regression-ready
# matrix — the design matrix X used by OLS.
# Converts the categorical variables into numerical ones.
def build_design_matrix(data: pd.DataFrame,
                        parent_vars: List[str]) -> pd.DataFrame:
    # Extract the predictors
    X = data[parent_vars].copy()
    # Identify categorical variables
    for col in X.columns:
        if (col in CATEGORICAL_VARS) or (data[col].dtype == "object"):
            X[col] = X[col].astype("category")
    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    # Adds a constant 1 to the matrix, required by the OLS method
    X = sm.add_constant(X, has_constant="add")
    return X


# Fit one structural equation in the SCM by regressing the dependent variable on its parent variables.
# It returns:
#   - the fitted model,
#   - a table with coefficient estimates and inference stats,
#   - the number of observations used,
#   - a note describing any binary-encoding performed.
def fit_linear_equation(dataset: pd.DataFrame,
                        dependent_var: str,
                        parent_vars: List[str],
                        robust_cov_type: str = "HC1"):
    outcome = dataset[dependent_var]
    note = ""
    if is_binary_non_numeric(outcome):
        classes = sorted(outcome.dropna().astype(str).unique().tolist())
        mapping = {classes[0]: 0, classes[1]: 1}
        outcome = outcome.astype(str).map(mapping)
        note = f"LPM with encoding: {classes[0]}→0, {classes[1]}→1"
    X = build_design_matrix(dataset, parent_vars)
    # Row alignment and NA handling
    # This ensures the regression is fit on the same rows for all variables (no misalignment).
    aligned = pd.concat([outcome, X], axis=1).dropna()
    y = aligned[dependent_var]
    X_clean = aligned.drop(columns=[dependent_var])
    # Fit the OLS regression
    model = sm.OLS(y, X_clean).fit(cov_type=robust_cov_type)
    # Construct the coefficient table
    coef = model.params.rename("coef").to_frame()
    coef["std_err"] = model.bse
    coef["t"] = model.tvalues
    coef["p_value"] = model.pvalues
    conf_int = model.conf_int()
    coef["ci_lower"] = conf_int[0]
    coef["ci_upper"] = conf_int[1]
    coef["dependent"] = dependent_var
    coef = coef.reset_index().rename(columns={"index": "term"})
    return model, coef, len(aligned), note


# Transforms numerical regression results into a readable algebraic equation,
# explicitly including ALL categories for categorical parents and removing the
# intercept by reparameterizing the model.
def render_equation(coeff_table: pd.DataFrame,
                    dependent_var: str,
                    parent_vars: List[str],
                    dataset: pd.DataFrame,) -> str:
    # Extract intercept
    intercept_row = coeff_table.loc[coeff_table["term"] == "const"]
    intercept = float(intercept_row["coef"].iloc[0]) if not intercept_row.empty else 0.0
    # Map term -> coefficient for convenience
    term_to_coef = dict(zip(coeff_table["term"], coeff_table["coef"]))
    # Identify categorical parents (consistent with build_design_matrix)
    categorical_parents = [
        col for col in parent_vars
        if (col in CATEGORICAL_VARS) or (dataset[col].dtype == "object")
    ]
    # Choose an anchor categorical variable: prefer 'date' if present
    anchor = None
    if "date" in categorical_parents:
        anchor = "date"
    elif categorical_parents:
        anchor = categorical_parents[0]
    terms_expr = []
    # For filtering out categorical dummy terms later
    cat_prefixes = {c: f"{c}_" for c in categorical_parents}
    alpha_beta_term_pairs = []
    # --- 1. Handle categorical variables: build α_{c,l} for all levels l ---
    for c in categorical_parents:
        # Get all observed levels (categories) in a stable order
        cats = dataset[c].astype("category").cat.categories
        if len(cats) == 0:
            continue
        base_level = cats[0]
        other_levels = cats[1:]
        # Base category coefficient:
        #   - For the anchor variable, α_{anchor, base} = intercept
        #   - For non-anchor categorical vars, α_{c, base} = 0
        if c == anchor:
            alpha_base = intercept
        else:
            alpha_base = 0.0
        # Explicit base-level term (even if coefficient is 0)
        terms_expr.append(f"({alpha_base:.4f}*{c}_{base_level})")
        alpha_beta_term_pairs.append((str(c) + '_' + str(base_level), alpha_base))
        # Non-base levels: for each, use the original dummy coefficient
        # term name is how get_dummies names it: "<col>_<level>"
        for level in other_levels:
            dummy_name = f"{c}_{level}"
            beta = float(term_to_coef.get(dummy_name, 0.0))
            if c == anchor:
                # For the anchor variable:
                #   α_{anchor, level} = intercept + β_dummy
                alpha = intercept + beta
            else:
                # For other categorical variables:
                #   α_{c, level} = β_dummy
                alpha = beta
            terms_expr.append(f"({alpha:.4f}*{c}_{level})")
            alpha_beta_term_pairs.append((str(c) + '_' + str(level), alpha))
    # --- 2. Handle non-categorical terms (numeric parents etc.) ---
    # Keep their coefficients as-is; just exclude 'const' and the dummy columns.
    for term, beta in term_to_coef.items():
        if term == "const":
            continue
        if any(term.startswith(prefix) for prefix in cat_prefixes.values()):
            # This term has been "absorbed" into the α_{c,l} above
            continue
        # Regular numeric (or already-encoded) predictor
        terms_expr.append(f"({beta:.4f}*{term})")
        alpha_beta_term_pairs.append((term, beta))
    # --- 3. Assemble the final equation (no separate intercept) ---
    if terms_expr:
        rhs = " + ".join(terms_expr)
    else:
        # Fallback: no predictors; keep intercept if present
        rhs = f"{intercept:.4f}"
    scm_dict[dependent_var] = alpha_beta_term_pairs
    return f"{dependent_var} = {rhs} + ε_{dependent_var}"


metrics_records = []
equations = []
# Iterate through each equation in the SCM
for dep_var, parent_vars in SCM.items():
    if dep_var not in data.columns:
        continue
    # Fit one linear model per dependent variable
    model, coef_tbl, nobs, note = fit_linear_equation(
        data, dep_var, parent_vars, robust_cov_type="HC1"
    )
    # Pass parent_vars and the original dataset so we can reconstruct all categories
    equations.append(
        render_equation(coef_tbl, dep_var, parent_vars, data)
    )
with open('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/linear_scm/scm_coefficients.json', 'a') as file:
    file.write(json.dumps(scm_dict, indent=4))
(OUTPUT_PATH / "algebraic_equations.txt").write_text("\n\n".join(equations))
print("\n" + "\n".join(equations))
print('\n\n*** Elapsed time for Linear SCM computation: ', round(time.time() - start_time_2, 2), 'seconds.\n')
print(50*'-')
print('\n*** Total execution time: ', round(time.time() - start_time, 2), 'seconds.\nthistthese')


# Computing the epsilons
print(50*'-')
print(50*'-')
# First, compute the counterfactuals
fitness_data_pids = {}
counterfactual_data_pids = {}
counterfactual_results_pids = {}
for pid in pids_personas:
    fitness_data_pids[pid] = data_testing[data_testing['participant_id'] == pid]
    counterfactual_data_pids[pid] = gcm.counterfactual_samples(causal_model,
                                                        {'duration_minutes': lambda x: -3},
                                                        observed_data=fitness_data_pids[pid])
    counterfactual_results_pids[pid] = counterfactual_data_pids[pid].to_dict()

with open('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/linear_scm/cf_results.json', 'w') as file:
    file.write(json.dumps(counterfactual_results_pids, indent=4))
print("Counterfactual results exported")

from collections import deque
import pandas as pd

def compute_linear_scm_counterfactuals(
    df_test,
    participant_ids,
    scm_parent_map,
    scm_coefficients,
    intervention,
    cf_results,
    pid_col="participant_id",
    date_col="date",
):
    # ------------------------------------------------------------
    # 1) Topological order over endogenous variables
    # ------------------------------------------------------------
    def topological_order(parent_map):
        variables = list(parent_map.keys())
        in_degree = {v: 0 for v in variables}
        children = {v: [] for v in variables}

        for child, parents in parent_map.items():
            for p in parents:
                if p in in_degree:
                    in_degree[child] += 1
                    children[p].append(child)

        queue = deque([v for v in variables if in_degree[v] == 0])
        order = []

        while queue:
            v = queue.popleft()
            order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)

        return order if len(order) == len(variables) else variables

    causal_order = topological_order(scm_parent_map)

    # ------------------------------------------------------------
    # 2) Term evaluation
    # ------------------------------------------------------------
    def get_term_value(term, row, computed):
        if "_" in term:
            base, level = term.rsplit("_", 1)
            if level.lstrip("-").isdigit():
                base_value = computed.get(base, row.get(base))
                try:
                    return 1.0 if int(base_value) == int(level) else 0.0
                except Exception:
                    return 0.0

        value = computed.get(term, row.get(term, 0.0))
        try:
            if pd.isna(value):
                return 0.0
        except Exception:
            pass

        try:
            return float(value)
        except Exception:
            return 0.0

    # ------------------------------------------------------------
    # 3) Evaluate one structural equation
    # ------------------------------------------------------------
    def evaluate_equation(dep_var, row, computed):
        total = 0.0
        for term, coef in scm_coefficients.get(dep_var, []):
            total += float(coef) * get_term_value(term, row, computed)
        return float(total)

    # ------------------------------------------------------------
    # 4) Main loop (ordered like cf_results)
    # ------------------------------------------------------------
    results = {}

    participant_id_set = {str(x) for x in participant_ids}

    # Iterate PIDs in the same order as cf_results.json
    for pid_str in cf_results.keys():
        if pid_str not in participant_id_set:
            continue

        # Variable order exactly as in cf_results.json for this PID
        allowed_vars_in_order = list(cf_results[pid_str].keys())

        pid_val = int(pid_str) if pid_str.isdigit() else pid_str
        df_pid = df_test[df_test[pid_col] == pid_val].copy()
        if df_pid.empty:
            # Still emit empty PID block (optional); comment out if undesired
            results[pid_str] = {v: {} for v in allowed_vars_in_order}
            continue

        if date_col in df_pid.columns:
            df_pid = df_pid.sort_values(date_col).reset_index(drop=True)
        else:
            df_pid = df_pid.reset_index(drop=True)

        # Pre-create variables in the correct order
        pid_block = {v: {} for v in allowed_vars_in_order}

        for idx, row in df_pid.iterrows():
            computed = {}

            # Apply intervention
            for var, value in intervention.items():
                computed[var] = float(value)

            # Recompute endogenous variables
            for var in causal_order:
                if var in intervention:
                    computed[var] = float(intervention[var])
                else:
                    computed[var] = evaluate_equation(var, row, computed)

            # Combine observed + computed
            new_row = row.to_dict()
            new_row.update(computed)

            idx_key = str(idx)

            # Fill variables in the same order as cf_results
            for var_name in allowed_vars_in_order:
                if var_name in new_row:
                    pid_block[var_name][idx_key] = new_row[var_name]

        results[pid_str] = pid_block

    return results



SCM_TXT_PATH = "./linear_scm/scm.txt"
SCM_COEFS_PATH = "./linear_scm/scm_coefficients.json"
CF_RESULTS_PATH = "./linear_scm/cf_results.json"
with open(SCM_TXT_PATH, "r") as f:
    scm_text = f.read().strip()
my_scm = ast.literal_eval(scm_text)
with open(SCM_COEFS_PATH, "r") as f:
    scm_coefficients = json.load(f)
with open(CF_RESULTS_PATH, "r") as f:
    cf_results = json.load(f)
intervention = {"duration_minutes": -3}

eq_results = compute_linear_scm_counterfactuals(data_testing, pids_personas, my_scm, scm_coefficients, intervention, cf_results, pid_col="participant_id", date_col="date",)

with open("/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/linear_scm/linear_cf_results.json", "w") as f:
    json.dump(eq_results, f, indent=4)

print(50*'-')
print(50*'-')