from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse

# Estimation of linear structural equations (with Ordinary Least Squares (OLS) / Linear Probability Model (LPM))
# for a given structural causal model (SCM) using a tabular dataset.

data = pd.read_csv('./datasets/labelled_regularised_averaged_health_fitness_dataset_training.csv')

OUTPUT_PATH = Path("./linear_scm")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

with open('linear_scm/scm.txt', 'r') as f:
    SCM_PARENTS = eval(f.read())
CATEGORICAL_PARENT_VARS = {"activity_type", "intensity", "smoking_status", "gender"}


# This method ensures that the date column (or any specified column) is converted from a text or datetime format into a
# numeric representation — specifically, an ordinal integer, which counts the number of days since a fixed reference
# (January 1, year 1).
# This transformation is important because:
#   - Linear regression models (OLS/LPM) in statsmodels require all predictor variables (X) to be numeric.
#   - If date were left as a string or datetime object, the regression fitting would fail.
#   - Converting it to an ordinal value allows date to be treated as a continuous variable (e.g., a time trend),
#     which can capture effects like gradual change over time.
def convert_date_to_ordinal_inplace(df: pd.DataFrame, column_name: str = "date") -> None:
    if column_name in df.columns and not pd.api.types.is_numeric_dtype(df[column_name]):
        parsed = pd.to_datetime(df[column_name], errors="coerce")
        df.loc[:, column_name] = parsed.map(lambda d: d.toordinal() if pd.notnull(d) else np.nan)


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
def build_design_matrix(data: pd.DataFrame, parent_vars: List[str]) -> pd.DataFrame:
    # Extract the predictors
    X = data[parent_vars].copy()
    # Identify categorical variables
    for col in X.columns:
        if (col in CATEGORICAL_PARENT_VARS) or (data[col].dtype == "object"):
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
def fit_linear_equation(dataset: pd.DataFrame, dependent_var: str, parent_vars: List[str]):
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
    model = sm.OLS(y, X_clean).fit()
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


# Transforms numerical regression results into a readable algebraic equation
def render_equation(coeff_table: pd.DataFrame, dependent_var: str) -> str:
    # Identify the intercept, i.e. the starting value of the dependent variable when all predictors are set to zero
    intercept_row = coeff_table.loc[coeff_table["term"] == "const"]
    intercept = intercept_row["coef"].iloc[0] if not intercept_row.empty else 0.0
    terms = []
    # Iterate through predictor terms
    for _, row in coeff_table.loc[coeff_table["term"] != "const", ["term", "coef"]].iterrows():
        terms.append(f"({row.coef:.2f}*{row.term})")
    # Assemble the right-hand side of the equation
    rhs_non_zero_intercept = (" + " + " + ".join(terms)) if terms else ""
    rhs_zero_intercept = (" + ".join(terms)) if terms else ""
    # Construct the final readable equation
    if -0.0001 < intercept < 0.0001:
        return f"{dependent_var} = {rhs_zero_intercept} + ε"
    else:
        return f"{dependent_var} = {intercept:.2f}{rhs_non_zero_intercept} + ε"


convert_date_to_ordinal_inplace(data, column_name="date")
metrics_records = []
equations = []
# Iterate through each equation in the SCM
for dep_var, parent_vars in SCM_PARENTS.items():
    if dep_var not in data.columns:
        continue
    # Fit one linear model per dependent variable
    model, coef_tbl, nobs, note = fit_linear_equation(data, dep_var, parent_vars)
    equations.append(render_equation(coef_tbl, dep_var))
# Aggregate all results
(OUTPUT_PATH / "algebraic_equations.txt").write_text("\n\n".join(equations))

print("\n".join(equations))
for p in sorted(OUTPUT_PATH.glob("coefficients_*.csv")):
    print(f" - {p.name}")
