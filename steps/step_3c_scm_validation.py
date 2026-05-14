"""
Self-contained SCM algebraic-equation validator
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from contextlib import redirect_stdout

from pipeline.config import LABELLED_TEST_DATASET, LINEAR_SCM_DIR, PREPROCESSING_METADATA_PATH
from pipeline.preprocessing import load_preprocessing_metadata


TEST_CSV = LABELLED_TEST_DATASET
EQUATIONS_TXT = LINEAR_SCM_DIR / "algebraic_equations.txt"

# Output location for metrics
OUTPUT_CSV = LINEAR_SCM_DIR / "metrics.csv"
OUTPUT_INTERPRETATION_TXT = LINEAR_SCM_DIR / "metrics_interpretation.txt"
PREPROCESSING_METADATA = load_preprocessing_metadata(PREPROCESSING_METADATA_PATH)\
    if Path(PREPROCESSING_METADATA_PATH).exists() else None

# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Term:
    coef: float
    feature: str

@dataclass
class Equation:
    target: str
    terms: List[Term]


# ============================================================
# Parsing algebraic equations
# ============================================================

_TERM_PATTERN = re.compile(
    r"""
    (?P<coef>[-+]?(?:\d+\.\d+|\d+))
    \s*\*\s*
    (?P<var>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_TARGET_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_CONST_PATTERN = re.compile(
    r"""
    =\s*
    (?P<const>[-+]?(?:\d+\.\d+|\d+))
    \s*(?:\+\s*ε_[A-Za-z_][A-Za-z0-9_]*)?\s*$
    """,
    re.VERBOSE,
)


def parse_equations(path: str | Path) -> Dict[str, Equation]:
    """
    Parse a text file with one equation per line, e.g.:
      bmi = (0.0539*age) + (0.0252*blood_pressure_systolic) + ... + ε_bmi

    Returns: mapping target -> Equation
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Equations file not found: {path.resolve()}")

    equations: Dict[str, Equation] = {}

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in tqdm(lines, desc="Parsing algebraic equations"):
        m = _TARGET_PATTERN.match(line)
        if not m:
            raise ValueError(f"Cannot parse target from line:\n{line}")

        target = m.group(1)
        terms: List[Term] = []

        for tm in _TERM_PATTERN.finditer(line):
            terms.append(Term(float(tm.group("coef")), tm.group("var")))

        if not terms:
            const_match = _CONST_PATTERN.search(line)
            if const_match:
                terms.append(Term(float(const_match.group("const")), "__CONST__"))
            else:
                raise ValueError(f"No RHS terms found for target '{target}'")

        equations[target] = Equation(target, terms)

    return equations


# ============================================================
# One-hot reconstruction
# ============================================================

def infer_onehot_structure(equations: Dict[str, Equation]) -> Dict[str, Set[str]]:
    """
    Detect features of the form "<base>_<level>" referenced in equations,
    treating the suffix as a level only if it is numeric.

    Example:
      intensity_0, intensity_1, intensity_2 -> base 'intensity', levels {'0','1','2'}
    """
    base_to_levels: Dict[str, Set[str]] = {}
    for eq in equations.values():
        for term in eq.terms:
            feat = term.feature
            if feat == "__CONST__":
                continue
            if "_" in feat:
                base, level = feat.rsplit("_", 1)
                if level.isdigit():
                    base_to_levels.setdefault(base, set()).add(level)
    return base_to_levels


def apply_onehot(df: pd.DataFrame, base_to_levels: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    From a raw dataframe containing categorical columns like 'intensity', 'date', etc.,
    create full-rank one-hot columns (no dropped reference), and ensure all levels
    referenced in equations exist (fill with 0 if missing in the dataset).

    If a base column is absent, no dummies are created for that base.
    """
    df_out = df.copy()

    for base, levels in base_to_levels.items():
        if base not in df_out.columns:
            # Either already encoded or base column name differs
            continue

        dummies = pd.get_dummies(df_out[base], prefix=base, prefix_sep="_", dtype=float)

        # Attach generated dummies
        for col in dummies.columns:
            if col not in df_out.columns:
                df_out[col] = dummies[col]

        # Ensure all referenced dummy columns exist (even if level not present in test set)
        for lvl in levels:
            col = f"{base}_{lvl}"
            if col not in df_out.columns:
                df_out[col] = 0.0

    return df_out


# ============================================================
# Prediction & metrics
# ============================================================

def predict(df: pd.DataFrame, eq: Equation) -> np.ndarray:
    """
    Compute y_hat = sum(coef * feature) for a single equation.
    Noise terms are excluded; intercept assumed absent.
    """
    y_hat = np.zeros(len(df), dtype=float)
    missing: List[str] = []

    for term in eq.terms:
        if term.feature == "__CONST__":
            y_hat += term.coef
            continue
        if term.feature not in df.columns:
            missing.append(term.feature)
        else:
            y_hat += term.coef * df[term.feature].to_numpy(dtype=float)

    if missing:
        raise KeyError(
            f"Missing regressors for target '{eq.target}': {missing}\n"
            "Fixes:\n"
            "  • Ensure your dataset columns match equation variable names\n"
            "  • Ensure categorical bases (e.g., 'intensity') exist so one-hot can be generated\n"
            "  • Ensure the category labeling matches the suffixes in the equations\n"
        )

    return y_hat


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def to_original_units(metric_value: float, target: str):
    if PREPROCESSING_METADATA is None:
        return None
    scaling_metadata = PREPROCESSING_METADATA.get("scaled_numeric_columns", {})
    if target not in scaling_metadata:
        return None
    return metric_value * float(scaling_metadata[target]["scale"])


def validate(df_raw: pd.DataFrame, equations: Dict[str, Equation]) -> pd.DataFrame:
    """
    End-to-end validation:
      1) infer one-hot bases/levels implied by equations
      2) create those columns from raw df (full rank)
      3) compute predictions and metrics per target
    """
    base_to_levels = infer_onehot_structure(equations)
    df_encoded = apply_onehot(df_raw, base_to_levels)

    rows = []
    for target, eq in tqdm(equations.items(), desc="Validating equations"):
        if target not in df_raw.columns:
            raise KeyError(f"Target '{target}' not found in dataset columns.")

        y_true = df_raw[target].to_numpy(dtype=float)
        y_pred = predict(df_encoded, eq)
        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        rows.append(
            {
                "target": target,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "rmse_original_units": to_original_units(rmse, target),
                "mae_original_units": to_original_units(mae, target),
                "n": len(y_true),
            }
        )

    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


# ============================================================
# Interpretation (SCM-oriented)
# ============================================================

def interpret(metrics_df: pd.DataFrame) -> None:
    with OUTPUT_INTERPRETATION_TXT.open('w', encoding="utf-8") as f:
        with redirect_stdout(f):

            print("\n=== SCM-Oriented Interpretation ===\n")

            print("\nRMSE — Root Mean Squared Error")
            print("|___ Lower is better, Best interpreted relative to the scale of the variable")
            print("|___ RMSE ≈ 0 → very accurate equation, Large RMSE does not imply causal incorrectness; it may indicate high intrinsic noise")
            print("MAE — Mean Absolute Error")
            print("|___ Lower is better, More robust to outliers than RMSE")
            print("|___ MAE < RMSE → occasional large errors are present, MAE ≈ RMSE → errors are fairly uniform")
            print("R² — Coefficient of Determination")
            print("|___ Higher is better, Bounded roughly between 0 and 1 (can be negative in poor models)")
            print("|___ R² ≥ 0.75 → strong structural determination by parents, 0.30 ≤ R² < 0.75 → partial determination; omitted causes likely, R² < 0.30 → noise-dominated equation\n")

            print("Key takeaway")
            print("RMSE / MAE tell how wrong predictions are.")
            print("R² tells how structurally informative the causal parents are.\n\n\n")

            for _, row in metrics_df.iterrows():
                target = row["target"]
                rmse = float(row["rmse"])
                mae = float(row["mae"])
                r2 = float(row["r2"])

                print(f"Target: {target}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE:  {mae:.6f}")
                print(f"  R²:   {r2:.6f}")
                if pd.notna(row.get("rmse_original_units")):
                    print(f"  RMSE (original units): {float(row['rmse_original_units']):.6f}")
                if pd.notna(row.get("mae_original_units")):
                    print(f"  MAE (original units):  {float(row['mae_original_units']):.6f}")

                if r2 >= 0.75:
                    print("  Interpretation:")
                    print("    • Strong structural equation: parents explain most variation.")
                    print("    • Mechanism is close to deterministic given the model class.")
                elif r2 >= 0.30:
                    print("  Interpretation:")
                    print("    • Moderate equation: parents are informative but not exhaustive.")
                    print("    • Omitted drivers / latent causes likely contribute substantially.")
                else:
                    print("  Interpretation:")
                    print("    • Noise-dominated equation under the current parent set.")
                    print("    • This is common in causal SCMs; low R² is not necessarily a bug.")

                if rmse >= 0.90 and pd.isna(row.get("rmse_original_units")):
                    print("  Scale note:")
                    print("    • RMSE near 1 is consistent with standardized/normalized targets.")

                print("-" * 72)

def main():
    test_path = Path(TEST_CSV)
    eq_path = Path(EQUATIONS_TXT)
    out_path = Path(OUTPUT_CSV)

    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path.resolve()}")
    if not eq_path.exists():
        raise FileNotFoundError(f"Equations TXT not found: {eq_path.resolve()}")

    df_test = pd.read_csv(test_path)
    equations = parse_equations(eq_path)
    results = validate(df_test, equations)
    results.to_csv(out_path, index=False)
    interpret(results)
    print(f"Saved validation metrics to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
