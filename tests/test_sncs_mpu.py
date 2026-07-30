import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sncs_mpu import intervention_effects, paired_gcm_counterfactuals


def test_known_linear_intervention_effect():
    dag = nx.DiGraph([("duration_minutes", "Y")])
    spec = {
        "duration_minutes": {"parents": [], "mean": 0.0},
        "Y": {"parents": ["duration_minutes"], "intercept": 0.0, "coefficients": {"duration_minutes": 2.0}},
    }
    baseline = pd.DataFrame({"duration_minutes": [0.0, 10.0]})
    _, _, effect = intervention_effects(spec, dag, baseline)
    assert np.allclose(effect["Y"], 10.0)


def test_intervention_changes_input_and_downstream_prediction():
    dag = nx.DiGraph([("duration_minutes", "calories_burned")])
    spec = {
        "duration_minutes": {"parents": [], "mean": 0.0},
        "calories_burned": {"parents": ["duration_minutes"], "intercept": 1.0, "coefficients": {"duration_minutes": 3.0}},
    }
    baseline = pd.DataFrame({"duration_minutes": [20.0]})
    base, do, effect = intervention_effects(spec, dag, baseline)
    assert not np.array_equal(baseline.duration_minutes, baseline.duration_minutes + 5)
    assert do.loc[0, "calories_burned"] != base.loc[0, "calories_burned"]
    assert effect.loc[0, "calories_burned"] == 15.0


def test_paired_gcm_counterfactuals_preserve_individual_noise():
    pytest.importorskip("dowhy")
    from dowhy import gcm
    from dowhy.gcm.ml import SklearnRegressionModel
    from sklearn.linear_model import LinearRegression

    graph = nx.DiGraph([("duration_minutes", "outcome")])
    model = gcm.InvertibleStructuralCausalModel(graph)
    model.set_causal_mechanism("duration_minutes", gcm.EmpiricalDistribution())
    model.set_causal_mechanism("outcome", gcm.AdditiveNoiseModel(SklearnRegressionModel(LinearRegression())))

    train = pd.DataFrame(
        {
            "duration_minutes": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "outcome": [0.5, 1.5, 4.5, 5.5, 8.5, 9.5],
        }
    )
    observed = pd.DataFrame(
        {
            "duration_minutes": [0.5, 2.5, 4.5],
            "outcome": [0.2, 5.1, 8.8],
        }
    )
    gcm.fit(model, train)
    baseline, intervened, _ = paired_gcm_counterfactuals(
        model,
        observed,
        {"duration_minutes": lambda x: x + 5.0},
    )

    effect = intervened["outcome"] - baseline["outcome"]
    assert np.allclose(effect.to_numpy(), 10.0, atol=1e-6)
