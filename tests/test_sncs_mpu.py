import networkx as nx
import numpy as np
import pandas as pd

from sncs_mpu import intervention_effects


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
