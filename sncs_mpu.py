"""Minimal, reproducible SN Computer Science causal experiment.

All estimators are trained on participant-disjoint training data.  The module is
also deliberately importable so split and bootstrap invariants can be tested.
"""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

VARIABLES = ["age", "height_cm", "weight_kg", "duration_minutes", "calories_burned",
             "avg_heart_rate", "hours_sleep", "stress_level", "daily_steps",
             "hydration_level", "bmi", "resting_heart_rate",
             "blood_pressure_systolic", "blood_pressure_diastolic", "fitness_level"]
NOMINAL = ["gender", "activity_type", "intensity", "health_condition", "smoking_status"]
SEED, BOOTSTRAPS = 7, 100


def participant_split(df: pd.DataFrame, seed=SEED):
    ids = np.sort(df["participant_id"].unique())
    rng = np.random.RandomState(seed)
    shuffled = ids[rng.permutation(len(ids))]
    n_train = int(round(.8 * len(ids)))
    train, test = np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])
    if set(train) & set(test):
        raise ValueError("train and test participant IDs overlap")
    if len(train) + len(test) != len(ids) or len(train) != n_train:
        raise ValueError("invalid participant split")
    return train, test


def cluster_bootstrap(df: pd.DataFrame, ids: Iterable, seed: int, replicate: int):
    ids = np.asarray(list(ids))
    draw = np.random.RandomState(seed + replicate).choice(ids, size=len(ids), replace=True)
    parts = []
    for draw_index, pid in enumerate(draw):
        block = df[df.participant_id == pid].copy()
        block["bootstrap_draw"] = draw_index
        parts.append(block)
    return pd.concat(parts, ignore_index=True), draw


def _versions(names):
    out = {"python": platform.python_version()}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            out[name] = "unavailable: " + str(exc)
    return out


def _edges_from_graph(graph, names):
    """Return directed and undirected edges from causal-learn graph objects."""
    directed, undirected = [], []
    try:
        for e in graph.get_graph_edges():
            a, b = str(e.get_node1()), str(e.get_node2())
            s = str(e)
            if "-->" in s or "o->" in s:
                directed.append([a, b])
            elif "---" in s or "o-o" in s:
                undirected.append(sorted([a, b]))
    except Exception:
        pass
    return directed, undirected


def deterministic_dag(names, directed, undirected):
    """Acyclic extension retaining directed edges; deterministic and auditable."""
    import networkx as nx
    g = nx.DiGraph(); g.add_nodes_from(names)
    for a, b in directed:
        if a in g and b in g:
            g.add_edge(a, b)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("GES compelled directions contain a cycle")
    for a, b in sorted({tuple(sorted(e)) for e in undirected}):
        candidates = [(a, b), (b, a)]
        added = False
        for x, y in candidates:
            g.add_edge(x, y)
            if nx.is_directed_acyclic_graph(g):
                added = True; break
            g.remove_edge(x, y)
        if not added:
            raise ValueError(f"cannot extend edge {a}-{b} without a cycle")
    return g


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run(input_path="datasets/averaged_health_fitness_dataset.csv", output_dir="artifacts/sncs_mpu"):
    import networkx as nx
    np.random.seed(SEED)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    missing = {c for c in ["participant_id", "date", *VARIABLES] if c not in df}
    if missing: raise ValueError(f"missing required columns: {sorted(missing)}")
    train_ids, test_ids = participant_split(df)
    train = df[df.participant_id.isin(train_ids)].copy()
    test = df[df.participant_id.isin(test_ids)].copy()
    if set(train.participant_id) & set(test.participant_id):
        raise ValueError("train and test participant IDs overlap")
    if set(df.groupby("participant_id").size()) != {12}:
        warnings.warn("not every participant has exactly 12 monthly rows")
    pd.DataFrame({"participant_id": train_ids}).to_csv(out / "train_participant_ids.csv", index=False)
    pd.DataFrame({"participant_id": test_ids}).to_csv(out / "test_participant_ids.csv", index=False)
    _write_json(out / "split_summary.json", {"seed": SEED, "train_participants": len(train_ids),
        "test_participants": len(test_ids), "train_rows": len(train), "test_rows": len(test),
        "participant_id_overlap": len(set(train_ids) & set(test_ids)),
        "months_per_participant": df.groupby("participant_id").size().value_counts().to_dict()})

    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    Xtr = prep.fit_transform(train[VARIABLES]); Xte = prep.transform(test[VARIABLES])
    joblib.dump(prep, out / "preprocessing.joblib")
    np.save(out / "train_matrix.npy", Xtr); np.save(out / "test_matrix.npy", Xte)
    graphs, failures = {}, []
    try:
        from causallearn.search.ConstraintBased.PC import pc
        for alpha in (.01, .05, .10):
            result = pc(Xtr, alpha=alpha, indep_test="fisherz", uc_rule=0, uc_priority=0)
            d, u = _edges_from_graph(result.G, VARIABLES)
            graphs[f"pc_alpha_{alpha:g}"] = {"directed": d, "undirected": u}
            _write_json(out / f"pc_alpha_{alpha:g}.json", graphs[f"pc_alpha_{alpha:g}"])
    except Exception as exc:
        failures.append({"configuration": "pc", "error": repr(exc)})
    try:
        from causallearn.search.ScoreBased.GES import ges
        result = ges(Xtr)
        d, u = _edges_from_graph(result["G"], VARIABLES)
        graphs["ges"] = {"directed": d, "undirected": u}
        _write_json(out / "ges.json", graphs["ges"])
    except Exception as exc:
        failures.append({"configuration": "ges", "error": repr(exc)})
    if "ges" not in graphs:
        raise RuntimeError("GES is required for the primary experiment: " + repr(failures))
    alpha_rows = []
    for key, value in graphs.items():
        if key.startswith("pc_alpha"):
            alpha_rows.append({"configuration": key, "directed_edges": len(value["directed"]),
                               "undirected_edges": len(value["undirected"]),
                               "edges": json.dumps(value["directed"] + value["undirected"], sort_keys=True)})
    pd.DataFrame(alpha_rows).to_csv(out / "pc_alpha_sensitivity.csv", index=False)
    pc05 = graphs.get("pc_alpha_0.05", {"directed": [], "undirected": []})
    pc_edges = {tuple(sorted(e)) for e in pc05["directed"] + pc05["undirected"]}
    ges_edges = {tuple(sorted(e)) for e in graphs["ges"]["directed"] + graphs["ges"]["undirected"]}
    pd.DataFrame([{"pc_configuration": "pc_alpha_0.05", "ges_configuration": "ges",
                    "pc_edges": len(pc_edges), "ges_edges": len(ges_edges),
                    "shared_adjacencies": len(pc_edges & ges_edges),
                    "pc_only": len(pc_edges - ges_edges), "ges_only": len(ges_edges - pc_edges)}]).to_csv(out / "pc_ges_comparison.csv", index=False)
    dag = deterministic_dag(VARIABLES, graphs["ges"]["directed"], graphs["ges"]["undirected"])
    dag_edges = [[a, b] for a, b in dag.edges()]
    _write_json(out / "primary_dag.json", {"algorithm": "GES", "extension": "lexicographic cycle-safe", "edges": dag_edges})
    pd.DataFrame(dag_edges, columns=["source", "target"]).to_csv(out / "primary_dag_edges.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(dag, seed=SEED)
        nx.draw_networkx(dag, pos=pos, node_size=900, font_size=7, arrows=True)
        plt.axis("off"); plt.tight_layout(); plt.savefig(out / "primary_dag.png", dpi=180); plt.close()
    except Exception as exc:
        failures.append({"configuration": "primary_graph_figure", "error": repr(exc)})

    stability = []
    for rep in range(BOOTSTRAPS):
        boot, draw = cluster_bootstrap(train, train_ids, SEED, rep)
        row = {"replicate": rep, "sampled_participants": draw.tolist(), "rows": len(boot)}
        try:
            result = ges(prep.transform(boot[VARIABLES]))
            d, u = _edges_from_graph(result["G"], VARIABLES)
            row["edges"] = d + u
        except Exception as exc:
            row["error"] = repr(exc); failures.append({"replicate": rep, "error": repr(exc)})
        stability.append(row)
    _write_json(out / "bootstrap_stability.json", stability)
    bootstrap_rows = []
    for item in stability:
        for edge in item.get("edges", []):
            bootstrap_rows.append({"replicate": item["replicate"], "source": edge[0], "target": edge[1]})
    bootstrap_detail = pd.DataFrame(bootstrap_rows, columns=["replicate", "source", "target"])
    if len(bootstrap_detail):
        bootstrap_detail = (bootstrap_detail.groupby(["source", "target"], as_index=False)
                            .replicate.nunique().rename(columns={"replicate": "replicates_with_edge"}))
        bootstrap_detail["frequency"] = bootstrap_detail.replicates_with_edge / BOOTSTRAPS
    else:
        bootstrap_detail["replicates_with_edge"] = pd.Series(dtype=int)
        bootstrap_detail["frequency"] = pd.Series(dtype=float)
    bootstrap_detail.to_csv(out / "participant_bootstrap_edge_frequency.csv", index=False)
    _write_json(out / "failures.json", failures)

    # Frozen linear SCM: one OLS mechanism per node, using only training rows.
    scm = {}
    for node in VARIABLES:
        parents = list(dag.predecessors(node))
        if parents:
            model = LinearRegression().fit(train[parents], train[node])
            scm[node] = {"parents": parents, "intercept": float(model.intercept_), "coefficients": dict(zip(parents, model.coef_.tolist()))}
        else:
            scm[node] = {"parents": [], "mean": float(train[node].mean()), "std": float(train[node].std(ddof=1))}
    _write_json(out / "linear_scm.json", scm)
    fit_rows = []
    for node, spec in scm.items():
        if spec["parents"]:
            pred = spec["intercept"] + sum(spec["coefficients"][p] * test[p] for p in spec["parents"])
            residual = train[node] - (spec["intercept"] + sum(spec["coefficients"][p] * train[p] for p in spec["parents"]))
            variance = max(float(np.var(residual, ddof=1)), 1e-12)
            errors = test[node] - pred
        else:
            variance = max(float(spec["std"] ** 2), 1e-12); errors = test[node] - spec["mean"]
        ll = float((-0.5 * (np.log(2 * np.pi * variance) + errors ** 2 / variance)).sum())
        fit_rows.append({"variable": node, "test_log_likelihood": ll, "test_mse": float(np.mean(errors ** 2)),
                         "train_residual_variance": variance})
    fit_table = pd.DataFrame(fit_rows)
    fit_table.to_csv(out / "heldout_structural_fit.csv", index=False)
    _write_json(out / "heldout_structural_fit_summary.json", {"total_test_log_likelihood": float(fit_table.test_log_likelihood.sum()),
        "interpretation": "held-out structural fit of frozen training SCM; not ground-truth causal recovery"})
    descendants = sorted(nx_descendants(dag, "duration_minutes"))
    outcomes = sorted(set(descendants) | {"calories_burned"})
    intervention = test[VARIABLES].copy(); intervention["duration_minutes"] += 5
    observed = test[VARIABLES].reset_index(drop=True); intervention = intervention.reset_index(drop=True)
    pred_obs = _scm_predict(scm, dag, observed); pred_do = _scm_predict(scm, dag, intervention)
    # Fit DoWhy GCM on standardized training observations and keep its fitted
    # mechanisms frozen for held-out intervention evaluation.
    gcm_do_raw = None
    gcm_error = None
    try:
        from dowhy import gcm
        gcm_graph = nx.DiGraph(); gcm_graph.add_nodes_from(VARIABLES); gcm_graph.add_edges_from(dag_edges)
        gcm_model = gcm.ProbabilisticCausalModel(gcm_graph)
        train_std = pd.DataFrame(Xtr, columns=VARIABLES)
        test_std = pd.DataFrame(Xte, columns=VARIABLES)
        gcm.auto.assign_causal_mechanisms(gcm_model, train_std)
        gcm.fit(gcm_model, train_std)
        delta = 5.0 / float(prep.named_steps["scaler"].scale_[VARIABLES.index("duration_minutes")])
        gcm_do = gcm.interventional_samples(gcm_model, test_std,
                    interventions={"duration_minutes": lambda x: x + delta})
        means = prep.named_steps["scaler"].mean_; scales = prep.named_steps["scaler"].scale_
        gcm_do_raw = gcm_do.copy()
        for i, node in enumerate(VARIABLES): gcm_do_raw[node] = gcm_do[node] * scales[i] + means[i]
        joblib.dump(gcm_model, out / "gcm_model.joblib")
    except Exception as exc:
        gcm_error = repr(exc)
        failures.append({"configuration": "gcm", "error": gcm_error})
    rows = []
    for node in outcomes:
        diff = pred_do[node] - pred_obs[node]
        rows.append({"variable": node, "downstream": node in descendants, "mean_effect": float(diff.mean()),
                     "q025": float(diff.quantile(.025)), "q975": float(diff.quantile(.975))})
    pd.DataFrame(rows).to_csv(out / "intervention_effects.csv", index=False)
    agree = []
    for node in outcomes:
        if gcm_do_raw is not None:
            a, b = gcm_do_raw[node], pred_do[node]
            delta = a - b
        else:
            delta = pd.Series(np.nan, index=pred_do.index)
        agree.append({"variable": node, "mean_signed_difference": float(delta.mean()),
                      "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
                      "q025": float(delta.quantile(.025)), "q975": float(delta.quantile(.975))})
    pd.DataFrame(agree).to_csv(out / "counterfactual_agreement.csv", index=False)
    baseline = test.groupby("participant_id").calories_burned.mean()
    chosen = int((baseline - baseline.median()).abs().sort_values().index[0])
    illustrative = test[test.participant_id == chosen].copy(); illustrative["counterfactual_duration_minutes"] = illustrative.duration_minutes + 5
    illustrative.to_csv(out / "illustrative_participant.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4)); plt.plot(illustrative.date, illustrative.calories_burned, marker="o", label="observed")
        plt.xlabel("month"); plt.ylabel("calories_burned (original units)"); plt.title(f"Deterministic test participant {chosen}")
        plt.legend(); plt.tight_layout(); plt.savefig(out / "participant_counterfactual.png", dpi=180); plt.close()
    except Exception as exc:
        failures.append({"configuration": "participant_counterfactual_figure", "error": repr(exc)})

    sha = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = "unknown"
    manifest = {"pipeline": "sncs_mpu", "input": input_path, "input_sha256": sha, "git_commit": commit,
      "seed": SEED, "package_versions": _versions(["numpy", "pandas", "sklearn", "causallearn", "dowhy"]),
      "variables": VARIABLES, "excluded_nominal": NOMINAL, "split": "participant-level 80/20", "bootstrap_replicates": BOOTSTRAPS,
      "primary_graph": "GES with deterministic lexicographic acyclic extension", "intervention": {"duration_minutes": "+5 original minutes"},
      "illustrative_rule": "test participant closest to median baseline calories_burned; ascending ID tie-break",
      "gcm_error": gcm_error, "artifacts": sorted(str(p.relative_to(out)) for p in out.iterdir())}
    _write_json(out / "manifest.json", manifest)
    pd.DataFrame([{**x, "status": "reported"} for x in rows]).to_csv(out / "publication_results.csv", index=False)
    pd.DataFrame([{"item": "data", "value": "averaged monthly observations"},
                  {"item": "split", "value": "participant-level 80/20, seed 7"},
                  {"item": "discovery", "value": "PC Fisher-z alpha 0.01/0.05/0.10 and GES"},
                  {"item": "bootstrap", "value": "100 participant-cluster replicates"},
                  {"item": "evaluation", "value": "held-out structural fit and original-unit intervention agreement"}]).to_csv(out / "methods_results.csv", index=False)
    _write_json(out / "failures.json", failures)


def nx_descendants(graph, node):
    import networkx as nx
    return nx.descendants(graph, node)


def _scm_predict(scm, dag, data):
    import networkx as nx
    out = data.copy()
    for node in nx.topological_sort(dag):
        spec = scm[node]; parents = spec["parents"]
        if parents:
            out[node] = spec["intercept"] + sum(spec["coefficients"][p] * out[p] for p in parents)
        else:
            out[node] = spec["mean"]
    return out


if __name__ == "__main__":
    run()
