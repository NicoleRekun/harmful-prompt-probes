"""Train one logistic-regression probe per layer and score it."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, scores >= threshold, zero_division=0)),
    }


def make_probe(C: float = 1.0, seed: int = 0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )


def probe_by_layer(X_train, y_train, X_test, y_test, C: float = 1.0, seed: int = 0) -> list[dict]:
    """X arrays have shape (n, n_layers, dim). Returns one metrics dict per layer."""
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("expected activations of shape (n, n_layers, dim)")
    results = []
    for layer in range(X_train.shape[1]):
        clf = make_probe(C, seed).fit(X_train[:, layer], y_train)
        scores = clf.predict_proba(X_test[:, layer])[:, 1]
        results.append({"layer": layer, **evaluate(y_test, scores)})
    return results


def best_layer(results: list[dict], metric: str = "auroc") -> dict:
    return max(results, key=lambda r: r[metric])


def shuffled_label_control(
    X_train, y_train, X_test, y_test, layer: int, C: float = 1.0, seed: int = 0, n_shuffles: int = 5
) -> dict:
    """Same probe, trained on shuffled labels, averaged over n_shuffles runs.

    Should score near chance (AUROC ~0.5). If it doesn't, the probe is picking up
    something other than the label.
    """
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_shuffles):
        y_shuf = rng.permutation(np.asarray(y_train))
        clf = make_probe(C, seed).fit(X_train[:, layer], y_shuf)
        runs.append(evaluate(y_test, clf.predict_proba(X_test[:, layer])[:, 1]))
    mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    return {"layer": layer, "n_shuffles": n_shuffles, **mean, "auroc_std": float(np.std([r["auroc"] for r in runs]))}
