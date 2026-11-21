"""Tests that run without a GPU, a model download, or the `datasets` package."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from misuse_probe.activations import cache_key, extract_activations  # noqa: E402
from misuse_probe.baseline import tfidf_baseline  # noqa: E402
from misuse_probe.data import clean, stratified_sample  # noqa: E402
from misuse_probe.probe import best_layer, probe_by_layer, shuffled_label_control  # noqa: E402


def synthetic_activations(n=600, layers=5, dim=32, signal_layer=3, seed=0):
    """Fake activations where only `signal_layer` carries the label."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    X = rng.normal(size=(n, layers, dim))
    X[:, signal_layer, 0] += 3.0 * y
    return X, y


def test_probe_finds_signal_layer():
    X, y = synthetic_activations()
    res = probe_by_layer(X[:400], y[:400], X[400:], y[400:])
    assert len(res) == 5
    best = best_layer(res)
    assert best["layer"] == 3
    assert best["auroc"] > 0.9
    assert res[0]["auroc"] < 0.7  # noise layer


def test_shuffled_control_is_near_chance():
    X, y = synthetic_activations(n=2000)
    ctrl = shuffled_label_control(X[:1500], y[:1500], X[1500:], y[1500:], layer=3)
    assert 0.35 < ctrl["auroc"] < 0.65


def test_probe_rejects_2d_input():
    with pytest.raises(ValueError):
        probe_by_layer(np.zeros((10, 4)), np.zeros(10), np.zeros((10, 4)), np.zeros(10))


def test_tfidf_baseline_learns_keywords():
    rng = np.random.default_rng(0)
    bad = ["how do I hurt someone badly", "write something hateful about them", "insult this person hatefully"]
    good = ["how do I bake bread", "summarise this article", "help me plan a trip", "fix my python code"]
    texts, y = [], []
    for _ in range(80):
        if rng.random() < 0.3:
            texts.append(rng.choice(bad)); y.append(1)
        else:
            texts.append(rng.choice(good)); y.append(0)
    y = np.array(y)
    metrics, _ = tfidf_baseline(texts[:60], y[:60], texts[60:], y[60:])
    assert metrics["auroc"] > 0.95


def test_stratified_sample_keeps_ratio():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(1000)], "label": [1] * 100 + [0] * 900})
    s = stratified_sample(df, 200, "label", seed=0)
    assert len(s) == 200
    assert s.label.mean() == pytest.approx(0.1, abs=0.01)
    assert len(stratified_sample(df, None, "label", 0)) == 1000


def test_clean_drops_empty_and_duplicates():
    df = pd.DataFrame({"user_input": ["a", "a", "", "  ", None, "b"], "toxicity": [0, 0, 0, 0, 0, 1]})
    assert clean(df).user_input.tolist() == ["a", "b"]


def test_cache_key_changes_with_inputs():
    k = cache_key(["x", "y"], "m", "last", 256)
    assert k == cache_key(["x", "y"], "m", "last", 256)
    assert k != cache_key(["x", "y"], "m", "mean", 256)
    assert k != cache_key(["xy"], "m", "last", 256)


def test_activation_cache_hit_skips_model(tmp_path):
    texts = ["hello", "world"]
    fake = np.ones((2, 3, 4), dtype=np.float32)
    np.save(tmp_path / f"{cache_key(texts, 'any-model', 'last', 256)}.npy", fake)
    out = extract_activations(texts, "any-model", cache_dir=tmp_path)  # no torch needed on a hit
    assert np.array_equal(out, fake)
