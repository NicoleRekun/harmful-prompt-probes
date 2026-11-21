"""Run the full experiment: TF-IDF baseline vs. per-layer linear probes on ToxicChat.

Usage (GPU recommended, e.g. free Colab T4):
    python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from misuse_probe.activations import extract_activations  # noqa: E402
from misuse_probe.baseline import tfidf_baseline  # noqa: E402
from misuse_probe.data import load_toxicchat  # noqa: E402
from misuse_probe.probe import best_layer, make_probe, probe_by_layer, shuffled_label_control  # noqa: E402


def subset_recall(y, scores, mask, threshold=0.5) -> float | None:
    """Recall on the positives inside `mask` (e.g. jailbreak attempts)."""
    pos = mask & (y == 1)
    if pos.sum() == 0:
        return None
    return float((scores[pos] >= threshold).mean())


def plot(results, baseline, out_path, model):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, [r["auroc"] for r in results], marker="o", label="Linear probe on activations")
    ax.axhline(baseline["auroc"], color="gray", linestyle="--", label="TF-IDF text classifier")
    ax.set_xlabel("Layer (0 = embeddings)")
    ax.set_ylabel("Test AUROC")
    ax.set_title(f"Detecting toxic prompts in ToxicChat\n{model}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, test = load_toxicchat(args.n_train, args.n_test, args.seed)
    y_tr, y_te = train.label.to_numpy(), test.label.to_numpy()
    print(f"train={len(train)} (toxic {y_tr.mean():.1%})  test={len(test)} (toxic {y_te.mean():.1%})")

    base_metrics, base_clf = tfidf_baseline(train.text, y_tr, test.text, y_te, args.seed)
    base_scores = base_clf.predict_proba(list(test.text))[:, 1]
    print("TF-IDF baseline:", base_metrics)

    kw = dict(pooling=args.pooling, batch_size=args.batch_size, max_length=args.max_length)
    X_tr = extract_activations(list(train.text), args.model, **kw)
    X_te = extract_activations(list(test.text), args.model, **kw)

    layers = probe_by_layer(X_tr, y_tr, X_te, y_te, seed=args.seed)
    best = best_layer(layers)
    control = shuffled_label_control(X_tr, y_tr, X_te, y_te, best["layer"], seed=args.seed)
    print("Best probe layer:", best)
    print("Shuffled-label control:", control)

    probe = make_probe(seed=args.seed).fit(X_tr[:, best["layer"]], y_tr)
    probe_scores = probe.predict_proba(X_te[:, best["layer"]])[:, 1]
    jb = test.jailbreaking.to_numpy() == 1

    summary = {
        "model": args.model,
        "pooling": args.pooling,
        "n_train": len(train),
        "n_test": len(test),
        "test_toxic_rate": float(y_te.mean()),
        "n_test_jailbreaks": int((jb & (y_te == 1)).sum()),
        "baseline_tfidf": base_metrics,
        "best_probe": best,
        "shuffled_label_control": control,
        "jailbreak_recall": {
            "baseline_tfidf": subset_recall(y_te, base_scores, jb),
            "best_probe": subset_recall(y_te, probe_scores, jb),
        },
        "probe_by_layer": layers,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot(layers, base_metrics, out / "auroc_by_layer.png", args.model)

    # Disagreements for error analysis (kept out of git: contains raw toxic prompts).
    errs = test.assign(baseline_score=base_scores, probe_score=probe_scores)
    disagree = (base_scores >= 0.5) != (probe_scores >= 0.5)
    errs[disagree].to_csv(out / "disagreements_local_only.csv", index=False)

    print(f"\nSaved {out/'metrics.json'} and {out/'auroc_by_layer.png'}")


if __name__ == "__main__":
    main()
