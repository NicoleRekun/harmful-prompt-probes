"""Text-only baseline: TF-IDF features + logistic regression. No language model involved."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .probe import evaluate


def tfidf_baseline(train_texts, y_train, test_texts, y_test, seed: int = 0):
    """Return (metrics, fitted_pipeline)."""
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    clf.fit(list(train_texts), y_train)
    scores = clf.predict_proba(list(test_texts))[:, 1]
    return evaluate(y_test, scores), clf
