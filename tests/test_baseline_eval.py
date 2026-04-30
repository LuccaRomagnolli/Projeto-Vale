from __future__ import annotations

import pandas as pd

from projeto_vale.baseline import FrequencyBaseline
from projeto_vale.evaluate import compute_classification_metrics


def test_baseline_and_metrics_runs_end_to_end():
    X_train = pd.DataFrame({"Tag": ["A", "A", "B", "B"]})
    y_train = pd.Series([1, 1, 0, 0])

    X_val = pd.DataFrame({"Tag": ["A", "B", "C"]})
    y_val = pd.Series([1, 0, 0])

    model = FrequencyBaseline(tag_col="Tag").fit(X_train, y_train)
    probs = model.predict_proba(X_val)

    metrics = compute_classification_metrics(y_val, probs, threshold=0.5)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "support" in metrics
