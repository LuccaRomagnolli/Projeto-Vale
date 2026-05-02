import pandas as pd
from src.models.validation import (
    choose_threshold_for_recall,
    compute_binary_metrics,
    temporal_train_val_test_split,
)


def test_temporal_train_val_test_split_preserves_time_order() -> None:
    df = pd.DataFrame(
        {
            "Fim": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
            "target_4h": [0, 1] * 50,
        }
    )

    train, val, test, metadata = temporal_train_val_test_split(df)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train["Fim"].max() < val["Fim"].min()
    assert val["Fim"].max() < test["Fim"].min()
    assert metadata["rows_total"] == 100


def test_compute_binary_metrics_handles_basic_case() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        threshold=0.5,
    )
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["auc_pr"] == 1.0


def test_choose_threshold_for_recall_returns_candidate() -> None:
    threshold = choose_threshold_for_recall(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        min_recall=1.0,
    )
    assert 0.0 <= threshold <= 1.0
