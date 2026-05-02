import numpy as np
import pandas as pd
from src.models.tune_hist_gbdt import (
    build_threshold_curve,
    make_backtest_folds,
    select_best_candidate,
)


def test_select_best_candidate_uses_validation_auc_pr() -> None:
    summary = pd.DataFrame(
        {
            "candidate": ["a", "b"],
            "val_auc_pr": [0.3, 0.4],
            "val_recall": [0.9, 0.7],
            "val_precision": [0.2, 0.1],
        }
    )

    best = select_best_candidate(summary)

    assert best["candidate"] == "b"


def test_build_threshold_curve_reports_false_negatives() -> None:
    curve = build_threshold_curve(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.4, 0.6, 0.9]),
        n_points=4,
    )

    assert {"recall", "precision", "false_positive", "false_negative"}.issubset(curve.columns)
    assert int(curve.iloc[-1]["false_negative"]) >= 1


def test_make_backtest_folds_preserves_temporal_order() -> None:
    df = pd.DataFrame(
        {
            "Fim": pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC"),
            "target_4h": ([0, 1, 0, 0] * 30),
            "feature": range(120),
        }
    )

    folds = make_backtest_folds(df, n_folds=3)

    assert len(folds) == 3
    for train, val, test in folds:
        assert train["Fim"].max() < val["Fim"].min()
        assert val["Fim"].max() < test["Fim"].min()
