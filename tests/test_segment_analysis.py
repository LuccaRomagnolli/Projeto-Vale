import pandas as pd
from src.evaluation.segment_analysis import (
    decode_one_hot_prefix,
    tag_hotspots,
    threshold_metrics_by_segment,
    topk_metrics_by_segment,
)


def _segmented_scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(10),
            "Tag": ["A", "A", "B", "C", "D", "A", "B", "C", "D", "E"],
            "Fim": pd.to_datetime(
                [
                    "2026-01-01 08:00:00+00:00",
                    "2026-01-01 09:00:00+00:00",
                    "2026-01-01 10:00:00+00:00",
                    "2026-01-01 11:00:00+00:00",
                    "2026-01-01 12:00:00+00:00",
                    "2026-01-02 08:00:00+00:00",
                    "2026-01-02 09:00:00+00:00",
                    "2026-01-02 10:00:00+00:00",
                    "2026-01-02 11:00:00+00:00",
                    "2026-01-02 12:00:00+00:00",
                ],
                utc=True,
            ),
            "target_4h": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "split": ["test"] * 10,
            "score": [0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.4, 0.2, 0.75],
            "threshold": [0.5] * 10,
            "prediction": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "Frota": ["F1", "F1", "F1", "F2", "F2", "F1", "F1", "F2", "F2", "F2"],
            "Tipo": ["Caminhao", "Caminhao", "Caminhao", "Escavadeira", "Escavadeira"] * 2,
            "turno": ["manha", "manha", "manha", "tarde", "tarde"] * 2,
            "Classe": ["Operando", "Operando", "Parado", "Operando", "Parado"] * 2,
        }
    )


def test_decode_one_hot_prefix_returns_category_names() -> None:
    df = pd.DataFrame(
        {
            "Frota_A": [True, False, False],
            "Frota_B": [False, True, False],
        }
    )

    decoded = decode_one_hot_prefix(df, "Frota")

    assert decoded.tolist() == ["A", "B", "desconhecido"]


def test_threshold_metrics_by_segment_computes_lift() -> None:
    metrics = threshold_metrics_by_segment(
        _segmented_scored_df(), segment_cols=("Frota",), min_rows=1
    )

    assert {"precision", "recall", "lift_vs_random"}.issubset(metrics.columns)
    assert set(metrics["segment_value"]) == {"F1", "F2"}


def test_topk_metrics_by_segment_uses_tag_day_level() -> None:
    metrics = topk_metrics_by_segment(
        _segmented_scored_df(),
        segment_cols=("Frota",),
        top_k_values=(1,),
        min_tag_days=1,
    )

    assert set(metrics["segment_value"]) == {"F1", "F2"}
    assert (metrics["selected_alerts"] > 0).all()


def test_tag_hotspots_reports_missed_positive_days() -> None:
    hotspots = tag_hotspots(_segmented_scored_df(), top_k=1, min_tag_days=1)

    assert "missed_positive_days" in hotspots.columns
    assert "selected_false_positive_days" in hotspots.columns
