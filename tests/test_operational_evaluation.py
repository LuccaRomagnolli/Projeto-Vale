import pandas as pd
from src.evaluation.evaluate_model import (
    build_tag_day_panel,
    compute_budget_metrics,
    compute_daily_topk_metrics,
    deduplicate_alerts,
)


def _scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(8),
            "Tag": ["A", "A", "B", "C", "A", "B", "C", "D"],
            "Fim": pd.to_datetime(
                [
                    "2026-01-01 08:00:00+00:00",
                    "2026-01-01 09:00:00+00:00",
                    "2026-01-01 10:00:00+00:00",
                    "2026-01-01 11:00:00+00:00",
                    "2026-01-02 08:00:00+00:00",
                    "2026-01-02 09:00:00+00:00",
                    "2026-01-02 10:00:00+00:00",
                    "2026-01-02 11:00:00+00:00",
                ],
                utc=True,
            ),
            "target_4h": [1, 1, 0, 0, 0, 1, 0, 1],
            "split": ["test"] * 8,
            "score": [0.9, 0.8, 0.2, 0.1, 0.3, 0.85, 0.4, 0.7],
            "threshold": [0.5] * 8,
            "prediction": [1, 1, 0, 0, 0, 1, 0, 1],
        }
    )


def test_compute_budget_metrics_reports_lift() -> None:
    metrics = compute_budget_metrics(_scored_df(), budgets=(0.25,))

    row = metrics.iloc[0]

    assert row["selected_rows"] == 2
    assert row["precision_at_budget"] == 1.0
    assert row["lift_vs_random"] > 1.0


def test_daily_topk_metrics_aggregate_by_tag_day() -> None:
    metrics = compute_daily_topk_metrics(_scored_df(), top_k_values=(1,))

    row = metrics.iloc[0]

    assert row["selected_alerts"] == 2
    assert row["precision_at_k"] == 1.0


def test_build_tag_day_panel_uses_max_score_and_target() -> None:
    panel = build_tag_day_panel(_scored_df())
    row = panel.loc[(panel["Tag"] == "A") & (panel["data"] == pd.Timestamp("2026-01-01").date())]

    assert float(row["score"].iloc[0]) == 0.9
    assert int(row["target_4h"].iloc[0]) == 1


def test_deduplicate_alerts_applies_cooldown_per_tag() -> None:
    dedup = deduplicate_alerts(_scored_df(), cooldown_hours=4)

    assert len(dedup.loc[dedup["Tag"] == "A"]) == 1
    assert len(dedup) == 3
