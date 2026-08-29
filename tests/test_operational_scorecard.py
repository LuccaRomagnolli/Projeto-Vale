"""Scorecard operacional: a metrica sobre a qual a politica de promocao decide.

O modulo define `PRIMARY_TOP_K` e calcula `precision@k`, `recall@k` e `lift@k` --
os numeros que aprovam ou barram a promocao de um modelo. Tinha seis modulos
dependentes e nenhum teste dedicado: era exercitado apenas de forma transitiva,
por testes cuja intencao era outra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.evaluation.operational_scorecard import (
    DEFAULT_TOP_K_VALUES,
    PRIMARY_TOP_K,
    add_segment_context,
    attach_operational_metrics,
    build_scored_frame,
    build_segment_tag_day_panel,
    build_tag_day_panel,
    compute_daily_topk_metrics,
    compute_split_topk_metrics,
    decode_one_hot_prefix,
    primary_topk_by_split,
    safe_divide,
    topk_metrics_by_segment,
)


def _panel_source(
    days: int = 3,
    tags_per_day: int = 6,
    positives_per_day: int = 2,
) -> pd.DataFrame:
    """Ciclos com positivos concentrados nas Tags de maior score.

    Assim o TopK ideal e conhecido de antemao e as metricas podem ser conferidas
    contra um valor calculado a mao.
    """
    rows = []
    identifier = 0
    for day in range(days):
        base = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(day, unit="D")
        for rank in range(tags_per_day):
            rows.append(
                {
                    "Id": identifier,
                    "Tag": f"CA{rank}",
                    "Fim": base + pd.Timedelta(rank, unit="h"),
                    "split": "test",
                    "score": 1.0 - rank * 0.1,
                    "threshold": 0.5,
                    "prediction": int(1.0 - rank * 0.1 >= 0.5),
                    "target_4h": int(rank < positives_per_day),
                }
            )
            identifier += 1
    return pd.DataFrame(rows)


# --- utilitarios ----------------------------------------------------------


def test_safe_divide_returns_zero_instead_of_raising() -> None:
    assert safe_divide(1.0, 0.0) == 0.0
    assert safe_divide(3.0, 6.0) == 0.5


def test_decode_one_hot_recovers_the_original_category() -> None:
    frame = pd.DataFrame({"Frota_F1": [True, False], "Frota_F2": [False, True], "outra": [1, 2]})
    decoded = decode_one_hot_prefix(frame, "Frota")
    assert decoded.tolist() == ["F1", "F2"]


def test_decode_one_hot_without_matching_columns_is_neutral() -> None:
    """Sem colunas do prefixo, o resultado precisa ser explicito, nao um erro."""
    decoded = decode_one_hot_prefix(pd.DataFrame({"x": [1, 2]}), "Frota")
    assert set(decoded.unique()) == {"desconhecido"}


# --- painel Tag-dia -------------------------------------------------------


def test_panel_keeps_one_row_per_tag_day_with_the_max_score() -> None:
    """A unidade executiva e Tag-dia: varios ciclos da mesma Tag viram uma linha."""
    source = _panel_source(days=2, tags_per_day=3)
    duplicated = pd.concat([source, source.assign(score=source["score"] - 0.05)])

    panel = build_tag_day_panel(duplicated)

    assert len(panel) == 2 * 3
    top = panel.sort_values("score", ascending=False).iloc[0]
    assert top["score"] == 1.0, "o painel deve reter o MAIOR score do dia"
    assert int(panel["ciclos"].iloc[0]) == 2


def test_panel_marks_the_day_positive_if_any_cycle_is() -> None:
    source = _panel_source(days=1, tags_per_day=2, positives_per_day=0)
    source.loc[source.index[-1], "target_4h"] = 1
    panel = build_tag_day_panel(source)
    assert int(panel["target_4h"].sum()) == 1


# --- metricas TopK --------------------------------------------------------


def test_topk_metrics_match_a_hand_computed_case() -> None:
    """3 dias, 6 Tags/dia, 2 positivos/dia entre as de maior score.

    Com `top_k=2` o modelo captura todos os positivos: precisao e recall valem
    1.0. A prevalencia Tag-dia e 2/6, logo o lift e 1/(1/3) = 3.
    """
    metrics = compute_daily_topk_metrics(_panel_source(), top_k_values=(2,))
    row = metrics.iloc[0]

    assert row["days"] == 3
    assert row["selected_alerts"] == 6
    assert row["total_positives"] == 6
    assert row["precision_at_k"] == 1.0
    assert row["recall_at_k"] == 1.0
    assert np.isclose(row["tag_day_prevalence"], 1 / 3)
    assert np.isclose(row["lift_vs_random"], 3.0)


def test_precision_falls_and_recall_saturates_as_k_grows() -> None:
    """Ampliar a lista diaria captura tudo, ao custo de precisao."""
    metrics = compute_daily_topk_metrics(_panel_source(), top_k_values=(2, 6)).set_index(
        "top_k_tags_per_day"
    )

    assert metrics.loc[2, "precision_at_k"] > metrics.loc[6, "precision_at_k"]
    assert metrics.loc[6, "recall_at_k"] == 1.0
    assert np.isclose(metrics.loc[6, "lift_vs_random"], 1.0), "selecionar tudo nao tem lift"


def test_alerts_per_day_reflects_the_daily_budget() -> None:
    metrics = compute_daily_topk_metrics(_panel_source(days=4), top_k_values=(3,)).iloc[0]
    assert metrics["alerts_per_day"] == 3.0


def test_empty_split_does_not_crash_the_scorecard() -> None:
    empty = _panel_source().iloc[0:0]
    metrics = compute_daily_topk_metrics(empty, top_k_values=(PRIMARY_TOP_K,)).iloc[0]
    assert metrics["selected_alerts"] == 0
    assert metrics["precision_at_k"] == 0.0
    assert metrics["lift_vs_random"] == 0.0


def test_split_metrics_cover_every_split_present() -> None:
    source = pd.concat(
        [_panel_source(days=2), _panel_source(days=2).assign(split="val")], ignore_index=True
    )
    metrics = compute_split_topk_metrics(source, top_k_values=(2,))
    assert set(metrics["split"]) == {"test", "val"}


def test_primary_topk_uses_the_policy_budget() -> None:
    metrics = compute_daily_topk_metrics(
        _panel_source(days=2, tags_per_day=20), top_k_values=DEFAULT_TOP_K_VALUES
    )
    primary = primary_topk_by_split(metrics, top_k=PRIMARY_TOP_K)
    assert primary["test"]["top_k_tags_per_day"] == PRIMARY_TOP_K


# --- segmentos ------------------------------------------------------------


def test_segment_panel_splits_by_the_requested_column() -> None:
    source = _panel_source(days=2, tags_per_day=4)
    source["Frota"] = ["F1", "F2"] * (len(source) // 2)

    panel = build_segment_tag_day_panel(source, "Frota")

    assert set(panel["Frota"].unique()) == {"F1", "F2"}


def test_rare_segment_is_marked_inconclusive_not_failed() -> None:
    """Segmento raro nao pode ser lido como falha do modelo."""
    source = _panel_source(days=3, tags_per_day=6)
    source["Frota"] = "F_COMUM"
    source.loc[source.index[:2], "Frota"] = "F_RARA"

    metrics = topk_metrics_by_segment(
        source, segment_cols=("Frota",), top_k_values=(2,), min_tag_days=20, min_positives=10
    )
    rara = metrics.loc[metrics["segment_value"] == "F_RARA"].iloc[0]

    assert "inconclusivo" in rara["status"]


def test_segment_context_is_attached_from_the_source_frame() -> None:
    scored = _panel_source(days=1, tags_per_day=3)
    source = scored.assign(Frota_F1=True, Tipo_Caminhao=True)
    enriched = add_segment_context(scored, source)
    assert "Frota" in enriched.columns


# --- integracao com o dicionario de metricas ------------------------------


def test_operational_metrics_are_attached_per_split() -> None:
    scored = _panel_source(days=3, tags_per_day=6)
    metrics = {"test": {"recall": 0.5}}

    enriched = attach_operational_metrics(metrics, scored, top_k_values=(2,), primary_top_k=2)

    assert enriched["test"]["top2_precision_at_k"] == 1.0
    assert enriched["operational_primary_top_k"] == 2
    # a metrica tecnica original nao pode ser perdida
    assert enriched["test"]["recall"] == 0.5


def test_build_scored_frame_carries_threshold_and_prediction() -> None:
    source = _panel_source(days=1, tags_per_day=4)
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    scored = build_scored_frame(source, scores, threshold=0.5, split_name="test")

    assert scored["prediction"].tolist() == [1, 1, 0, 0]
    assert (scored["threshold"] == 0.5).all()
    assert (scored["split"] == "test").all()
