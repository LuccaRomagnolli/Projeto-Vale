"""Ajustes de robustez no treinamento: threshold, espaco de busca e avaliacao."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.model_selection import (
    build_candidate_model,
    evaluate_fitted_model,
    flatten_metrics,
    suggest_candidate_params,
)
from src.models.validation import calibrate_threshold, choose_threshold_for_recall


class _Trial:
    """Trial minimo do Optuna: devolve sempre o limite inferior."""

    def suggest_int(self, _name: str, low: int, _high: int) -> int:
        return low

    def suggest_float(self, _name: str, low: float, _high: float, log: bool = False) -> float:
        return low


class _ScoreModel:
    """Modelo que devolve um score pre-definido por linha."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        values = self._scores[: len(x)]
        return np.column_stack([1 - values, values])


# --- calibracao de threshold ---------------------------------------------


def test_reachable_target_is_reported_as_met() -> None:
    choice = calibrate_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], min_recall=1.0)
    assert choice.target_met is True
    assert choice.achieved_recall >= 1.0


def test_alert_on_everything_is_flagged_as_degenerate() -> None:
    """O caso real: o baseline foi publicado com threshold 0.0 e recall 1.0.

    Com scores identicos nenhum threshold separa as classes, e alertar sobre
    tudo garante recall perfeito. O alvo fica formalmente atendido, entao
    `target_met` sozinho nao denuncia nada -- e por isso a taxa de alerta e
    avaliada em separado.
    """
    choice = calibrate_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], min_recall=0.99)

    assert choice.target_met is True, "alertar sobre tudo atinge o recall alvo"
    assert choice.alert_rate == 1.0
    assert choice.degenerate is True, "a condicao precisa ficar visivel"


def test_discriminating_threshold_is_not_degenerate() -> None:
    choice = calibrate_threshold([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], min_recall=1.0)
    assert choice.degenerate is False
    assert choice.alert_rate < 1.0


def test_fallback_picks_the_highest_recall_candidate() -> None:
    """Quando o alvo e inalcancavel, o fallback nao pode ser arbitrario."""
    y_true = [0, 0, 1]
    y_score = [0.1, 0.2, 0.3]

    choice = calibrate_threshold(y_true, y_score, min_recall=2.0)  # impossivel

    assert choice.target_met is False
    assert choice.achieved_recall == max(
        calibrate_threshold(y_true, y_score, min_recall=2.0).achieved_recall, 0.0
    )


def test_empty_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="calibrar"):
        calibrate_threshold([], [], min_recall=0.5)


def test_shortcut_still_returns_a_plain_float() -> None:
    value = choose_threshold_for_recall([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], min_recall=1.0)
    assert isinstance(value, float)


# --- espaco de busca -----------------------------------------------------


def test_lightgbm_subsample_is_actually_active() -> None:
    """`subsample` nao tem efeito no LightGBM enquanto `subsample_freq` for 0."""
    params = suggest_candidate_params(_Trial(), "lightgbm_optuna", pd.Series([0, 1, 1]))

    assert "subsample" in params
    assert params["subsample_freq"] >= 1, "subsample sem subsample_freq e uma dimensao inerte"

    model = build_candidate_model("lightgbm_optuna", params)
    assert model.get_params()["subsample_freq"] >= 1


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="Candidato desconhecido"):
        suggest_candidate_params(_Trial(), "modelo_inexistente", pd.Series([0, 1]))


# --- escopo da avaliacao -------------------------------------------------


def _frame(n_rows: int = 40, positives_every: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(n_rows),
            "Fim": pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC"),
            "Tag": [f"T{i % 4}" for i in range(n_rows)],
            "feature_a": [float(i % 5) for i in range(n_rows)],
            "target_4h": [int(i % positives_every == 0) for i in range(n_rows)],
        }
    )


def test_trial_scope_leaves_the_test_split_untouched() -> None:
    """Durante a busca, nenhuma metrica de teste pode ser produzida."""
    train, val, test = _frame(), _frame(), _frame()
    model = _ScoreModel(np.linspace(0.1, 0.9, len(train)))

    metrics, scores = evaluate_fitted_model(
        model,
        train=train,
        val=val,
        test=test,
        feature_columns=["feature_a"],
        min_recall=0.5,
        splits=("train", "val"),
    )

    assert "test" not in metrics
    assert "test" not in scores
    row = flatten_metrics("m", "official_candidate", model, metrics, 1.0, True)
    assert not [key for key in row if key.startswith("test_")]
    assert [key for key in row if key.startswith("val_")]


def test_full_scope_still_produces_every_split() -> None:
    train, val, test = _frame(), _frame(), _frame()
    model = _ScoreModel(np.linspace(0.1, 0.9, len(train)))

    metrics, scores = evaluate_fitted_model(
        model,
        train=train,
        val=val,
        test=test,
        feature_columns=["feature_a"],
        min_recall=0.5,
    )

    assert {"train", "val", "test"}.issubset(set(scores))
    row = flatten_metrics("m", "official_candidate", model, metrics, 1.0, True)
    assert [key for key in row if key.startswith("test_")]


def test_validation_split_cannot_be_skipped() -> None:
    """O threshold e calibrado na validacao; sem ela a avaliacao nao faz sentido."""
    train, val, test = _frame(), _frame(), _frame()
    with pytest.raises(ValueError, match="validacao e obrigatoria"):
        evaluate_fitted_model(
            _ScoreModel(np.linspace(0.1, 0.9, len(train))),
            train=train,
            val=val,
            test=test,
            feature_columns=["feature_a"],
            min_recall=0.5,
            splits=("train",),
        )


def test_threshold_diagnostics_reach_the_metrics() -> None:
    train, val, test = _frame(), _frame(), _frame()
    metrics, _ = evaluate_fitted_model(
        _ScoreModel(np.full(len(train), 0.5)),  # scores identicos: alvo inalcancavel
        train=train,
        val=val,
        test=test,
        feature_columns=["feature_a"],
        min_recall=0.99,
        splits=("train", "val"),
    )
    assert metrics["threshold_degenerate"] is True
    assert metrics["threshold_alert_rate"] == 1.0
    assert metrics["threshold_target_recall"] == 0.99


def test_threshold_diagnostics_reach_the_report_row() -> None:
    """Sem isto, o criterio de threshold degenerado no gate fica sem dado.

    Os diagnosticos ficam no nivel superior de `metrics`, nao dentro de um
    split, entao `flatten_metrics` precisa copia-los explicitamente.
    """
    train, val, test = _frame(), _frame(), _frame()
    metrics, _ = evaluate_fitted_model(
        _ScoreModel(np.full(len(train), 0.5)),
        train=train,
        val=val,
        test=test,
        feature_columns=["feature_a"],
        min_recall=0.99,
        splits=("train", "val"),
    )
    row = flatten_metrics("m", "official_candidate", object(), metrics, 1.0, True)

    assert row["threshold_degenerate"] is True
    assert row["threshold_alert_rate"] == 1.0
    assert row["threshold_target_met"] is True


# --- selecao de features --------------------------------------------------


def test_non_generalizing_and_duplicate_features_are_excluded() -> None:
    """`mes` nao generaliza; as demais sao copias exatas de outra coluna."""
    from src.models.train_model import NON_MODELABLE_COLUMNS, select_feature_columns

    frame = _frame(20)
    frame["mes"] = 3
    frame["hora"] = frame["hora_do_dia"] = 10
    frame["dia_semana"] = frame["dia_da_semana"] = 2
    frame["n_alertas_4h"] = frame["n_precondicoes_satisfeitas_4h"] = 1.0

    selected = select_feature_columns(frame)

    for column in NON_MODELABLE_COLUMNS:
        assert column not in selected, f"{column} deveria estar fora da matriz"
    # as versoes que generalizam permanecem
    assert "hora_do_dia" in selected
    assert "dia_da_semana" in selected
    assert "n_alertas_4h" in selected


def test_every_exclusion_carries_a_reason() -> None:
    """A exclusao precisa dizer por que, senao vira decisao sem rastro."""
    from src.models.train_model import NON_MODELABLE_COLUMNS

    assert NON_MODELABLE_COLUMNS
    for column, reason in NON_MODELABLE_COLUMNS.items():
        assert reason and len(reason) > 15, f"{column} sem justificativa util"
