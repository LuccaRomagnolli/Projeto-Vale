"""Ordenacao lexicografica, calibracao de probabilidade e rastreamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.calibration import ProbabilityCalibrator, calibration_error
from src.models.model_selection import (
    PRIMARY_OBJECTIVE_COLUMN,
    primary_objective,
    select_model,
    selection_key,
)
from src.utils.tracking import RunLogger, track_run


def _row(recall: float, precision: float, lift: float, auc: float = 0.5) -> dict[str, float]:
    return {
        "val_top15_recall_at_k": recall,
        "val_top15_precision_at_k": precision,
        "val_top15_lift_vs_random": lift,
        "val_auc_pr": auc,
    }


# --- ordenacao lexicografica ---------------------------------------------


def test_recall_dominates_regardless_of_the_other_metrics() -> None:
    better_recall = _row(0.90, 0.10, 1.0, 0.10)
    worse_recall = _row(0.89, 0.99, 9.9, 0.99)
    assert selection_key(better_recall) > selection_key(worse_recall)


def test_tiny_recall_difference_is_not_swallowed() -> None:
    """O score escalar antigo multiplicava recall por 1e6 e precisao por 1e3.

    Uma diferenca de recall de 1e-4 valia 100 no total, enquanto uma diferenca
    de precisao de 0.5 valia 500 -- entao a precisao vencia, contrariando a
    politica. Com tupla, qualquer diferenca de recall decide primeiro.
    """
    a = _row(0.9001, 0.10, 1.0)
    b = _row(0.9000, 0.99, 1.0)
    assert selection_key(a) > selection_key(b)


def test_unbounded_lift_cannot_invert_the_priority() -> None:
    """`lift` entrava sem escala no score antigo; acima de 1000 inverteria tudo."""
    modest_lift_better_recall = _row(0.90, 0.50, 1.5)
    huge_lift_worse_recall = _row(0.80, 0.50, 5000.0)
    assert selection_key(modest_lift_better_recall) > selection_key(huge_lift_worse_recall)


def test_missing_metric_ranks_below_a_legitimate_zero() -> None:
    """Ausencia de metrica nao pode empatar com um zero medido."""
    measured_zero = _row(0.0, 0.0, 0.0, 0.0)
    missing = {"val_top15_precision_at_k": 0.0}
    assert selection_key(measured_zero) > selection_key(missing)


def test_primary_objective_reads_the_policy_metric() -> None:
    assert primary_objective(_row(0.87, 0.5, 2.0)) == 0.87
    assert PRIMARY_OBJECTIVE_COLUMN == "val_top15_recall_at_k"


def test_select_model_ignores_ineligible_candidates() -> None:
    summary = pd.DataFrame(
        [
            {"model_name": "baseline", "eligible_for_selection": False, **_row(0.99, 0.99, 9.0)},
            {"model_name": "oficial", "eligible_for_selection": True, **_row(0.70, 0.60, 2.0)},
        ]
    )
    assert select_model(summary) == "oficial"


# --- calibracao de probabilidade -----------------------------------------


def _calibration_sample(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=n)
    # scores deslocados para cima: ordenam bem, mas superestimam a probabilidade
    raw = np.clip(y * 0.4 + rng.random(n) * 0.5 + 0.3, 0, 1)
    return y, raw


def test_calibration_preserves_ranking() -> None:
    """Isotonica e monotona, entao as metricas TopK ficam identicas por construcao.

    O que importa e que nenhuma inversao ocorra: ordenando pelo score bruto, o
    score calibrado precisa ser nao-decrescente. Empates sao esperados, ja que
    a isotonica achata faixas.
    """
    y, raw = _calibration_sample()
    calibrated = ProbabilityCalibrator().fit(y, raw).transform(raw)

    by_raw = np.argsort(raw, kind="stable")
    ordered = calibrated[by_raw]
    assert np.all(np.diff(ordered) >= -1e-9), "a calibracao inverteu a ordem de dois itens"


def test_calibration_reduces_the_calibration_error() -> None:
    y, raw = _calibration_sample()
    calibrated = ProbabilityCalibrator().fit(y, raw).transform(raw)
    assert calibration_error(y, calibrated) < calibration_error(y, raw)


def test_calibrator_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="duas classes"):
        ProbabilityCalibrator().fit(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_calibrator_must_be_fitted_before_use() -> None:
    with pytest.raises(ValueError, match="nao ajustado"):
        ProbabilityCalibrator().transform([0.5])


def test_calibrator_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="calibrar"):
        ProbabilityCalibrator().fit([], [])


# --- rastreamento --------------------------------------------------------


def test_tracking_writes_a_run(tmp_path) -> None:
    pytest.importorskip("mlflow")
    with track_run("teste", experiment="teste-vale", tracking_dir=tmp_path / "mlruns") as run:
        assert run.active is True
        run.log_params({"a": 1})
        run.log_metrics({"m": 0.5})
    assert (tmp_path / "mlruns").exists()


def test_tracking_can_be_disabled_by_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING", "0")
    with track_run("teste", tracking_dir=tmp_path / "mlruns") as run:
        assert run.active is False
    assert not (tmp_path / "mlruns").exists()


def test_logger_without_mlflow_is_a_silent_noop() -> None:
    """Falha de rastreamento nao pode invalidar um treino de 90 trials."""
    logger = RunLogger(None)
    assert logger.active is False
    logger.log_params({"a": 1})
    logger.log_metrics({"m": 1.0})
    logger.set_tags({"t": "x"})
    logger.log_artifact("/caminho/inexistente")


def test_logger_swallows_backend_failures() -> None:
    class _Broken:
        def log_params(self, _p):
            raise RuntimeError("backend fora do ar")

        def log_metrics(self, _m):
            raise RuntimeError("backend fora do ar")

        def set_tags(self, _t):
            raise RuntimeError("backend fora do ar")

    logger = RunLogger(_Broken())
    logger.log_params({"a": 1})
    logger.log_metrics({"m": 1.0})
    logger.set_tags({"t": "x"})


def test_logger_only_sends_numeric_metrics() -> None:
    sent: dict = {}

    class _Capture:
        def log_metrics(self, metrics):
            sent.update(metrics)

    RunLogger(_Capture()).log_metrics({"ok": 1.5, "texto": "x", "flag": True, "nulo": None})
    assert sent == {"ok": 1.5}


def test_disabled_tracking_explains_why() -> None:
    """Degradar em silencio foi o que escondeu a mudanca de backend do MLflow."""
    logger = RunLogger(None, reason="mlflow indisponivel")
    assert logger.active is False
    assert logger.reason
