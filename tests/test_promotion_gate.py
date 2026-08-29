"""A politica de promocao precisa barrar de verdade, nao so registrar."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.models.promotion_gate import (
    MIN_TEST_LIFT_VS_RANDOM,
    MIN_TEST_PRECISION_AT_K,
    MIN_TEST_RECALL_AT_K,
    evaluate_performance_criteria,
    load_selected_metrics,
    main,
    run_promotion_gate,
)


def _selection_report(path: Path, **overrides: float) -> Path:
    metrics = {
        "test_top15_precision_at_k": 0.81,
        "test_top15_recall_at_k": 0.89,
        "test_top15_lift_vs_random": 2.51,
    }
    metrics.update(overrides)
    path.write_text(
        json.dumps({"selected_model": {"model_name": "hist_gbdt_optuna", **metrics}}),
        encoding="utf-8",
    )
    return path


def _backtest(path: Path, recall_std_high: bool = False) -> Path:
    recalls = [0.60, 0.90, 0.75] if recall_std_high else [0.89, 0.89, 0.90]
    pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "test_top15_recall_at_k": recalls,
            "test_top15_precision_at_k": [0.81, 0.82, 0.80],
        }
    ).to_csv(path, index=False)
    return path


def _run(tmp_path: Path, **overrides: float) -> dict:
    return run_promotion_gate(
        report_path=_selection_report(tmp_path / "selection.json", **overrides),
        backtest_path=_backtest(tmp_path / "backtest.csv"),
        output_path=tmp_path / "promotion.json",
    )


def test_compliant_model_is_approved(tmp_path: Path) -> None:
    verdict = _run(tmp_path)
    assert verdict["aprovado"] is True
    assert verdict["criterios_reprovados"] == []
    assert len(verdict["criterios"]) == 5


def test_precision_below_the_floor_blocks_promotion(tmp_path: Path) -> None:
    """O caso central: 0.59 esta abaixo do piso de 0.60 da politica."""
    verdict = _run(tmp_path, test_top15_precision_at_k=0.59)
    assert verdict["aprovado"] is False
    assert verdict["criterios_reprovados"] == ["test_top15_precision_at_k"]


def test_recall_below_the_floor_blocks_promotion(tmp_path: Path) -> None:
    verdict = _run(tmp_path, test_top15_recall_at_k=0.69)
    assert verdict["aprovado"] is False
    assert "test_top15_recall_at_k" in verdict["criterios_reprovados"]


def test_lift_below_the_floor_blocks_promotion(tmp_path: Path) -> None:
    verdict = _run(tmp_path, test_top15_lift_vs_random=1.89)
    assert verdict["aprovado"] is False
    assert "test_top15_lift_vs_random" in verdict["criterios_reprovados"]


def test_value_exactly_on_the_floor_is_approved(tmp_path: Path) -> None:
    """A politica diz `>=`, entao o limite exato passa."""
    verdict = _run(
        tmp_path,
        test_top15_precision_at_k=MIN_TEST_PRECISION_AT_K,
        test_top15_recall_at_k=MIN_TEST_RECALL_AT_K,
        test_top15_lift_vs_random=MIN_TEST_LIFT_VS_RANDOM,
    )
    assert verdict["aprovado"] is True


def test_unstable_backtest_blocks_promotion(tmp_path: Path) -> None:
    verdict = run_promotion_gate(
        report_path=_selection_report(tmp_path / "selection.json"),
        backtest_path=_backtest(tmp_path / "backtest.csv", recall_std_high=True),
        output_path=tmp_path / "promotion.json",
    )
    assert verdict["aprovado"] is False
    assert "std_test_recall_at_k" in verdict["criterios_reprovados"]


def test_missing_metric_blocks_instead_of_assuming_zero(tmp_path: Path) -> None:
    """Metrica ausente nao pode virar aprovacao silenciosa nem 0.0 implicito."""
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"selected_model": {"model_name": "m", "test_top15_recall_at_k": 0.9}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metricas exigidas"):
        evaluate_performance_criteria(load_selected_metrics(path))


def test_missing_selection_report_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model-selection"):
        load_selected_metrics(tmp_path / "ausente.json")


def test_report_without_selected_model_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_model": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="model_name"):
        load_selected_metrics(path)


def test_main_exits_nonzero_when_blocked(tmp_path: Path, monkeypatch) -> None:
    """O CI so barra a promocao se o processo terminar diferente de zero."""
    monkeypatch.setattr(
        "src.models.promotion_gate.SELECTION_REPORT_JSON",
        _selection_report(tmp_path / "selection.json", test_top15_precision_at_k=0.10),
    )
    monkeypatch.setattr(
        "src.models.promotion_gate.BACKTEST_REPORT_CSV", _backtest(tmp_path / "backtest.csv")
    )
    monkeypatch.setattr(
        "src.models.promotion_gate.PROMOTION_REPORT_JSON", tmp_path / "promotion.json"
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert "test_top15_precision_at_k" in str(exc.value)


def test_verdict_is_persisted_for_audit(tmp_path: Path) -> None:
    verdict = _run(tmp_path)
    saved = json.loads((tmp_path / "promotion.json").read_text(encoding="utf-8"))
    assert saved["aprovado"] == verdict["aprovado"]
    assert saved["politica"] == "docs/politica_promocao_modelo.md"
