"""Testes da ablacao controlada de vazamento."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from src.evaluation.leakage_ablation import (
    load_selected_configuration,
    run_leakage_ablation,
)


def _features_df(n_rows: int = 240) -> pd.DataFrame:
    target = ([0, 1, 0, 1, 1, 0] * (n_rows // 6))[:n_rows]
    return pd.DataFrame(
        {
            "Id": range(n_rows),
            "Fim": pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC"),
            "Tag": ["A", "B", "C", "D"] * (n_rows // 4),
            "Classe": ["Operando", "Parado", "Carregando", "Fila"] * (n_rows // 4),
            "Frota": ["F1", "F2"] * (n_rows // 2),
            "Tipo": ["Caminhao", "Escavadeira"] * (n_rows // 2),
            "turno": ["manha", "tarde"] * (n_rows // 2),
            "target_4h": target,
            "n_alertas_24h": [float(v) for v in target],
            "duracao_ciclo_min": [10.0, 15.0, 20.0, 18.0] * (n_rows // 4),
        }
    )


def _write_selection_report(path: Path, model_name: str = "hist_gbdt_optuna") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_model": {
            "model_name": model_name,
            "best_params": json.dumps({"max_iter": 30, "max_leaf_nodes": 7}),
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_selected_configuration_reads_family_and_params(tmp_path: Path) -> None:
    report = _write_selection_report(tmp_path / "selection.json")
    model_name, params = load_selected_configuration(report)
    assert model_name == "hist_gbdt_optuna"
    assert params["max_iter"] == 30


def test_load_selected_configuration_requires_the_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model-selection"):
        load_selected_configuration(tmp_path / "ausente.json")


def test_ablation_compares_both_arms_on_the_same_test_set(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    _features_df().to_parquet(features_path, index=False)
    report = _write_selection_report(tmp_path / "selection.json")
    output = tmp_path / "ablation.json"

    result = run_leakage_ablation(
        features_path=features_path,
        report_path=report,
        output_path=output,
    )

    # A validade da comparacao depende disso.
    assert result["same_test_set"] is True
    assert result["com_vazamento"]["rows_test"] == result["sem_vazamento"]["rows_test"]

    # O embargo so pode encolher treino e validacao, nunca o teste.
    assert result["sem_vazamento"]["rows_train"] < result["com_vazamento"]["rows_train"]
    assert result["sem_vazamento"]["rows_val"] < result["com_vazamento"]["rows_val"]

    assert set(result["custo_do_fix"]) == {
        "test_top15_precision_at_k",
        "test_top15_recall_at_k",
        "test_top15_lift_vs_random",
        "test_auc_pr",
    }
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["model_name"] == "hist_gbdt_optuna"


def test_ablation_requires_the_features_dataset(tmp_path: Path) -> None:
    report = _write_selection_report(tmp_path / "selection.json")
    with pytest.raises(FileNotFoundError, match="tasks.py features"):
        run_leakage_ablation(
            features_path=tmp_path / "ausente.parquet",
            report_path=report,
            output_path=tmp_path / "out.json",
        )
