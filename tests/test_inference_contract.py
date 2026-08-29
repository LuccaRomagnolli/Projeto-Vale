"""Contrato de entrada da inferencia: o lote ruim precisa falhar, nao pontuar."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from src.features.encoders import CategoricalEncoder, save_encoder
from src.inference import align_feature_schema, apply_categorical_encoder, run_inference
from src.inference_contract import (
    InferenceContractError,
    check_feature_coverage,
    validate_batch_values,
)

FEATURE_COLUMNS = ["n_alertas_4h", "duracao_ciclo_min", "Tag_freq"]


class _FakeModel:
    """Modelo minimo: score proporcional a primeira coluna."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.clip(x.iloc[:, 0].to_numpy(dtype=float) / 10.0, 0, 1)
        return np.column_stack([1 - positive, positive])


def _batch(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tag": [f"T{i % 3}" for i in range(n_rows)],
            "Fim": pd.date_range("2026-03-01", periods=n_rows, freq="h", tz="UTC"),
            "n_alertas_4h": [float(i % 4) for i in range(n_rows)],
            "duracao_ciclo_min": [12.0] * n_rows,
            "dias_desde_ultimo_alerta": [1.5] * n_rows,
        }
    )


# --- camada de valores ---------------------------------------------------


def test_batch_without_context_columns_is_rejected() -> None:
    batch = _batch().drop(columns=["Tag"])
    with pytest.raises(InferenceContractError, match="contexto obrigatorias"):
        validate_batch_values(batch)


def test_batch_with_null_tag_is_rejected() -> None:
    batch = _batch()
    batch.loc[0, "Tag"] = None
    with pytest.raises(InferenceContractError, match="reprovado"):
        validate_batch_values(batch)


def test_negative_cycle_duration_is_rejected() -> None:
    batch = _batch()
    batch.loc[2, "duracao_ciclo_min"] = -5.0
    with pytest.raises(InferenceContractError, match="reprovado"):
        validate_batch_values(batch)


def test_epoch_scale_days_since_alert_is_rejected() -> None:
    """Pega exatamente o defeito de resolucao temporal encontrado na Fase 2.

    O dataset corrompido tinha media de 20159 dias em
    `dias_desde_ultimo_alerta` -- dias desde a epoca Unix. O contrato agora
    barra esse lote em vez de pontua-lo.
    """
    batch = _batch()
    batch["dias_desde_ultimo_alerta"] = 20159.0
    with pytest.raises(InferenceContractError, match="resolucao temporal"):
        validate_batch_values(batch)


def test_valid_batch_passes_and_keeps_rows() -> None:
    batch = _batch()
    validated = validate_batch_values(batch)
    assert len(validated) == len(batch)


# --- camada estrutural ---------------------------------------------------


def test_missing_feature_fails_loudly_by_default() -> None:
    batch = _batch().drop(columns=["duracao_ciclo_min"])
    with pytest.raises(InferenceContractError, match="ranking sem sentido"):
        check_feature_coverage(batch, FEATURE_COLUMNS)


def test_missing_feature_can_be_allowed_explicitly() -> None:
    batch = _batch().drop(columns=["duracao_ciclo_min"])
    coverage = check_feature_coverage(batch, FEATURE_COLUMNS, allow_missing=True)
    assert coverage["missing_feature_columns"] == ["duracao_ciclo_min", "Tag_freq"]


def test_extra_columns_are_tolerated_and_reported() -> None:
    batch = _batch()
    batch["coluna_extra"] = 1
    batch["Tag_freq"] = 0.1
    coverage = check_feature_coverage(batch, FEATURE_COLUMNS)
    assert "coluna_extra" in coverage["extra_columns_ignored"]
    assert coverage["missing_feature_columns"] == []


def test_align_feature_schema_no_longer_fills_silently() -> None:
    """Regressao direta do comportamento antigo."""
    batch = _batch()
    with pytest.raises(InferenceContractError):
        align_feature_schema(batch, FEATURE_COLUMNS)


# --- encoder no caminho de inferencia ------------------------------------


def test_raw_categorical_batch_gets_the_trained_encoder(tmp_path: Path) -> None:
    train = _batch(60)
    train["Classe"] = "Operando"
    train["Frota"] = "F1"
    train["Tipo"] = "Caminhao"
    train["target_4h"] = [i % 2 for i in range(60)]

    encoder_path = save_encoder(
        CategoricalEncoder().fit(train), tmp_path / "categorical_encoder.joblib"
    )

    batch = _batch(10)
    batch["Classe"] = "Operando"
    batch["Frota"] = "F1"
    batch["Tipo"] = "Caminhao"

    encoded, applied = apply_categorical_encoder(batch, encoder_path)

    assert applied is True
    assert "Tag_freq" in encoded.columns
    assert "Frota_F1" in encoded.columns
    # colunas cruas consumidas pelo one-hot saem do frame
    assert "Frota" not in encoded.columns


def test_already_encoded_batch_skips_the_encoder(tmp_path: Path) -> None:
    batch = _batch()
    batch["Tag_freq"] = 0.2
    encoded, applied = apply_categorical_encoder(batch, tmp_path / "inexistente.joblib")
    assert applied is False
    assert encoded is batch


# --- ponta a ponta -------------------------------------------------------


def test_run_inference_rejects_a_batch_missing_features(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {"model": _FakeModel(), "feature_columns": FEATURE_COLUMNS, "threshold": 0.5},
        artifact_path,
    )
    input_path = tmp_path / "lote.parquet"
    _batch().to_parquet(input_path, index=False)  # sem Tag_freq

    with pytest.raises(InferenceContractError, match="features"):
        run_inference(
            input_path=input_path,
            model_path=artifact_path,
            output_path=tmp_path / "scores.parquet",
            priority_output_path=tmp_path / "ranking.csv",
            encoder_path=None,
        )


def test_run_inference_scores_a_complete_batch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(
        {"model": _FakeModel(), "feature_columns": FEATURE_COLUMNS, "threshold": 0.5},
        artifact_path,
    )
    batch = _batch()
    batch["Tag_freq"] = 0.2
    input_path = tmp_path / "lote.parquet"
    batch.to_parquet(input_path, index=False)

    result = run_inference(
        input_path=input_path,
        model_path=artifact_path,
        output_path=tmp_path / "scores.parquet",
        priority_output_path=tmp_path / "ranking.csv",
        encoder_path=None,
    )

    assert result["rows"] == len(batch)
    assert result["missing_feature_columns"] == []
    assert result["values_validated"] is True
    assert (tmp_path / "scores.parquet").exists()
