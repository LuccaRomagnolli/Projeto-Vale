"""Job de lote diario: isolamento de falhas e idempotencia."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.batch_inference import batch_output_paths, discover_batches, run_daily_batch

FEATURE_COLUMNS = ["n_alertas_4h", "duracao_ciclo_min", "Tag_freq"]


class _FakeModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.clip(x.iloc[:, 0].to_numpy(dtype=float) / 10.0, 0, 1)
        return np.column_stack([1 - positive, positive])


def _valid_batch(n_rows: int = 12, start: str = "2026-03-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tag": [f"T{i % 3}" for i in range(n_rows)],
            "Fim": pd.date_range(start, periods=n_rows, freq="h", tz="UTC"),
            "n_alertas_4h": [float(i % 4) for i in range(n_rows)],
            "duracao_ciclo_min": [12.0] * n_rows,
            "dias_desde_ultimo_alerta": [1.5] * n_rows,
            "Tag_freq": [0.2] * n_rows,
        }
    )


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    joblib.dump({"model": _FakeModel(), "feature_columns": FEATURE_COLUMNS, "threshold": 0.5}, path)
    return path


def _dirs(tmp_path: Path) -> dict[str, Path]:
    layout = {
        "input_dir": tmp_path / "entrada",
        "processed_dir": tmp_path / "processados",
        "rejected_dir": tmp_path / "rejeitados",
        "output_dir": tmp_path / "saida",
        "log_path": tmp_path / "saida" / "batch_log.json",
    }
    layout["input_dir"].mkdir(parents=True)
    return layout


def test_discover_batches_ignores_unsupported_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "entrada"
    input_dir.mkdir()
    (input_dir / "lote.parquet").touch()
    (input_dir / "notas.txt").touch()
    (input_dir / "planilha.csv").touch()

    found = [p.name for p in discover_batches(input_dir)]
    assert found == ["lote.parquet", "planilha.csv"]


def test_discover_batches_tolerates_missing_directory(tmp_path: Path) -> None:
    assert discover_batches(tmp_path / "inexistente") == []


def test_bad_batch_is_isolated_and_good_ones_still_run(tmp_path: Path) -> None:
    """Um lote reprovado nao pode impedir os demais nem virar ranking."""
    layout = _dirs(tmp_path)
    _valid_batch().to_parquet(layout["input_dir"] / "2026-03-01.parquet", index=False)
    # lote ruim: sem a feature Tag_freq exigida pelo artefato
    _valid_batch(start="2026-03-02").drop(columns=["Tag_freq"]).to_parquet(
        layout["input_dir"] / "2026-03-02.parquet", index=False
    )

    summary = run_daily_batch(**layout, model_path=_artifact(tmp_path), encoder_path=None)

    assert summary["batches_found"] == 2
    assert summary["batches_processed"] == 1
    assert summary["batches_rejected"] == 1

    rejected = next(r for r in summary["results"] if r["status"] == "rejeitado")
    assert "2026-03-02" in rejected["batch"]
    assert "Tag_freq" in rejected["motivo"]

    # arquivos vao para diretorios distintos, entrada fica limpa
    assert (layout["processed_dir"] / "2026-03-01.parquet").exists()
    assert (layout["rejected_dir"] / "2026-03-02.parquet").exists()
    assert discover_batches(layout["input_dir"]) == []

    # o lote rejeitado nao gerou ranking
    assert not (layout["output_dir"] / "2026-03-02_priority.csv").exists()
    assert (layout["output_dir"] / "2026-03-01_priority.csv").exists()


def test_reprocessing_the_same_date_is_idempotent(tmp_path: Path) -> None:
    layout = _dirs(tmp_path)
    model_path = _artifact(tmp_path)
    batch = _valid_batch()

    batch.to_parquet(layout["input_dir"] / "2026-03-01.parquet", index=False)
    first = run_daily_batch(**layout, model_path=model_path, encoder_path=None)
    priority_path = batch_output_paths("2026-03-01.parquet", layout["output_dir"])["priority"]
    first_content = priority_path.read_text(encoding="utf-8")

    # mesmo arquivo depositado de novo
    batch.to_parquet(layout["input_dir"] / "2026-03-01.parquet", index=False)
    second = run_daily_batch(**layout, model_path=model_path, encoder_path=None)

    assert first["batches_processed"] == second["batches_processed"] == 1
    assert priority_path.read_text(encoding="utf-8") == first_content
    # o arquivo arquivado foi sobrescrito, nao duplicado
    assert len(list(layout["processed_dir"].iterdir())) == 1


def test_log_records_every_batch(tmp_path: Path) -> None:
    layout = _dirs(tmp_path)
    _valid_batch().to_parquet(layout["input_dir"] / "2026-03-01.parquet", index=False)

    run_daily_batch(
        **layout,
        model_path=_artifact(tmp_path),
        encoder_path=None,
        executed_at="2026-03-01T06:00:00+00:00",
    )

    log = json.loads(layout["log_path"].read_text(encoding="utf-8"))
    assert log["executed_at_utc"] == "2026-03-01T06:00:00+00:00"
    assert log["results"][0]["status"] == "processado"


def test_empty_input_directory_is_not_an_error(tmp_path: Path) -> None:
    layout = _dirs(tmp_path)
    summary = run_daily_batch(**layout, model_path=_artifact(tmp_path), encoder_path=None)
    assert summary["batches_found"] == 0
    assert summary["batches_rejected"] == 0
