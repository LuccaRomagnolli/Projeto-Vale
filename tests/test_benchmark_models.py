from pathlib import Path

import pandas as pd
import pytest
from src.models.benchmark_models import (
    build_candidate_models,
    run_benchmark_pipeline,
    select_winner,
)


def _benchmark_df(n_rows: int = 90) -> pd.DataFrame:
    target = ([0, 1, 0, 1, 1, 0] * (n_rows // 6))[:n_rows]
    return pd.DataFrame(
        {
            "Id": range(n_rows),
            "Inicio": pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC"),
            "Fim": pd.date_range("2026-01-01 00:30:00", periods=n_rows, freq="h", tz="UTC"),
            "Tag": ["A", "B", "C"] * (n_rows // 3),
            "Classe": ["Operando", "Parado", "Carregando"] * (n_rows // 3),
            "next_critical_event_time": pd.NaT,
            "tte_horas": [1.0] * n_rows,
            "target_4h": target,
            "n_alertas_24h": [float(value) for value in target],
            "duracao_ciclo_min": [10.0, 15.0, 20.0] * (n_rows // 3),
            "Tag_freq": [0.3, 0.4, 0.3] * (n_rows // 3),
            "Classe_target_enc": [0.2, 0.5, 0.7] * (n_rows // 3),
            "Frota_X": [True, False, True] * (n_rows // 3),
        }
    )


def test_select_winner_uses_operational_validation_scorecard() -> None:
    summary = pd.DataFrame(
        {
            "model_name": ["a", "b"],
            "val_top15_recall_at_k": [0.8, 0.7],
            "val_top15_precision_at_k": [0.6, 0.9],
            "val_top15_lift_vs_random": [1.8, 2.5],
            "val_auc_pr": [0.2, 0.4],
            "val_precision": [0.1, 0.9],
            "val_f1": [0.8, 0.2],
        }
    )
    assert select_winner(summary) == "a"


def test_build_candidate_models_includes_lightgbm_when_installed() -> None:
    pytest.importorskip("lightgbm")

    candidates = build_candidate_models()

    assert "lightgbm_balanced" in candidates


def test_run_benchmark_pipeline_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    df = _benchmark_df()
    df.iloc[:60].to_parquet(split_dir / "features_train.parquet", index=False)
    df.iloc[60:75].to_parquet(split_dir / "features_val.parquet", index=False)
    df.iloc[75:].to_parquet(split_dir / "features_test.parquet", index=False)

    monkeypatch.setattr("src.models.benchmark_models.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.models.benchmark_models.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        "src.models.benchmark_models.BENCHMARK_REPORT_JSON",
        tmp_path / "reports" / "model_benchmark_report.json",
    )
    monkeypatch.setattr(
        "src.models.benchmark_models.BENCHMARK_REPORT_CSV",
        tmp_path / "reports" / "model_benchmark_report.csv",
    )
    monkeypatch.setattr(
        "src.models.benchmark_models.BENCHMARK_SCORES_PATH",
        tmp_path / "reports" / "model_benchmark_scores.parquet",
    )
    monkeypatch.setattr(
        "src.models.benchmark_models.BENCHMARK_WINNER_PATH",
        tmp_path / "models" / "model_benchmark_winner.joblib",
    )

    result = run_benchmark_pipeline(split_dir)

    assert result["models_trained"] >= 3
    assert result["winner_name"]
    assert Path(result["json_path"]).exists()
    assert Path(result["csv_path"]).exists()
    assert Path(result["scores_path"]).exists()
    assert Path(result["winner_artifact_path"]).exists()
