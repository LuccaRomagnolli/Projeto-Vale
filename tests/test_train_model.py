from pathlib import Path

import pandas as pd
from src.models.train_model import (
    LEAKAGE_COLUMNS,
    prepare_model_matrix,
    run_model_pipeline,
    select_feature_columns,
)


def _model_df(n_rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": range(n_rows),
            "Inicio": pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC"),
            "Fim": pd.date_range("2026-01-01 00:30:00", periods=n_rows, freq="h", tz="UTC"),
            "Tag": ["A", "B"] * (n_rows // 2),
            "Classe": ["Operando", "Parado"] * (n_rows // 2),
            "next_critical_event_time": pd.NaT,
            "tte_horas": [1.0] * n_rows,
            "target_4h": [0, 1, 0, 1, 1, 0] * (n_rows // 6),
            "n_alertas_24h": [0, 1, 2, 3, 4, 5] * (n_rows // 6),
            "Tag_freq": [0.5] * n_rows,
            "Classe_target_enc": [0.2, 0.4] * (n_rows // 2),
            "Frota_X": [True, False] * (n_rows // 2),
        }
    )


def test_select_feature_columns_removes_leakage_columns() -> None:
    df = _model_df()
    features = select_feature_columns(df)
    assert not set(features) & LEAKAGE_COLUMNS
    assert "n_alertas_24h" in features
    assert "Frota_X" in features


def test_prepare_model_matrix_converts_bool_to_numeric() -> None:
    df = _model_df()
    features = select_feature_columns(df)
    matrix = prepare_model_matrix(df, features)
    assert str(matrix["Frota_X"].dtype) == "float32"


def test_run_model_pipeline_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    df = _model_df()
    df.iloc[:40].to_parquet(split_dir / "features_train.parquet", index=False)
    df.iloc[40:50].to_parquet(split_dir / "features_val.parquet", index=False)
    df.iloc[50:].to_parquet(split_dir / "features_test.parquet", index=False)

    monkeypatch.setattr("src.models.train_model.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.models.train_model.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        "src.models.train_model.MODEL_ARTIFACT_PATH",
        tmp_path / "models" / "model_principal.joblib",
    )
    monkeypatch.setattr(
        "src.models.train_model.MODEL_REPORT_PATH",
        tmp_path / "reports" / "model_principal_report.json",
    )
    monkeypatch.setattr(
        "src.models.train_model.MODEL_SCORES_PATH",
        tmp_path / "reports" / "model_principal_scores.parquet",
    )
    monkeypatch.setattr(
        "src.models.train_model.FEATURE_IMPORTANCE_PATH",
        tmp_path / "reports" / "model_feature_importance.csv",
    )

    result = run_model_pipeline(split_dir)

    assert Path(result["artifact_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["scores_path"]).exists()
    assert result["feature_count"] > 0
