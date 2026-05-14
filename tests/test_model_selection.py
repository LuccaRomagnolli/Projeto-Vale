from pathlib import Path

import joblib
import pandas as pd
import pytest
from src.models.model_selection import (
    BASELINE_MODEL_NAME,
    OFFICIAL_CANDIDATE_NAMES,
    run_model_selection_pipeline,
    select_model,
)


def _selection_df(n_rows: int = 96) -> pd.DataFrame:
    target = ([0, 1, 0, 1, 1, 0] * (n_rows // 6))[:n_rows]
    return pd.DataFrame(
        {
            "Id": range(n_rows),
            "Inicio": pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC"),
            "Fim": pd.date_range("2026-01-01 00:30:00", periods=n_rows, freq="h", tz="UTC"),
            "Tag": ["A", "B", "C", "D"] * (n_rows // 4),
            "Classe": ["Operando", "Parado", "Carregando", "Fila"] * (n_rows // 4),
            "turno": ["manha", "tarde"] * (n_rows // 2),
            "next_critical_event_time": pd.NaT,
            "tte_horas": [1.0] * n_rows,
            "target_4h": target,
            "n_alertas_24h": [float(value) for value in target],
            "duracao_ciclo_min": [10.0, 15.0, 20.0, 18.0] * (n_rows // 4),
            "Tag_freq": [0.3, 0.4, 0.2, 0.1] * (n_rows // 4),
            "Classe_target_enc": [0.2, 0.5, 0.7, 0.4] * (n_rows // 4),
            "Frota_X": [True, False, True, False] * (n_rows // 4),
            "Tipo_Caminhao": [True, True, False, False] * (n_rows // 4),
        }
    )


def test_official_candidates_exclude_diagnostic_baseline() -> None:
    assert OFFICIAL_CANDIDATE_NAMES == (
        "lightgbm_optuna",
        "xgboost_optuna",
        "hist_gbdt_optuna",
        "extra_trees_optuna",
    )
    assert BASELINE_MODEL_NAME not in OFFICIAL_CANDIDATE_NAMES


def test_select_model_uses_only_eligible_operational_scorecard() -> None:
    summary = pd.DataFrame(
        {
            "model_name": ["baseline", "candidate_a", "candidate_b"],
            "eligible_for_selection": [False, True, True],
            "val_top15_recall_at_k": [0.99, 0.80, 0.75],
            "val_top15_precision_at_k": [0.99, 0.60, 0.90],
            "val_top15_lift_vs_random": [3.0, 2.0, 2.5],
            "val_auc_pr": [0.9, 0.3, 0.8],
        }
    )

    assert select_model(summary) == "candidate_a"


def test_run_model_selection_pipeline_writes_neutral_outputs(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("optuna")

    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True)
    df = _selection_df()
    df.iloc[:64].to_parquet(split_dir / "features_train.parquet", index=False)
    df.iloc[64:80].to_parquet(split_dir / "features_val.parquet", index=False)
    df.iloc[80:].to_parquet(split_dir / "features_test.parquet", index=False)

    monkeypatch.setattr("src.models.model_selection.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(
        "src.models.model_selection.REPORTS_MODEL_SELECTION_DIR",
        tmp_path / "reports",
    )
    monkeypatch.setattr(
        "src.models.model_selection.OFFICIAL_CANDIDATE_NAMES",
        ("hist_gbdt_optuna",),
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTED_MODEL_PATH",
        tmp_path / "models" / "model_selected.joblib",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTION_REPORT_JSON",
        tmp_path / "reports" / "model_selection_report.json",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTION_REPORT_CSV",
        tmp_path / "reports" / "model_selection_report.csv",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTION_TRIALS_CSV",
        tmp_path / "reports" / "model_selection_trials.csv",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTION_SCORES_PATH",
        tmp_path / "reports" / "model_selection_scores.parquet",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTION_BACKTEST_CSV",
        tmp_path / "reports" / "model_selection_backtest_report.csv",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTED_THRESHOLD_CURVE_CSV",
        tmp_path / "reports" / "model_selected_threshold_curve.csv",
    )
    monkeypatch.setattr(
        "src.models.model_selection.SELECTED_FEATURE_IMPORTANCE_CSV",
        tmp_path / "reports" / "model_selected_feature_importance.csv",
    )

    result = run_model_selection_pipeline(
        split_dir=split_dir,
        features_path=tmp_path / "missing_features.parquet",
        n_trials=1,
        n_backtest_folds=2,
    )

    assert result["selected_model_name"] == "hist_gbdt_optuna"
    assert Path(result["artifact_path"]).exists()
    assert Path(result["json_path"]).exists()
    assert Path(result["trials_path"]).exists()
    assert Path(result["scores_path"]).exists()
    artifact = joblib.load(result["artifact_path"])
    assert {"model", "model_name", "feature_columns", "threshold"} <= set(artifact)
