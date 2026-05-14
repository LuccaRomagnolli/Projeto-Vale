from pathlib import Path

import pandas as pd
from src.models.train_baseline import baseline_score, run_baseline_pipeline


def test_baseline_score_uses_alert_frequency() -> None:
    df = pd.DataFrame({"n_alertas_24h": [0, 12, 48]})
    scores = baseline_score(df)
    assert scores.tolist() == [0.0, 0.5, 1.0]


def test_run_baseline_pipeline_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "Id": range(40),
            "Tag": ["A"] * 20 + ["B"] * 20,
            "Fim": pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC"),
            "target_4h": [0, 1] * 20,
            "n_alertas_24h": [0, 4, 8, 12] * 10,
        }
    )
    features_path = tmp_path / "features_dataset.parquet"
    split_dir = tmp_path / "splits"
    df.to_parquet(features_path, index=False)

    monkeypatch.setattr("src.models.train_baseline.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.models.train_baseline.REPORTS_BASELINE_DIR", tmp_path / "reports")
    monkeypatch.setattr(
        "src.models.train_baseline.BASELINE_ARTIFACT_PATH",
        tmp_path / "models" / "baseline_heuristico.joblib",
    )
    monkeypatch.setattr(
        "src.models.train_baseline.BASELINE_REPORT_PATH",
        tmp_path / "reports" / "baseline_report.json",
    )
    monkeypatch.setattr(
        "src.models.train_baseline.BASELINE_SCORES_PATH",
        tmp_path / "reports" / "baseline_scores.parquet",
    )
    monkeypatch.setattr(
        "src.models.train_baseline.SPLIT_METADATA_PATH",
        split_dir / "split_metadata.json",
    )

    result = run_baseline_pipeline(features_path=features_path, split_dir=split_dir)

    assert Path(result["artifact_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["scores_path"]).exists()
    assert Path(result["split_metadata_path"]).exists()
    assert result["split_metadata"]["rows_total"] == 40
