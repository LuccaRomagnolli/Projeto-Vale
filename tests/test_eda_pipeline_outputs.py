import json
from pathlib import Path

import pandas as pd
import pytest
import src.eda.run_eda as run_eda_module
from src.eda.run_eda import (
    generate_eda_figures,
    generate_project_artifact_figures,
    load_labeled_dataset,
    run_eda_pipeline,
    write_eda_report,
)


def _small_eda_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Inicio": pd.to_datetime(
                [
                    "2026-01-01 08:00:00+00:00",
                    "2026-01-01 09:00:00+00:00",
                    "2026-01-01 10:00:00+00:00",
                ],
                utc=True,
            ),
            "Fim": pd.to_datetime(
                [
                    "2026-01-01 08:30:00+00:00",
                    "2026-01-01 09:40:00+00:00",
                    "2026-01-01 11:10:00+00:00",
                ],
                utc=True,
            ),
            "Tag": ["A", "B", "C"],
            "Frota": ["F1", "F1", "F2"],
            "Tipo": ["Caminhao", "Caminhao", "Carregadeira"],
            "Classe": ["Operando", "Parado", "Operando"],
            "target_4h": [0, 1, 0],
            "duracao_ciclo_min": [30.0, 40.0, 70.0],
            "hora_do_dia": [8, 9, 10],
            "dia_da_semana": [3, 3, 3],
            "is_fim_de_semana": [0, 0, 0],
        }
    )


def test_load_labeled_dataset_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_labeled_dataset(tmp_path / "missing.parquet")


def test_generate_eda_figures_creates_expected_files(tmp_path: Path) -> None:
    figures = generate_eda_figures(_small_eda_df(), figures_dir=tmp_path)
    assert len(figures) == 14
    for file_path in figures:
        assert file_path.exists()
        assert file_path.suffix == ".png"


def test_generate_project_artifact_figures_adds_threshold_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_selection_dir = tmp_path / "model_selection"
    operational_dir = tmp_path / "operational"
    figures_dir = tmp_path / "figures"
    model_selection_dir.mkdir()
    operational_dir.mkdir()

    pd.DataFrame(
        {
            "threshold": [0.10, 0.20, 0.30],
            "recall": [0.90, 0.75, 0.60],
            "precision": [0.20, 0.35, 0.50],
        }
    ).to_csv(model_selection_dir / "model_selected_threshold_curve.csv", index=False)
    (operational_dir / "operational_metrics_report.json").write_text(
        json.dumps(
            {
                "threshold_metrics": [
                    {
                        "split": "test",
                        "rows": 100,
                        "threshold": 0.20,
                        "true_positive": 30,
                        "false_positive": 20,
                        "false_negative": 10,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(run_eda_module, "REPORTS_MODEL_SELECTION_DIR", model_selection_dir)
    monkeypatch.setattr(run_eda_module, "REPORTS_OPERATIONAL_DIR", operational_dir)

    figures = generate_project_artifact_figures(figures_dir=figures_dir)

    expected_path = figures_dir / "threshold_diagnostic_confusion_matrix.png"
    assert expected_path in figures
    assert expected_path.exists()


def test_write_eda_report_creates_markdown(tmp_path: Path) -> None:
    report_path = write_eda_report(
        summary={
            "total_rows": 3,
            "total_tags": 3,
            "total_frotas": 2,
            "total_tipos": 2,
            "target_4h_positives": 1,
            "target_4h_positive_rate_pct": 33.3333,
            "inicio_min": "2026-01-01 08:00:00+00:00",
            "inicio_max": "2026-01-01 10:00:00+00:00",
            "duracao_ciclo_min_mean": 46.6667,
            "duracao_ciclo_min_median": 40.0,
            "duracao_ciclo_min_p95": 67.0,
            "missing_pct_top5": {"Classe": 0.0},
        },
        figure_paths=[tmp_path / "fig1.png"],
        report_path=tmp_path / "eda_report.md",
    )
    assert report_path.exists()
    content = report_path.read_text()
    assert "Relatório de EDA - Etapa 4" in content


def test_run_eda_pipeline_end_to_end_with_temp_paths(tmp_path: Path) -> None:
    dataset_path = tmp_path / "apontamentos_labeled.parquet"
    base = pd.DataFrame(
        {
            "Inicio": ["2026-01-01 08:00:00+00:00", "2026-01-01 09:00:00+00:00"],
            "Fim": ["2026-01-01 08:40:00+00:00", "2026-01-01 09:50:00+00:00"],
            "Tag": ["A", "B"],
            "Frota": ["F1", "F2"],
            "Tipo": ["Caminhao", "Caminhao"],
            "Classe": ["Operando", "Parado"],
            "target_4h": [0, 0],
        }
    )
    base.to_parquet(dataset_path, index=False)

    figures_dir = tmp_path / "figures"
    report_path = tmp_path / "eda_report.md"
    result = run_eda_pipeline(
        dataset_path=dataset_path,
        figures_dir=figures_dir,
        report_path=report_path,
        include_project_artifacts=False,
    )

    assert Path(result["report_path"]).exists()
    assert len(result["figures"]) == 14
    assert result["total_rows"] == 2
