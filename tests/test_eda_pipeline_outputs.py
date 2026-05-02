from pathlib import Path

import pandas as pd
import pytest
from src.eda.run_eda import (
    generate_eda_figures,
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
    assert len(figures) == 5
    for file_path in figures:
        assert file_path.exists()
        assert file_path.suffix == ".png"


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
    )

    assert Path(result["report_path"]).exists()
    assert len(result["figures"]) == 5
    assert result["total_rows"] == 2
