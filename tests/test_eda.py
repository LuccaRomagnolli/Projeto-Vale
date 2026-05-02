import pandas as pd
from src.eda.run_eda import build_eda_summary, enrich_for_eda


def test_enrich_for_eda_creates_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "Inicio": ["2026-01-01 08:00:00+00:00"],
            "Fim": ["2026-01-01 09:30:00+00:00"],
            "Tag": ["TAG-1"],
            "Frota": ["FROTA-1"],
            "Tipo": ["Caminhao"],
            "Classe": ["Operando"],
            "target_4h": [1],
        }
    )
    out = enrich_for_eda(df)
    assert "duracao_ciclo_min" in out.columns
    assert "hora_do_dia" in out.columns
    assert "dia_da_semana" in out.columns
    assert "is_fim_de_semana" in out.columns
    assert round(float(out.loc[0, "duracao_ciclo_min"]), 2) == 90.0


def test_build_eda_summary_returns_core_metrics() -> None:
    df = pd.DataFrame(
        {
            "Inicio": pd.to_datetime(
                ["2026-01-01 08:00:00+00:00", "2026-01-01 10:00:00+00:00"], utc=True
            ),
            "Fim": pd.to_datetime(
                ["2026-01-01 09:00:00+00:00", "2026-01-01 12:00:00+00:00"], utc=True
            ),
            "Tag": ["A", "B"],
            "Frota": ["F1", "F2"],
            "Tipo": ["Caminhao", "Caminhao"],
            "Classe": ["Operando", None],
            "duracao_ciclo_min": [60.0, 120.0],
            "target_4h": [1, 0],
        }
    )
    summary = build_eda_summary(df)
    assert summary["total_rows"] == 2
    assert summary["target_4h_positives"] == 1
    assert summary["total_tags"] == 2
    assert "Classe" in summary["missing_pct_top5"]
