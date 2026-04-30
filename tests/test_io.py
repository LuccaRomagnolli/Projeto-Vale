from __future__ import annotations

import pandas as pd

from projeto_vale.io import read_apontamentos, read_telemetria


def test_read_apontamentos_parquet(tmp_path):
    path = tmp_path / "apont.parquet"
    df = pd.DataFrame(
        {
            "Id": [1],
            "Inicio": ["2025-01-01 00:00:00"],
            "Fim": ["2025-01-01 00:10:00"],
            "Tag": ["CA1"],
            "Frota": ["793"],
            "Tipo": ["Caminhao"],
            "Classe": ["Operando"],
        }
    )
    df.to_parquet(path, index=False)

    out = read_apontamentos(path)
    assert set(["Id", "Inicio", "Fim", "Tag", "Frota", "Tipo", "Classe"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["Inicio"])


def test_read_telemetria_xlsx(tmp_path):
    path = tmp_path / "tel.xlsx"
    df = pd.DataFrame(
        {
            "Data_Evento": ["2025-01-01 00:00:01"],
            "TAG": ["CA1"],
            "Alarme": ["Low Oil"],
            "Criticidade": ["Muito Alto"],
        }
    )
    df.to_excel(path, index=False)

    out = read_telemetria(path)
    assert set(["Data_Evento", "TAG"]).issubset(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["Data_Evento"])
