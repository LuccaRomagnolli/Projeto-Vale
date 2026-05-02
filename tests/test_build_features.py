import pandas as pd
from src.features.build_features import create_temporal_features


def test_create_temporal_features_expected_columns() -> None:
    df = pd.DataFrame({"Inicio": ["2026-01-10 08:30:00"]})
    out = create_temporal_features(df, "Inicio")

    assert "hora" in out.columns
    assert "dia_semana" in out.columns
    assert "mes" in out.columns
    assert int(out.loc[0, "hora"]) == 8
    assert int(out.loc[0, "mes"]) == 1
