"""Engenharia de features para modelagem preditiva."""

import pandas as pd


def create_temporal_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """Cria features temporais básicas."""
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out["hora"] = out[dt_col].dt.hour
    out["dia_semana"] = out[dt_col].dt.dayofweek
    out["mes"] = out[dt_col].dt.month
    return out


def main() -> None:
    print("[TODO] Implementar rolling windows por Tag e features derivadas de regra")


if __name__ == "__main__":
    main()
