from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "Inicio",
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal ordenado sem vazamento entre treino/validacao/teste."""
    if time_col not in df.columns:
        raise ValueError(f"Coluna temporal nao encontrada: {time_col}")
    if not 0 < train_size < 1:
        raise ValueError("train_size deve estar em (0,1)")
    if not 0 < val_size < 1:
        raise ValueError("val_size deve estar em (0,1)")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size deve ser menor que 1")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    return train, val, test
