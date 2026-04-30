from __future__ import annotations

import pandas as pd

from projeto_vale.split import temporal_split


def test_temporal_split_has_no_time_overlap():
    df = pd.DataFrame(
        {
            "Inicio": pd.date_range("2025-01-01", periods=20, freq="h"),
            "x": range(20),
        }
    )

    train, val, test = temporal_split(df, time_col="Inicio", train_size=0.7, val_size=0.15)

    assert train["Inicio"].max() <= val["Inicio"].min()
    assert val["Inicio"].max() <= test["Inicio"].min()
    assert len(train) + len(val) + len(test) == len(df)
