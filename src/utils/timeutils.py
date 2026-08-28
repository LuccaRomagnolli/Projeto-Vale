"""Conversoes temporais com unidade explicita.

O pandas expoe a resolucao interna dos datetimes em `astype("int64")` e em
`Timedelta.value`. Essa resolucao era sempre nanossegundos ate o pandas 2.x e
passou a ser microssegundos por padrao no pandas 3.0. Qualquer aritmetica que
assuma nanossegundos de forma implicita fica 1000x errada apos a atualizacao,
sem erro nem aviso -- foi o que aconteceu com as janelas moveis de alertas e
com `dias_desde_ultimo_alerta`.

Este modulo centraliza a conversao para que a unidade seja sempre explicita e
independente da versao do pandas instalada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NS_PER_SECOND = 1_000_000_000
NS_PER_HOUR = 60 * 60 * NS_PER_SECOND
NS_PER_DAY = 24 * NS_PER_HOUR


def to_epoch_ns(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Converte datetimes (com ou sem timezone) para inteiros em nanossegundos."""
    if isinstance(values, pd.Series):
        naive = values.dt.tz_convert(None) if values.dt.tz is not None else values
    else:
        naive = values.tz_convert(None) if values.tz is not None else values
    return naive.to_numpy(dtype="datetime64[ns]").astype("int64")


def hours_to_ns(hours: float) -> int:
    """Converte horas para nanossegundos."""
    return int(hours * NS_PER_HOUR)
