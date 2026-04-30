from __future__ import annotations

import numpy as np
import pandas as pd

TURNOS = {
    "madrugada": range(0, 6),
    "manha": range(6, 12),
    "tarde": range(12, 18),
    "noite": range(18, 24),
}


def _infer_turno(hour: int) -> str:
    for name, hours in TURNOS.items():
        if hour in hours:
            return name
    return "desconhecido"


def build_apontamentos_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features temporais e rolling por Tag para os apontamentos."""
    out = df.copy()
    out["Inicio"] = pd.to_datetime(out["Inicio"], errors="coerce")
    out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce")

    out = out.dropna(subset=["Inicio", "Fim", "Tag"]).copy()
    out = out.sort_values(["Tag", "Inicio"]).reset_index(drop=True)

    out["ciclo_duracao_min"] = (out["Fim"] - out["Inicio"]).dt.total_seconds() / 60.0
    out["ciclo_duracao_min"] = out["ciclo_duracao_min"].clip(lower=0)

    out["hora_dia"] = out["Inicio"].dt.hour
    out["dia_semana"] = out["Inicio"].dt.dayofweek
    out["mes"] = out["Inicio"].dt.month
    out["turno"] = out["hora_dia"].map(_infer_turno)

    out["ciclos_previos_tag"] = out.groupby("Tag").cumcount()

    rolling_mean = (
        out.groupby("Tag")["ciclo_duracao_min"]
        .rolling(window=6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std = (
        out.groupby("Tag")["ciclo_duracao_min"]
        .rolling(window=6, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    out["rolling_duracao_media_6"] = rolling_mean
    out["rolling_duracao_std_6"] = rolling_std

    if "Operador" not in out.columns:
        out["Operador"] = "desconhecido"

    out["is_weekend"] = out["dia_semana"].isin([5, 6]).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out
