from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def build_next_alert_target(
    apontamentos_df: pd.DataFrame,
    alert_events_df: pd.DataFrame,
    horizon_hours: int = 4,
    tag_col: str = "Tag",
    event_time_col: str = "Inicio",
) -> pd.Series:
    """Target binario: 1 se houver alerta critico nas proximas N horas para a mesma Tag."""
    if tag_col not in apontamentos_df.columns:
        raise ValueError(f"Coluna de tag ausente em apontamentos: {tag_col}")
    if event_time_col not in apontamentos_df.columns:
        raise ValueError(f"Coluna de tempo ausente em apontamentos: {event_time_col}")

    df = apontamentos_df[[tag_col, event_time_col]].copy()
    df[event_time_col] = pd.to_datetime(df[event_time_col], errors="coerce")

    alerts = alert_events_df.copy()
    if "tag" not in alerts.columns or "alert_time" not in alerts.columns:
        raise ValueError("alert_events_df deve conter colunas ['tag', 'alert_time']")
    alerts["alert_time"] = pd.to_datetime(alerts["alert_time"], errors="coerce")

    horizon = timedelta(hours=horizon_hours)
    y = np.zeros(len(df), dtype=int)

    grouped_alerts = {
        tag: g["alert_time"].dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        for tag, g in alerts.groupby("tag", sort=False)
    }

    for idx, row in df.iterrows():
        tag = row[tag_col]
        event_time = row[event_time_col]
        if pd.isna(event_time) or tag not in grouped_alerts:
            continue

        arr = grouped_alerts[tag]
        if arr.size == 0:
            continue

        current = np.datetime64(event_time.to_datetime64())
        pos = np.searchsorted(arr, current, side="right")
        if pos >= arr.size:
            continue

        next_alert = pd.Timestamp(arr[pos])
        if next_alert <= event_time + horizon:
            y[idx] = 1

    return pd.Series(y, index=apontamentos_df.index, name="target_critical_4h")
