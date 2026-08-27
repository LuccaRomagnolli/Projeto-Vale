"""Camada de dados da interface operacional de engenharia de minas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.utils.config import (
    BASE_DIR,
    CRITICAL_EVENTS_PATH,
    INTERIM_DIR,
    LABELED_DATASET_PATH,
    REPORTS_DIR,
    REPORTS_MODEL_SELECTION_DIR,
    REPORTS_OPERATIONAL_DIR,
    REPORTS_SEGMENT_DIR,
)

DEFAULT_PRIORITY_PATH = REPORTS_DIR / "daily_priority_top15.csv"
DEFAULT_METRICS_PATH = REPORTS_OPERATIONAL_DIR / "operational_metrics_report.json"
DEFAULT_HOTSPOTS_PATH = REPORTS_SEGMENT_DIR / "segment_tag_hotspots.csv"
DEFAULT_SELECTION_PATH = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.json"
DEFAULT_WORKLOG_PATH = INTERIM_DIR / "ui_worklog.json"
LOGO_PATH = BASE_DIR / "pictures" / "vale-logo-removebg-preview.png"

LABELED_COLUMNS = [
    "Id",
    "Inicio",
    "Fim",
    "Tag",
    "Frota",
    "Tipo",
    "Classe",
    "tte_horas",
    "target_4h",
]
EVENT_COLUMNS = ["TAG", "EVENT_TIME"]
VALID_STATUSES = {
    "pendente",
    "em_inspecao",
    "em_andamento",
    "resolvido",
    "descartado",
}
RISK_LABELS = {
    "alto_acima_threshold": "Alto · acima do limiar",
    "alto_por_ranking": "Alto · ranking",
    "medio_por_ranking": "Medio · ranking",
    "monitorar_por_ranking": "Monitorar",
}


def json_safe(value: Any) -> Any:
    """Converte valores pandas/numpy/datetime para JSON nativo."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serializa um DataFrame em lista de dicionarios JSON-safe."""
    if frame.empty:
        return []
    payload: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        payload.append({str(key): json_safe(value) for key, value in row.items()})
    return payload


def paginate(frame: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int]:
    """Aplica paginacao 1-based e devolve o recorte com o total."""
    safe_page = max(int(page), 1)
    safe_size = max(min(int(page_size), 200), 1)
    total = int(len(frame))
    start = (safe_page - 1) * safe_size
    return frame.iloc[start : start + safe_size].copy(), total


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    available = set(pq.read_schema(path).names)
    selected = [column for column in columns if column in available]
    if not selected:
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=selected)


def _to_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


@dataclass(frozen=True)
class StorePaths:
    priority: Path = DEFAULT_PRIORITY_PATH
    labeled: Path = LABELED_DATASET_PATH
    events: Path = CRITICAL_EVENTS_PATH
    metrics: Path = DEFAULT_METRICS_PATH
    hotspots: Path = DEFAULT_HOTSPOTS_PATH
    selection: Path = DEFAULT_SELECTION_PATH
    worklog: Path = DEFAULT_WORKLOG_PATH
    logo: Path = LOGO_PATH


class OperationalStore:
    """Carrega artefatos do pipeline e expoe consultas da interface operacional."""

    def __init__(self, paths: StorePaths | None = None) -> None:
        self.paths = paths or StorePaths()
        self._priority = pd.DataFrame()
        self._cycles = pd.DataFrame()
        self._events = pd.DataFrame()
        self._hotspots = pd.DataFrame()
        self._metrics: dict[str, Any] = {}
        self._selection: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        self._priority = self._load_priority()
        self._cycles = self._load_cycles()
        self._events = self._load_events()
        self._hotspots = _read_csv(self.paths.hotspots)
        self._metrics = _read_json(self.paths.metrics)
        self._selection = _read_json(self.paths.selection)

    def _load_priority(self) -> pd.DataFrame:
        frame = _read_csv(self.paths.priority)
        if frame.empty:
            return frame
        frame["data"] = pd.to_datetime(frame["data"], errors="coerce").dt.date
        frame["score"] = pd.to_numeric(frame.get("score"), errors="coerce")
        frame["rank"] = pd.to_numeric(frame.get("rank"), errors="coerce")
        if "risco_segmento" in frame.columns:
            frame["risco_rotulo"] = frame["risco_segmento"].map(RISK_LABELS).fillna(
                frame["risco_segmento"]
            )
        return frame.sort_values(["data", "rank"], ascending=[False, True])

    def _load_cycles(self) -> pd.DataFrame:
        frame = _read_parquet_columns(self.paths.labeled, LABELED_COLUMNS)
        if frame.empty:
            return pd.DataFrame(columns=[*LABELED_COLUMNS, "data", "duracao_ciclo_min"])
        frame["Inicio"] = _to_utc(frame["Inicio"]) if "Inicio" in frame.columns else pd.NaT
        frame["Fim"] = _to_utc(frame["Fim"]) if "Fim" in frame.columns else pd.NaT
        frame["data"] = frame["Fim"].dt.date
        if "Inicio" in frame.columns and "Fim" in frame.columns:
            frame["duracao_ciclo_min"] = (frame["Fim"] - frame["Inicio"]).dt.total_seconds() / 60.0
        else:
            frame["duracao_ciclo_min"] = pd.NA
        return frame

    def _load_events(self) -> pd.DataFrame:
        frame = _read_parquet_columns(self.paths.events, EVENT_COLUMNS)
        if frame.empty:
            extra = pd.read_parquet(self.paths.events) if self.paths.events.exists() else pd.DataFrame()
            if extra.empty:
                return pd.DataFrame(columns=["Tag", "EVENT_TIME", "data"])
            frame = extra
        rename = {"TAG": "Tag"} if "TAG" in frame.columns and "Tag" not in frame.columns else {}
        frame = frame.rename(columns=rename)
        if "EVENT_TIME" in frame.columns:
            frame["EVENT_TIME"] = _to_utc(frame["EVENT_TIME"])
            frame["data"] = frame["EVENT_TIME"].dt.date
        return frame

    def _worklog(self) -> dict[str, Any]:
        payload = _read_json(self.paths.worklog)
        items = payload.get("items", payload if isinstance(payload, dict) else {})
        if not isinstance(items, dict):
            return {}
        return items

    def save_worklog(self, items: dict[str, Any]) -> None:
        self.paths.worklog.parent.mkdir(parents=True, exist_ok=True)
        self.paths.worklog.write_text(
            json.dumps({"items": items}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def available_dates(self) -> list[str]:
        dates = []
        if not self._priority.empty:
            dates.extend(self._priority["data"].dropna().tolist())
        if not self._cycles.empty:
            dates.extend(self._cycles["data"].dropna().tolist())
        unique = sorted({str(item) for item in dates if item is not None}, reverse=True)
        return unique

    def latest_date(self) -> str | None:
        dates = self.available_dates()
        return dates[0] if dates else None

    def filters(self) -> dict[str, Any]:
        cycles = self._cycles
        priority = self._priority
        tags = sorted(
            set(cycles.get("Tag", pd.Series(dtype=str)).dropna().astype(str))
            | set(priority.get("Tag", pd.Series(dtype=str)).dropna().astype(str))
        )
        frotas = sorted(set(cycles.get("Frota", pd.Series(dtype=str)).dropna().astype(str)))
        tipos = sorted(set(cycles.get("Tipo", pd.Series(dtype=str)).dropna().astype(str)))
        classes = sorted(set(cycles.get("Classe", pd.Series(dtype=str)).dropna().astype(str)))
        riscos = sorted(
            set(priority.get("risco_segmento", pd.Series(dtype=str)).dropna().astype(str))
        )
        return {
            "dates": self.available_dates(),
            "latest_date": self.latest_date(),
            "tags": tags,
            "frotas": frotas,
            "tipos": tipos,
            "classes": classes,
            "riscos": riscos,
            "statuses": sorted(VALID_STATUSES),
        }

    def _priority_for_date(self, selected_date: str | None) -> pd.DataFrame:
        if self._priority.empty:
            return self._priority
        target = selected_date or self.latest_date()
        if not target:
            return self._priority.iloc[0:0]
        mask = self._priority["data"].astype(str) == str(target)
        return self._priority.loc[mask].copy()

    def overview(self, selected_date: str | None = None) -> dict[str, Any]:
        day = selected_date or self.latest_date()
        priority = self._priority_for_date(day)
        worklog = self._worklog()
        high_risk = 0
        if not priority.empty and "risco_segmento" in priority.columns:
            high_risk = int(
                priority["risco_segmento"].isin(["alto_acima_threshold", "alto_por_ranking"]).sum()
            )

        cycles_day = self._cycles
        if day and not cycles_day.empty:
            cycles_day = cycles_day.loc[cycles_day["data"].astype(str) == str(day)]
        events_day = self._events
        if day and not events_day.empty and "data" in events_day.columns:
            events_day = events_day.loc[events_day["data"].astype(str) == str(day)]

        positives = int(cycles_day["target_4h"].sum()) if "target_4h" in cycles_day.columns else 0
        cycle_count = int(len(cycles_day))
        selected = self._selection.get("selected_model", {})
        test_topk = None
        for row in self._metrics.get("test_daily_topk_metrics", []):
            if int(row.get("top_k_tags_per_day", 0)) == 15:
                test_topk = row
                break

        pending = sum(
            1
            for item in worklog.values()
            if isinstance(item, dict) and item.get("status") in {"pendente", "em_inspecao", "em_andamento"}
        )
        return {
            "date": day,
            "kpis": {
                "priority_count": int(len(priority)),
                "high_risk_count": high_risk,
                "cycle_count": cycle_count,
                "positive_count": positives,
                "positive_rate": (positives / cycle_count) if cycle_count else 0.0,
                "critical_events": int(len(events_day)),
                "open_actions": pending,
                "unique_tags": int(self._cycles["Tag"].nunique()) if not self._cycles.empty else 0,
                "unique_fleets": int(self._cycles["Frota"].nunique()) if not self._cycles.empty else 0,
            },
            "model": {
                "name": selected.get("model_name") or self._metrics.get("model_name"),
                "threshold": selected.get("threshold") or self._metrics.get("threshold"),
                "horizon": "4h",
                "top_k": 15,
            },
            "topk": test_topk,
            "sources": {
                "priority": self.paths.priority.exists(),
                "labeled": self.paths.labeled.exists(),
                "events": self.paths.events.exists(),
                "metrics": self.paths.metrics.exists(),
            },
        }

    def priority_board(
        self,
        selected_date: str | None = None,
        frota: str | None = None,
        risco: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        frame = self._priority_for_date(selected_date)
        if frota and "Frota" in frame.columns:
            frame = frame.loc[frame["Frota"].astype(str) == frota]
        if risco and "risco_segmento" in frame.columns:
            frame = frame.loc[frame["risco_segmento"].astype(str) == risco]
        if query:
            needle = query.strip().casefold()
            frame = frame.loc[frame["Tag"].astype(str).str.casefold().str.contains(needle, na=False)]

        worklog = self._worklog()
        rows = []
        for row in records(frame):
            key = f"priority:{row.get('data')}:{row.get('Tag')}"
            action = worklog.get(key, {})
            row["action_key"] = key
            row["status"] = action.get("status", "pendente")
            row["note"] = action.get("note", "")
            row["operator"] = action.get("operator", "")
            row["updated_at"] = action.get("updated_at")
            rows.append(row)

        risk_counts: dict[str, int] = {}
        for row in rows:
            label = str(row.get("risco_rotulo") or row.get("risco_segmento") or "indefinido")
            risk_counts[label] = risk_counts.get(label, 0) + 1
        return {
            "date": selected_date or self.latest_date(),
            "count": len(rows),
            "risk_counts": risk_counts,
            "items": rows,
        }

    def alerts(
        self,
        selected_date: str | None = None,
        tag: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        frame = self._events.copy()
        if selected_date and "data" in frame.columns:
            frame = frame.loc[frame["data"].astype(str) == str(selected_date)]
        if tag and "Tag" in frame.columns:
            frame = frame.loc[frame["Tag"].astype(str) == tag]
        if not frame.empty and "EVENT_TIME" in frame.columns:
            frame = frame.sort_values("EVENT_TIME", ascending=False)
        page_frame, total = paginate(frame, page, page_size)
        worklog = self._worklog()
        items = []
        for row in records(page_frame):
            key = f"event:{row.get('Tag')}:{row.get('EVENT_TIME')}"
            action = worklog.get(key, {})
            row["action_key"] = key
            row["status"] = action.get("status", "pendente")
            row["note"] = action.get("note", "")
            items.append(row)
        return {"total": total, "page": max(page, 1), "page_size": page_size, "items": items}

    def cycles(
        self,
        selected_date: str | None = None,
        tag: str | None = None,
        frota: str | None = None,
        classe: str | None = None,
        target: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        frame = self._cycles.copy()
        if selected_date and "data" in frame.columns:
            frame = frame.loc[frame["data"].astype(str) == str(selected_date)]
        if tag and "Tag" in frame.columns:
            frame = frame.loc[frame["Tag"].astype(str) == tag]
        if frota and "Frota" in frame.columns:
            frame = frame.loc[frame["Frota"].astype(str) == frota]
        if classe and "Classe" in frame.columns:
            frame = frame.loc[frame["Classe"].astype(str) == classe]
        if target in {"0", "1"} and "target_4h" in frame.columns:
            frame = frame.loc[frame["target_4h"].astype(int) == int(target)]
        if not frame.empty and "Fim" in frame.columns:
            frame = frame.sort_values("Fim", ascending=False)
        page_frame, total = paginate(frame, page, page_size)
        return {"total": total, "page": max(page, 1), "page_size": page_size, "items": records(page_frame)}

    def processing_summary(self, selected_date: str | None = None) -> dict[str, Any]:
        frame = self._cycles
        if selected_date and not frame.empty:
            frame = frame.loc[frame["data"].astype(str) == str(selected_date)]
        if frame.empty:
            return {
                "by_fleet": [],
                "by_class": [],
                "hourly": [],
                "duration": {"median": None, "p95": None},
            }
        count_col = "Id" if "Id" in frame.columns else frame.columns[0]
        target_col = "target_4h" if "target_4h" in frame.columns else None
        aggregations = {"ciclos": (count_col, "count")}
        if target_col:
            aggregations["positivos"] = (target_col, "sum")
        by_fleet = (
            frame.groupby("Frota", dropna=False)
            .agg(**aggregations)
            .reset_index()
            .sort_values("ciclos", ascending=False)
        )
        by_class = (
            frame.groupby("Classe", dropna=False)
            .agg(**aggregations)
            .reset_index()
            .sort_values("ciclos", ascending=False)
            .head(12)
        )
        if "Fim" in frame.columns:
            hourly = (
                frame.assign(hora=frame["Fim"].dt.hour)
                .groupby("hora", dropna=False)
                .agg(**aggregations)
                .reindex(range(24), fill_value=0)
                .reset_index()
            )
        else:
            hourly = pd.DataFrame(columns=["hora", "ciclos", "positivos"])
        duration = (
            frame["duracao_ciclo_min"].dropna()
            if "duracao_ciclo_min" in frame.columns
            else pd.Series(dtype=float)
        )
        return {
            "by_fleet": records(by_fleet),
            "by_class": records(by_class),
            "hourly": records(hourly),
            "duration": {
                "median": json_safe(duration.median()) if not duration.empty else None,
                "p95": json_safe(duration.quantile(0.95)) if not duration.empty else None,
            },
        }

    def equipment(self, tag: str) -> dict[str, Any]:
        tag_norm = str(tag).strip()
        priority = self._priority
        if not priority.empty:
            priority = priority.loc[priority["Tag"].astype(str) == tag_norm].head(30)
        cycles = self._cycles
        if not cycles.empty:
            cycles = cycles.loc[cycles["Tag"].astype(str) == tag_norm].sort_values(
                "Fim", ascending=False
            )
        events = self._events
        if not events.empty:
            events = events.loc[events["Tag"].astype(str) == tag_norm].sort_values(
                "EVENT_TIME", ascending=False
            )
        hotspot = {}
        if not self._hotspots.empty and "Tag" in self._hotspots.columns:
            match = self._hotspots.loc[self._hotspots["Tag"].astype(str) == tag_norm]
            if not match.empty:
                hotspot = records(match.head(1))[0]
        latest_cycle = records(cycles.head(1))[0] if not cycles.empty else {}
        return {
            "tag": tag_norm,
            "context": {
                "Frota": latest_cycle.get("Frota"),
                "Tipo": latest_cycle.get("Tipo"),
            },
            "hotspot": hotspot,
            "priority_history": records(priority),
            "recent_cycles": records(cycles.head(40)),
            "recent_alerts": records(events.head(40)),
            "totals": {
                "cycles": int(len(cycles)),
                "alerts": int(len(events)),
                "positives": int(cycles["target_4h"].sum()) if "target_4h" in cycles.columns else 0,
            },
        }

    def performance(self) -> dict[str, Any]:
        selected = self._selection.get("selected_model", {})
        return {
            "model": {
                "name": selected.get("model_name") or self._metrics.get("model_name"),
                "threshold": selected.get("threshold") or self._metrics.get("threshold"),
                "selection_rule": self._selection.get("selection_rule"),
            },
            "test_daily_topk": self._metrics.get("test_daily_topk_metrics", []),
            "threshold_metrics": self._metrics.get("threshold_metrics", []),
            "deduplicated_alerts": self._metrics.get("deduplicated_alerts", {}),
            "hotspots": records(self._hotspots.head(20)),
        }

    def upsert_action(
        self,
        item_type: str,
        status: str,
        tag: str | None = None,
        selected_date: str | None = None,
        event_time: str | None = None,
        cycle_id: str | None = None,
        note: str = "",
        operator: str = "",
    ) -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Status invalido: {status}")
        if item_type == "priority":
            key = f"priority:{selected_date}:{tag}"
        elif item_type == "event":
            key = f"event:{tag}:{event_time}"
        elif item_type == "cycle":
            key = f"cycle:{cycle_id}"
        else:
            raise ValueError(f"Tipo de item invalido: {item_type}")
        items = self._worklog()
        action = {
            "key": key,
            "item_type": item_type,
            "tag": tag,
            "date": selected_date,
            "event_time": event_time,
            "cycle_id": cycle_id,
            "status": status,
            "note": note,
            "operator": operator,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        items[key] = action
        self.save_worklog(items)
        return action

    def actions(self) -> list[dict[str, Any]]:
        items = list(self._worklog().values())
        items.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return items
