"""Rotulacao de alertas criticos e construcao de target de 4 horas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import LABELED_DIR, RULES_PATH, TELEMETRY_DIR, TELEMETRY_SAMPLE_PATH

CRITICAL_LEVEL = "muito alto"
HORIZON_HOURS_DEFAULT = 4
DEFAULT_TIMEZONE = "America/Sao_Paulo"
TELEMETRY_FILE_GLOB = "telemetry_*.parquet"


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _normalize_tag(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _normalize_event_time(series: pd.Series, timezone: str = DEFAULT_TIMEZONE) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        localized = parsed
    return localized.dt.tz_convert("UTC")


def load_alarm_rules(
    rules_path: Path = RULES_PATH,
    sheet_name: str = "CMA",
    critical_level: str = CRITICAL_LEVEL,
) -> pd.DataFrame:
    """Carrega regras de negocio e filtra criticidade maxima."""
    rules = pd.read_excel(rules_path, sheet_name=sheet_name)
    required_cols = ["TIPO", "EVENTO", "SITUACAO", "NIVEL"]
    missing = [column for column in required_cols if column not in rules.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes no catalogo de alarmes: {missing}")

    critical_level_norm = critical_level.casefold()
    level_series = rules["NIVEL"].fillna("").astype(str).str.strip().str.casefold()
    filtered = rules[level_series == critical_level_norm].copy()
    if filtered.empty:
        filtered = rules[level_series.str.contains(critical_level_norm, regex=False)].copy()

    return filtered[required_cols].drop_duplicates()


def list_telemetry_files(telemetry_dir: Path = TELEMETRY_DIR) -> list[Path]:
    files = sorted([path for path in telemetry_dir.glob(TELEMETRY_FILE_GLOB) if path.is_file()])
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo `{TELEMETRY_FILE_GLOB}` encontrado em {telemetry_dir}."
        )
    return files


def _resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {column.casefold(): column for column in df.columns}
    for candidate in candidates:
        column = lower_map.get(candidate.casefold())
        if column:
            return column
    return None


def prepare_event_columns(
    events_df: pd.DataFrame, timezone: str = DEFAULT_TIMEZONE
) -> pd.DataFrame:
    """Padroniza colunas de eventos para processamento de regras."""
    df = events_df.copy()

    col_tag = _resolve_column_name(df, ["TAG", "Tag", "tag"])
    col_time = _resolve_column_name(df, ["Data_Evento", "data_evento", "EVENT_TIME"])
    col_event = _resolve_column_name(df, ["Alarme", "EVENTO", "evento"])
    col_tipo = _resolve_column_name(df, ["TIPO", "Tipo"])
    col_situacao = _resolve_column_name(df, ["SITUACAO", "Situacao", "Criticidade"])
    col_nivel = _resolve_column_name(df, ["NIVEL", "Nivel"])
    col_is_dont_go = _resolve_column_name(df, ["Is_Dont_Go", "is_dont_go"])

    if not col_tag or not col_time or not col_event:
        raise ValueError("Eventos precisam conter colunas de Tag, Data_Evento e Evento/Alarme.")

    out = pd.DataFrame(
        {
            "TAG": df[col_tag].map(_normalize_tag),
            "EVENT_TIME": _normalize_event_time(df[col_time], timezone=timezone),
            "EVENTO": df[col_event],
            "TIPO": df[col_tipo] if col_tipo else "ALARME OEM",
            "SITUACAO": df[col_situacao] if col_situacao else np.nan,
            "NIVEL": df[col_nivel] if col_nivel else np.nan,
            "IS_DONT_GO": df[col_is_dont_go] if col_is_dont_go else 0,
        }
    )

    out["EVENTO_NORM"] = out["EVENTO"].map(_normalize_text)
    out["TIPO_NORM"] = out["TIPO"].map(_normalize_text)
    out["SITUACAO_NORM"] = out["SITUACAO"].map(_normalize_text)
    out["NIVEL_NORM"] = out["NIVEL"].map(_normalize_text)
    out["IS_DONT_GO"] = pd.to_numeric(out["IS_DONT_GO"], errors="coerce").fillna(0).astype(int)
    return out


def _extract_critical_from_prepared_events(
    events: pd.DataFrame, rules_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    rules = rules_df.copy()
    rules["EVENTO_NORM"] = rules["EVENTO"].map(_normalize_text)
    rules["TIPO_NORM"] = rules["TIPO"].map(_normalize_text)
    rules["SITUACAO_NORM"] = rules["SITUACAO"].map(_normalize_text)
    rules["NIVEL_NORM"] = rules["NIVEL"].map(_normalize_text)

    full_keys = ["TIPO_NORM", "EVENTO_NORM", "SITUACAO_NORM", "NIVEL_NORM"]
    event_has_full_context = events["SITUACAO_NORM"].str.len().gt(0) & events[
        "NIVEL_NORM"
    ].str.len().gt(0)
    rules_full = set(map(tuple, rules[full_keys].drop_duplicates().to_numpy()))
    rules_event = set(rules["EVENTO_NORM"].drop_duplicates().to_list())

    event_keys = pd.MultiIndex.from_frame(events[full_keys])
    full_match = event_keys.isin(rules_full) & event_has_full_context.to_numpy()
    event_only_match = events["EVENTO_NORM"].isin(rules_event) & ~event_has_full_context
    dont_go_fallback = events["IS_DONT_GO"].eq(1)

    events["MATCH_FULL_COMBO"] = full_match
    events["MATCH_EVENT_ONLY"] = event_only_match
    events["MATCH_IS_DONT_GO"] = dont_go_fallback
    events["IS_CRITICAL_EVENT"] = full_match | event_only_match | dont_go_fallback

    critical_events = events.loc[
        events["IS_CRITICAL_EVENT"] & events["EVENT_TIME"].notna() & events["TAG"].str.len().gt(0),
        ["TAG", "EVENT_TIME"],
    ].copy()
    critical_events = critical_events.sort_values(["TAG", "EVENT_TIME"]).drop_duplicates()

    audit = {
        "rows_total_events": int(len(events)),
        "rows_match_full_combo": int(events["MATCH_FULL_COMBO"].sum()),
        "rows_match_event_only": int(events["MATCH_EVENT_ONLY"].sum()),
        "rows_match_is_dont_go": int(events["MATCH_IS_DONT_GO"].sum()),
        "rows_critical_events_final": int(len(critical_events)),
    }
    return critical_events, audit


def build_critical_events(
    events_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Identifica eventos criticos para um dataframe de eventos ja carregado."""
    prepared = prepare_event_columns(events_df, timezone=timezone)
    return _extract_critical_from_prepared_events(prepared, rules_df)


def _load_telemetry_subset(file_path: Path) -> pd.DataFrame:
    expected_columns = [
        "TAG",
        "Data_Evento",
        "Alarme",
        "Tipo",
        "Criticidade",
        "Is_Dont_Go",
    ]
    return pd.read_parquet(file_path, columns=expected_columns)


def build_critical_events_from_source(
    rules_df: pd.DataFrame,
    events_source: Path | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extrai eventos criticos de arquivo unico ou diretorio de telemetria mensal."""
    source = events_source or TELEMETRY_DIR
    critical_frames: list[pd.DataFrame] = []
    aggregate_audit = {
        "files_processed": 0,
        "rows_total_events": 0,
        "rows_match_full_combo": 0,
        "rows_match_event_only": 0,
        "rows_match_is_dont_go": 0,
        "rows_critical_events_final": 0,
    }

    if source.is_dir():
        files = list_telemetry_files(source)
        for file_path in files:
            telemetry_df = _load_telemetry_subset(file_path)
            critical_df, audit = build_critical_events(
                events_df=telemetry_df,
                rules_df=rules_df,
                timezone=timezone,
            )
            critical_frames.append(critical_df)
            aggregate_audit["files_processed"] += 1
            aggregate_audit["rows_total_events"] += audit["rows_total_events"]
            aggregate_audit["rows_match_full_combo"] += audit["rows_match_full_combo"]
            aggregate_audit["rows_match_event_only"] += audit["rows_match_event_only"]
            aggregate_audit["rows_match_is_dont_go"] += audit["rows_match_is_dont_go"]
    else:
        suffix = source.suffix.lower()
        if suffix == ".parquet":
            events_df = pd.read_parquet(source)
        elif suffix == ".csv":
            events_df = pd.read_csv(source)
        elif suffix in {".xlsx", ".xls"}:
            events_df = pd.read_excel(source)
        else:
            raise ValueError(f"Formato de eventos nao suportado: {source}")

        critical_df, audit = build_critical_events(
            events_df=events_df, rules_df=rules_df, timezone=timezone
        )
        critical_frames.append(critical_df)
        aggregate_audit["files_processed"] = 1
        aggregate_audit["rows_total_events"] = audit["rows_total_events"]
        aggregate_audit["rows_match_full_combo"] = audit["rows_match_full_combo"]
        aggregate_audit["rows_match_event_only"] = audit["rows_match_event_only"]
        aggregate_audit["rows_match_is_dont_go"] = audit["rows_match_is_dont_go"]

    if critical_frames:
        critical_events = pd.concat(critical_frames, ignore_index=True)
        critical_events = (
            critical_events.drop_duplicates()
            .sort_values(["TAG", "EVENT_TIME"])
            .reset_index(drop=True)
        )
    else:
        critical_events = pd.DataFrame(columns=["TAG", "EVENT_TIME"])

    aggregate_audit["rows_critical_events_final"] = int(len(critical_events))
    return critical_events, aggregate_audit


def build_targets_from_events(
    apontamentos_df: pd.DataFrame,
    critical_events_df: pd.DataFrame,
    reference_col: str = "Fim",
    tag_col: str = "Tag",
    horizon_hours: int = HORIZON_HOURS_DEFAULT,
) -> pd.DataFrame:
    """Constroi target_4h e tte_horas por Tag a partir de eventos criticos futuros."""
    df = apontamentos_df.copy()
    df[tag_col] = df[tag_col].map(_normalize_tag)
    df[reference_col] = pd.to_datetime(df[reference_col], errors="coerce", utc=True)
    df = df.sort_values([tag_col, reference_col]).reset_index(drop=True)

    critical = critical_events_df.copy()
    critical = critical.rename(columns={"TAG": tag_col, "EVENT_TIME": "next_critical_event_time"})
    critical[tag_col] = critical[tag_col].map(_normalize_tag)
    critical["next_critical_event_time"] = pd.to_datetime(
        critical["next_critical_event_time"], errors="coerce", utc=True
    )
    critical = critical.sort_values([tag_col, "next_critical_event_time"]).reset_index(drop=True)

    critical_by_tag = {
        tag: group["next_critical_event_time"].dropna().sort_values().reset_index(drop=True)
        for tag, group in critical.groupby(tag_col, sort=False)
    }

    chunks: list[pd.DataFrame] = []
    for tag, df_tag in df.groupby(tag_col, sort=False):
        left = df_tag.sort_values(reference_col).reset_index(drop=True)
        right_series = critical_by_tag.get(tag)
        if right_series is None or right_series.empty:
            left["next_critical_event_time"] = pd.NaT
            chunks.append(left)
            continue

        left[reference_col] = pd.to_datetime(left[reference_col], errors="coerce", utc=True).astype(
            "datetime64[ns, UTC]"
        )
        right = pd.DataFrame(
            {
                "next_critical_event_time": pd.to_datetime(
                    right_series, errors="coerce", utc=True
                ).astype("datetime64[ns, UTC]")
            }
        )
        merged_tag = pd.merge_asof(
            left,
            right,
            left_on=reference_col,
            right_on="next_critical_event_time",
            direction="forward",
            allow_exact_matches=False,
        )
        chunks.append(merged_tag)

    merged = (
        pd.concat(chunks, ignore_index=True)
        .sort_values([tag_col, reference_col])
        .reset_index(drop=True)
    )
    merged["next_critical_event_time"] = pd.to_datetime(
        merged["next_critical_event_time"], errors="coerce", utc=True
    )
    merged[reference_col] = pd.to_datetime(merged[reference_col], errors="coerce", utc=True)

    delta_hours = (
        merged["next_critical_event_time"] - merged[reference_col]
    ).dt.total_seconds() / 3600.0
    merged["tte_horas"] = delta_hours.where(delta_hours > 0)
    merged["target_4h"] = merged["tte_horas"].le(horizon_hours).fillna(False).astype(int)
    return merged


def write_labeling_report(
    audit: dict[str, int],
    labeled_df: pd.DataFrame,
    output_dir: Path = LABELED_DIR,
) -> tuple[Path, Path]:
    """Persiste relatorio de auditoria da rotulacao."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "labeling_report.json"
    md_path = output_dir / "labeling_report.md"

    positives = int(labeled_df["target_4h"].sum())
    total = int(len(labeled_df))
    positive_rate = round((positives / total) * 100.0, 6) if total else 0.0

    payload = {
        **audit,
        "rows_labeled_dataset": total,
        "target_4h_positives": positives,
        "target_4h_positive_rate_pct": positive_rate,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    md_lines = [
        "# Relatorio de Rotulacao - Etapa 3b",
        "",
        "## Auditoria de eventos criticos",
        "",
        f"- Arquivos de telemetria processados: `{audit.get('files_processed', 1)}`",
        f"- Total de eventos analisados: `{audit['rows_total_events']}`",
        f"- Match completo TIPO+EVENTO+SITUACAO+NIVEL: `{audit['rows_match_full_combo']}`",
        f"- Match por EVENTO (sem contexto completo): `{audit['rows_match_event_only']}`",
        f"- Match fallback Is_Dont_Go=1: `{audit['rows_match_is_dont_go']}`",
        f"- Eventos criticos finais: `{audit['rows_critical_events_final']}`",
        f"- Tags com sobreposicao apontamento x evento: `{audit.get('tag_overlap_count', 0)}`",
        f"- Inicio eventos criticos: `{audit.get('critical_events_time_min', 'NaT')}`",
        f"- Fim eventos criticos: `{audit.get('critical_events_time_max', 'NaT')}`",
        "",
        "## Resultado do target",
        "",
        f"- Total de registros rotulados: `{total}`",
        f"- Positivos target_4h: `{positives}`",
        f"- Taxa de positivos: `{positive_rate}%`",
        "",
    ]
    md_path.write_text("\n".join(md_lines))
    return json_path, md_path


def save_labeled_dataset(labeled_df: pd.DataFrame, output_dir: Path = LABELED_DIR) -> Path:
    """Salva dataset final rotulado para as proximas etapas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "apontamentos_labeled.parquet"
    labeled_df.to_parquet(output_path, index=False)
    return output_path


def save_critical_events_dataset(
    critical_events_df: pd.DataFrame, output_dir: Path = LABELED_DIR
) -> Path:
    """Salva eventos criticos extraidos para auditoria e reuso."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "critical_events.parquet"
    critical_events_df.to_parquet(output_path, index=False)
    return output_path


def run_labeling_pipeline(
    apontamentos_df: pd.DataFrame,
    rules_path: Path = RULES_PATH,
    events_source: Path = TELEMETRY_DIR,
    output_dir: Path = LABELED_DIR,
    horizon_hours: int = HORIZON_HOURS_DEFAULT,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Executa rotulacao de ponta a ponta."""
    rules_df = load_alarm_rules(rules_path=rules_path)
    critical_events_df, audit = build_critical_events_from_source(
        rules_df=rules_df,
        events_source=events_source,
        timezone=timezone,
    )

    labeled_df = build_targets_from_events(
        apontamentos_df=apontamentos_df,
        critical_events_df=critical_events_df,
        horizon_hours=horizon_hours,
    )
    critical_events_path = save_critical_events_dataset(
        critical_events_df=critical_events_df,
        output_dir=output_dir,
    )

    tag_overlap = len(
        set(apontamentos_df["Tag"].map(_normalize_tag)) & set(critical_events_df["TAG"])
    )
    audit["tag_overlap_count"] = int(tag_overlap)
    audit["critical_events_time_min"] = str(critical_events_df["EVENT_TIME"].min())
    audit["critical_events_time_max"] = str(critical_events_df["EVENT_TIME"].max())

    labeled_path = save_labeled_dataset(labeled_df=labeled_df, output_dir=output_dir)
    report_json, report_md = write_labeling_report(
        audit=audit, labeled_df=labeled_df, output_dir=output_dir
    )

    return {
        "rules_path": str(rules_path),
        "events_source": str(events_source),
        "sample_events_path": str(TELEMETRY_SAMPLE_PATH),
        "critical_events_path": str(critical_events_path),
        "labeled_path": str(labeled_path),
        "labeling_report_json": str(report_json),
        "labeling_report_md": str(report_md),
        "target_4h_positives": int(labeled_df["target_4h"].sum()),
        "total_rows": int(len(labeled_df)),
        "rows_critical_events_final": int(audit["rows_critical_events_final"]),
        "files_processed": int(audit.get("files_processed", 1)),
    }
