"""Ingestao robusta com contrato de schema para dados de apontamentos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import LABELED_DIR, RAW_DIR

REQUIRED_COLUMNS = ("Tag", "Frota", "Tipo", "Inicio", "Fim")
SUPPORTED_EXTENSIONS = (".parquet", ".csv", ".xlsx", ".xls")
DEFAULT_TIMEZONE = "America/Sao_Paulo"
MAX_CYCLE_HOURS_DEFAULT = 24


def find_apontamentos_file(raw_dir: Path = RAW_DIR) -> Path:
    """Localiza o arquivo de apontamentos em data/raw, priorizando parquet."""
    candidates = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "apontamento" in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo de apontamentos encontrado em {raw_dir} "
            f"com extensoes {SUPPORTED_EXTENSIONS}."
        )

    extension_priority = {".parquet": 0, ".csv": 1, ".xlsx": 2, ".xls": 3}
    candidates.sort(key=lambda item: (extension_priority[item.suffix.lower()], str(item)))
    return candidates[0]


def read_dataset(path: Path) -> pd.DataFrame:
    """Carrega dataset por extensao."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato nao suportado: {path}")


def validate_required_columns(
    df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> None:
    """Valida colunas obrigatorias do contrato de dados."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")


def _normalize_datetime_series(series: pd.Series, timezone: str) -> pd.Series:
    """Converte serie temporal para UTC preservando robustez em dados invalidos."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    else:
        localized = parsed
    return localized.dt.tz_convert("UTC")


def standardize_datetime_columns(
    df: pd.DataFrame,
    start_col: str = "Inicio",
    end_col: str = "Fim",
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Padroniza colunas de data/hora para UTC."""
    out = df.copy()
    out[start_col] = _normalize_datetime_series(out[start_col], timezone=timezone)
    out[end_col] = _normalize_datetime_series(out[end_col], timezone=timezone)
    return out


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def build_quality_report(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
    start_col: str = "Inicio",
    end_col: str = "Fim",
    max_cycle_hours: int = MAX_CYCLE_HOURS_DEFAULT,
) -> dict[str, Any]:
    """Gera relatorio de qualidade com nulos, duplicatas e outliers."""
    total_records = int(len(df))
    report: dict[str, Any] = {
        "total_records": total_records,
        "required_columns": list(required_columns),
        "missing_by_column": {},
        "missing_by_column_pct": {},
    }

    for column in required_columns:
        missing = int(df[column].isna().sum())
        report["missing_by_column"][column] = missing
        report["missing_by_column_pct"][column] = _safe_ratio(missing, total_records)

    duplicate_rows = int(df.duplicated().sum())
    report["duplicate_full_rows"] = duplicate_rows
    report["duplicate_full_rows_pct"] = _safe_ratio(duplicate_rows, total_records)

    key_columns = [col for col in ("Tag", start_col, end_col) if col in df.columns]
    if len(key_columns) == 3:
        duplicate_keys = int(df.duplicated(subset=key_columns).sum())
    else:
        duplicate_keys = 0
    report["duplicate_tag_inicio_fim"] = duplicate_keys
    report["duplicate_tag_inicio_fim_pct"] = _safe_ratio(duplicate_keys, total_records)

    duration_hours = (df[end_col] - df[start_col]).dt.total_seconds() / 3600.0
    negative_duration = int((duration_hours < 0).sum())
    above_threshold = int((duration_hours > max_cycle_hours).sum())
    missing_duration = int(duration_hours.isna().sum())

    report["invalid_inicio"] = int(df[start_col].isna().sum())
    report["invalid_fim"] = int(df[end_col].isna().sum())
    report["duration_negative_count"] = negative_duration
    report["duration_gt_max_cycle_hours_count"] = above_threshold
    report["duration_missing_count"] = missing_duration
    report["max_cycle_hours_threshold"] = max_cycle_hours

    return report


def write_quality_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Persiste relatorio de qualidade em JSON e Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report_ingestao.json"
    md_path = output_dir / "quality_report_ingestao.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md_lines = [
        "# Relatório de Qualidade - Etapa 2 (Ingestão)",
        "",
        f"- Total de registros: `{report['total_records']}`",
        "",
        "## Nulos por coluna obrigatória",
        "",
        "| Coluna | Nulos | Percentual |",
        "|---|---:|---:|",
    ]
    for column, count in report["missing_by_column"].items():
        pct = report["missing_by_column_pct"][column]
        md_lines.append(f"| {column} | {count} | {pct}% |")

    md_lines.extend(
        [
            "",
            "## Duplicatas",
            "",
            f"- Linhas totalmente duplicadas: `{report['duplicate_full_rows']}` "
            f"({report['duplicate_full_rows_pct']}%)",
            f"- Duplicatas por chave Tag+Inicio+Fim: `{report['duplicate_tag_inicio_fim']}` "
            f"({report['duplicate_tag_inicio_fim_pct']}%)",
            "",
            "## Duração de ciclo",
            "",
            f"- Duração negativa: `{report['duration_negative_count']}`",
            f"- Duração acima de `{report['max_cycle_hours_threshold']}h`: "
            f"`{report['duration_gt_max_cycle_hours_count']}`",
            f"- Duração ausente: `{report['duration_missing_count']}`",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines))
    return json_path, md_path


def save_validated_snapshot(df: pd.DataFrame, output_dir: Path = LABELED_DIR) -> Path:
    """Salva snapshot validado para uso nas proximas etapas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "apontamentos_validado.parquet"
    df.to_parquet(snapshot_path, index=False)
    return snapshot_path


def run_ingestion_contract(
    input_path: Path | None = None,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = LABELED_DIR,
    timezone: str = DEFAULT_TIMEZONE,
    max_cycle_hours: int = MAX_CYCLE_HOURS_DEFAULT,
) -> dict[str, Any]:
    """Executa pipeline completo de ingestao e contrato de dados."""
    source_path = input_path or find_apontamentos_file(raw_dir=raw_dir)
    df = read_dataset(source_path)
    validate_required_columns(df)
    df = standardize_datetime_columns(df, timezone=timezone)
    report = build_quality_report(df, max_cycle_hours=max_cycle_hours)
    snapshot_path = save_validated_snapshot(df, output_dir=output_dir)
    json_report_path, md_report_path = write_quality_report(report, output_dir=output_dir)

    return {
        "source_path": str(source_path),
        "snapshot_path": str(snapshot_path),
        "quality_report_json": str(json_report_path),
        "quality_report_md": str(md_report_path),
        "total_records": report["total_records"],
    }
