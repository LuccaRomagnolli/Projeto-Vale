from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

REQUIRED_APONTAMENTOS_COLUMNS = {"Id", "Inicio", "Fim", "Tag", "Frota", "Tipo", "Classe"}
REQUIRED_TELEMETRIA_COLUMNS = {"Data_Evento", "TAG"}


DATETIME_CANDIDATES = (
    "Inicio",
    "Fim",
    "Data_Evento",
    "Inicio_Turno",
    "Fim_Turno",
)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _to_datetime_if_present(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Formato nao suportado: {path}")


def read_apontamentos(path: str | Path) -> pd.DataFrame:
    """Le arquivo de apontamentos e padroniza os tipos basicos."""
    df = _read_table(Path(path))
    df = _standardize_columns(df)
    df = _to_datetime_if_present(df, DATETIME_CANDIDATES)

    missing = REQUIRED_APONTAMENTOS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes em apontamentos: {sorted(missing)}")

    return df


def read_telemetria(path: str | Path) -> pd.DataFrame:
    """Le arquivo de telemetria e padroniza os tipos basicos."""
    df = _read_table(Path(path))
    df = _standardize_columns(df)
    df = _to_datetime_if_present(df, DATETIME_CANDIDATES)

    if "TAG" not in df.columns and "Tag" in df.columns:
        df = df.rename(columns={"Tag": "TAG"})

    missing = REQUIRED_TELEMETRIA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes em telemetria: {sorted(missing)}")

    return df


def read_telemetria_dir(telemetria_dir: str | Path) -> pd.DataFrame:
    """Concatena arquivos de telemetria (.parquet/.xlsx) encontrados no diretorio."""
    telemetria_dir = Path(telemetria_dir)
    if not telemetria_dir.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {telemetria_dir}")

    files = sorted(
        p
        for p in telemetria_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".parquet", ".xlsx", ".xls"}
    )
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo de telemetria encontrado em {telemetria_dir}")

    chunks = [read_telemetria(path) for path in files]
    return pd.concat(chunks, ignore_index=True)


def load_alarm_rules(path: str | Path) -> pd.DataFrame:
    """Carrega todas as regras da planilha e concatena abas validas."""
    path = Path(path)
    xls = pd.ExcelFile(path)

    frames: list[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df = _standardize_columns(df)
        df["rule_sheet"] = sheet
        frames.append(df)

    all_rules = pd.concat(frames, ignore_index=True)
    return all_rules
