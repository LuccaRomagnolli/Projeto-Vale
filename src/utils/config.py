"""Configurações centralizadas do projeto."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | None, default: Path) -> Path:
    """Resolve caminhos de env vars mantendo fallback local do projeto."""
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = _resolve_path(os.getenv("DATA_DIR"), BASE_DIR / "data")
RAW_DIR = _resolve_path(os.getenv("RAW_DATA_DIR"), DATA_DIR / "raw")
INTERIM_DIR = _resolve_path(os.getenv("INTERIM_DATA_DIR"), DATA_DIR / "interim")
PROCESSED_DIR = _resolve_path(os.getenv("PROCESSED_DATA_DIR"), DATA_DIR / "processed")
LABELED_DIR = PROCESSED_DIR / "labeled"
FEATURES_DIR = PROCESSED_DIR / "features"
EXTERNAL_DIR = _resolve_path(os.getenv("EXTERNAL_DATA_DIR"), DATA_DIR / "external")
RULES_PATH = _resolve_path(
    os.getenv("BUSINESS_RULES_PATH"),
    EXTERNAL_DIR / "regras_negocio" / "Alarmes - Regra de Negocio.xlsx",
)
TELEMETRY_DIR = _resolve_path(
    os.getenv("TELEMETRY_DIR"),
    RAW_DIR / "datasets" / "datasets" / "telemetria",
)
TELEMETRY_SAMPLE_PATH = TELEMETRY_DIR / "desenvolver_dontgo.xlsx"
REPORTS_DIR = _resolve_path(os.getenv("REPORT_DIR"), BASE_DIR / "reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = _resolve_path(os.getenv("MODEL_DIR"), BASE_DIR / "models")
LABELED_DATASET_PATH = LABELED_DIR / "apontamentos_labeled.parquet"
CRITICAL_EVENTS_PATH = LABELED_DIR / "critical_events.parquet"
FEATURES_DATASET_PATH = FEATURES_DIR / "features_dataset.parquet"
SPLIT_DIR = FEATURES_DIR / "splits"

# Convencao de versionamento de artefatos para rastreabilidade corporativa.
DATA_VERSION = "data_v1"
FEATURES_VERSION = "features_v1"
MODEL_VERSION = "model_v1"
