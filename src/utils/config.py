"""Configurações centralizadas do projeto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = PROCESSED_DIR / "labeled"
FEATURES_DIR = PROCESSED_DIR / "features"
EXTERNAL_DIR = DATA_DIR / "external"
RULES_PATH = EXTERNAL_DIR / "regras_negocio" / "Alarmes - Regra de Negocio.xlsx"
TELEMETRY_SAMPLE_PATH = RAW_DIR / "datasets" / "datasets" / "telemetria" / "desenvolver_dontgo.xlsx"
TELEMETRY_DIR = RAW_DIR / "datasets" / "datasets" / "telemetria"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
LABELED_DATASET_PATH = LABELED_DIR / "apontamentos_labeled.parquet"
CRITICAL_EVENTS_PATH = LABELED_DIR / "critical_events.parquet"
FEATURES_DATASET_PATH = FEATURES_DIR / "features_dataset.parquet"
SPLIT_DIR = FEATURES_DIR / "splits"

# Convencao de versionamento de artefatos para rastreabilidade corporativa.
DATA_VERSION = "data_v1"
FEATURES_VERSION = "features_v1"
MODEL_VERSION = "model_v1"
