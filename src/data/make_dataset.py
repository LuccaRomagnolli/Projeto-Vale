"""Orquestracao da Etapa 2: ingestao e contrato de dados."""

from pathlib import Path

import pandas as pd

from src.alert_labeler import run_labeling_pipeline
from src.data_loader import read_dataset, run_ingestion_contract
from src.utils.config import TELEMETRY_DIR
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Wrapper mantido para compatibilidade com testes existentes."""
    return read_dataset(path)


def main() -> None:
    setup_logging()
    ingestion = run_ingestion_contract()
    logger.info(f"Fonte carregada: {ingestion['source_path']}")
    logger.info(f"Snapshot validado: {ingestion['snapshot_path']}")
    logger.info(f"Relatorio JSON: {ingestion['quality_report_json']}")
    logger.info(f"Relatorio Markdown: {ingestion['quality_report_md']}")
    logger.info(f"Registros processados: {ingestion['total_records']}")

    snapshot_df = pd.read_parquet(ingestion["snapshot_path"])
    labeling = run_labeling_pipeline(
        apontamentos_df=snapshot_df,
        events_source=TELEMETRY_DIR,
    )
    logger.info(f"Dataset rotulado: {labeling['labeled_path']}")
    logger.info(f"Eventos criticos extraidos: {labeling['critical_events_path']}")
    logger.info(f"Label report JSON: {labeling['labeling_report_json']}")
    logger.info(f"Label report Markdown: {labeling['labeling_report_md']}")
    logger.info(f"Arquivos de telemetria processados: {labeling['files_processed']}")
    logger.info(f"Eventos criticos finais: {labeling['rows_critical_events_final']}")
    logger.info(
        "Positivos target_4h: " f"{labeling['target_4h_positives']} de {labeling['total_rows']}"
    )


if __name__ == "__main__":
    main()
