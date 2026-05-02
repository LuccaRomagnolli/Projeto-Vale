"""Orquestracao da Etapa 2: ingestao e contrato de dados."""

from pathlib import Path

import pandas as pd

from src.alert_labeler import run_labeling_pipeline
from src.data_loader import read_dataset, run_ingestion_contract
from src.utils.config import TELEMETRY_DIR


def load_raw_data(path: Path) -> pd.DataFrame:
    """Wrapper mantido para compatibilidade com testes existentes."""
    return read_dataset(path)


def main() -> None:
    ingestion = run_ingestion_contract()
    print(f"[OK] Fonte carregada: {ingestion['source_path']}")
    print(f"[OK] Snapshot validado: {ingestion['snapshot_path']}")
    print(f"[OK] Relatorio JSON: {ingestion['quality_report_json']}")
    print(f"[OK] Relatorio Markdown: {ingestion['quality_report_md']}")
    print(f"[OK] Registros processados: {ingestion['total_records']}")

    snapshot_df = pd.read_parquet(ingestion["snapshot_path"])
    labeling = run_labeling_pipeline(
        apontamentos_df=snapshot_df,
        events_source=TELEMETRY_DIR,
    )
    print(f"[OK] Dataset rotulado: {labeling['labeled_path']}")
    print(f"[OK] Eventos criticos extraidos: {labeling['critical_events_path']}")
    print(f"[OK] Label report JSON: {labeling['labeling_report_json']}")
    print(f"[OK] Label report Markdown: {labeling['labeling_report_md']}")
    print(f"[OK] Arquivos de telemetria processados: {labeling['files_processed']}")
    print(f"[OK] Eventos criticos finais: {labeling['rows_critical_events_final']}")
    print(
        "[OK] Positivos target_4h: "
        f"{labeling['target_4h_positives']} de {labeling['total_rows']}"
    )


if __name__ == "__main__":
    main()
