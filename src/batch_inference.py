"""Job de lote diario para o piloto operacional.

Le arquivos depositados num diretorio de entrada, valida cada um contra o
contrato de inferencia, pontua e grava a saida versionada por data. Um lote
reprovado nao contamina a lista de inspecao: ele vai para o diretorio de
rejeitados com o motivo registrado, e os demais seguem.

Idempotencia: reprocessar a mesma data sobrescreve deterministicamente a saida
daquela data, e o arquivo de entrada e movido para `processados/` ao final. Por
isso, rodar o job duas vezes seguidas nao duplica nem corrompe o ranking.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.inference import (
    DEFAULT_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPERATIONAL_TOP_K,
    run_inference,
)
from src.inference_contract import InferenceContractError
from src.utils.config import INTERIM_DIR, REPORTS_INFERENCE_DIR

BATCH_INPUT_DIR = INTERIM_DIR / "lotes" / "entrada"
BATCH_PROCESSED_DIR = INTERIM_DIR / "lotes" / "processados"
BATCH_REJECTED_DIR = INTERIM_DIR / "lotes" / "rejeitados"
BATCH_OUTPUT_DIR = REPORTS_INFERENCE_DIR / "lotes"
BATCH_LOG_PATH = BATCH_OUTPUT_DIR / "batch_log.json"

SUPPORTED_SUFFIXES = (".parquet", ".csv")


def discover_batches(input_dir: Path = BATCH_INPUT_DIR) -> list[Path]:
    """Lista os lotes pendentes, em ordem estavel."""
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def batch_output_paths(batch_name: str, output_dir: Path = BATCH_OUTPUT_DIR) -> dict[str, Path]:
    """Caminhos de saida derivados do nome do lote, para reprocesso deterministico."""
    stem = Path(batch_name).stem
    return {
        "scores": output_dir / f"{stem}_scores.parquet",
        "priority": output_dir / f"{stem}_priority.csv",
    }


def process_batch(
    batch_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    encoder_path: Path | None = DEFAULT_ENCODER_PATH,
    output_dir: Path = BATCH_OUTPUT_DIR,
    top_k: int = DEFAULT_OPERATIONAL_TOP_K,
) -> dict[str, Any]:
    """Pontua um unico lote. Falha de contrato vira resultado, nao excecao."""
    outputs = batch_output_paths(batch_path.name, output_dir)
    try:
        result = run_inference(
            input_path=batch_path,
            model_path=model_path,
            output_path=outputs["scores"],
            priority_output_path=outputs["priority"],
            top_k=top_k,
            encoder_path=encoder_path,
        )
    except (InferenceContractError, FileNotFoundError, ValueError) as exc:
        return {
            "batch": batch_path.name,
            "status": "rejeitado",
            "motivo": f"{type(exc).__name__}: {exc}",
        }
    return {
        "batch": batch_path.name,
        "status": "processado",
        "rows": result["rows"],
        "priority_rows": result["priority_rows"],
        "priority_days": result["priority_days"],
        "encoder_applied": result["encoder_applied"],
        "scores_path": str(outputs["scores"]),
        "priority_path": str(outputs["priority"]),
    }


def _archive(batch_path: Path, destination_dir: Path) -> Path:
    """Move o lote para o destino, sobrescrevendo em reprocesso."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / batch_path.name
    if destination.exists():
        destination.unlink()
    shutil.move(str(batch_path), str(destination))
    return destination


def run_daily_batch(
    input_dir: Path = BATCH_INPUT_DIR,
    processed_dir: Path = BATCH_PROCESSED_DIR,
    rejected_dir: Path = BATCH_REJECTED_DIR,
    output_dir: Path = BATCH_OUTPUT_DIR,
    log_path: Path = BATCH_LOG_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    encoder_path: Path | None = DEFAULT_ENCODER_PATH,
    top_k: int = DEFAULT_OPERATIONAL_TOP_K,
    executed_at: str | None = None,
) -> dict[str, Any]:
    """Processa todos os lotes pendentes e registra o resultado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = discover_batches(input_dir)

    results = []
    for batch_path in batches:
        result = process_batch(
            batch_path,
            model_path=model_path,
            encoder_path=encoder_path,
            output_dir=output_dir,
            top_k=top_k,
        )
        target = processed_dir if result["status"] == "processado" else rejected_dir
        result["arquivado_em"] = str(_archive(batch_path, target))
        results.append(result)

    summary = {
        "executed_at_utc": executed_at or datetime.now().astimezone().isoformat(),
        "input_dir": str(input_dir),
        "batches_found": len(batches),
        "batches_processed": sum(1 for r in results if r["status"] == "processado"),
        "batches_rejected": sum(1 for r in results if r["status"] == "rejeitado"),
        "results": results,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["log_path"] = str(log_path)
    return summary


def main() -> None:
    summary = run_daily_batch()
    print(f"[OK] Diretorio de entrada: {summary['input_dir']}")
    print(f"[OK] Lotes encontrados: {summary['batches_found']}")
    print(f"[OK] Lotes processados: {summary['batches_processed']}")
    print(f"[OK] Lotes rejeitados: {summary['batches_rejected']}")
    for result in summary["results"]:
        if result["status"] == "rejeitado":
            print(f"[ERROR] {result['batch']}: {result['motivo'][:160]}")
    print(f"[OK] Log do lote: {summary['log_path']}")


if __name__ == "__main__":
    main()
