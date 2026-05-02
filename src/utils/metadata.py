"""Helpers de metadados para rastreabilidade de execucoes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.config import BASE_DIR, DATA_VERSION, FEATURES_VERSION, MODEL_VERSION


def _hash_payload(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def to_repo_relative_path(path: Path | str) -> str:
    """Normaliza caminho para formato relativo ao repositorio quando possivel."""
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
        return str(resolved.relative_to(BASE_DIR))
    except Exception:
        return str(candidate)


def build_execution_metadata(
    *,
    component: str,
    feature_count: int,
    seed: int,
    config_payload: dict[str, Any],
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    """Retorna metadados padrao para auditoria de experimentos."""
    return {
        "component": component,
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "data_version": DATA_VERSION,
        "features_version": FEATURES_VERSION,
        "model_version": MODEL_VERSION,
        "feature_count": int(feature_count),
        "seed": int(seed),
        "period_start": period_start,
        "period_end": period_end,
        "config_hash": _hash_payload(config_payload),
        "config_payload": config_payload,
    }
