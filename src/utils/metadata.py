"""Helpers de metadados para rastreabilidade de execucoes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
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
    except (OSError, ValueError, RuntimeError):
        return str(candidate)


def git_revision() -> str | None:
    """SHA do commit atual, quando o codigo roda dentro de um repositorio git.

    Sem isto, um artefato promovido nao permite reconstruir o codigo que o
    gerou -- e a pergunta "qual versao produziu este modelo" nao tem resposta.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def library_versions() -> dict[str, str]:
    """Versoes das bibliotecas que determinam o comportamento do modelo."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def file_digest(path: Path | str, chunk_size: int = 1 << 20) -> str | None:
    """SHA-256 do arquivo, para amarrar o artefato aos dados que o treinaram."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def build_artifact_provenance(
    *,
    seed: int,
    training_period: dict[str, str] | None = None,
    data_path: Path | str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Procedencia embutida no artefato promovido.

    O `.joblib` guardava apenas modelo, colunas e threshold. Sem versoes de
    biblioteca, revisao do codigo e hash dos dados, um artefato encontrado no
    disco e irreprodutivel: nao da para dizer com que codigo, com que
    dependencias nem sobre quais dados ele foi treinado.
    """
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "library_versions": library_versions(),
        "seed": int(seed),
        "training_period": training_period or {},
        "training_data_sha256": file_digest(data_path) if data_path else None,
        "metrics": metrics or {},
    }


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
