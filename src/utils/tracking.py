"""Rastreamento de experimentos com MLflow.

Os relatorios em `reports/` descrevem sempre a ULTIMA execucao: cada rodada de
`model-selection` sobrescreve os arquivos anteriores. Nao havia como comparar
duas execucoes, nem responder "o que mudou desde a rodada passada" sem recorrer
ao historico do git dos artefatos.

Este modulo registra params, metricas e artefatos por execucao num store local
(`mlruns/`, fora do versionamento). O rastreamento e deliberadamente opcional: o
pipeline precisa rodar em maquina sem MLflow instalado, entao toda falha aqui
degrada para no-op em vez de interromper um treino de 90 trials.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.utils.config import BASE_DIR

DEFAULT_EXPERIMENT = "vale-model-selection"
TRACKING_DIR = BASE_DIR / "mlruns"
# O MLflow 3.x colocou o backend de arquivos em modo de manutencao e recusa
# gravar nele. O store precisa ser um banco; sqlite mantem tudo local e sem
# servico externo, adequado ao piloto em maquina unica.
TRACKING_DB_NAME = "mlflow.db"

# Numeros e textos longos poluem a UI e nao cabem como parametro do MLflow.
MAX_PARAM_CHARS = 500


def tracking_enabled() -> bool:
    """Rastreamento ligado por padrao; desligavel com `MLFLOW_TRACKING=0`."""
    return os.getenv("MLFLOW_TRACKING", "1").strip().lower() not in {"0", "false", "no"}


def _import_mlflow() -> Any | None:
    if not tracking_enabled():
        return None
    try:
        import mlflow
    except ModuleNotFoundError:
        return None
    return mlflow


@contextmanager
def track_run(
    run_name: str,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_dir: Path = TRACKING_DIR,
) -> Iterator[RunLogger]:
    """Abre uma execucao rastreada; vira no-op se o MLflow nao estiver disponivel."""
    mlflow = _import_mlflow()
    if mlflow is None:
        yield RunLogger(None, reason="mlflow indisponivel ou desligado por MLFLOW_TRACKING")
        return

    try:
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / TRACKING_DB_NAME).resolve()}")
        mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001 - rastreamento nunca derruba o pipeline
        # Degradar, mas nunca em silencio: a primeira versao engolia a excecao
        # e o rastreamento ficava desligado sem qualquer indicacao de que havia
        # falhado -- foi assim que a mudanca de backend do MLflow 3.x passou
        # despercebida.
        yield RunLogger(None, reason=f"{type(exc).__name__}: {exc}")
        return

    try:
        with mlflow.start_run(run_name=run_name):
            yield RunLogger(mlflow)
    except Exception as exc:  # noqa: BLE001
        yield RunLogger(None, reason=f"{type(exc).__name__}: {exc}")


class RunLogger:
    """Fachada fina sobre o MLflow que ignora falhas de registro.

    Um erro ao gravar metadados nao pode invalidar um treino que levou minutos,
    entao cada operacao e protegida individualmente.
    """

    def __init__(self, mlflow: Any | None, reason: str | None = None) -> None:
        self._mlflow = mlflow
        self.reason = reason

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        cleaned = {
            key: str(value)[:MAX_PARAM_CHARS] for key, value in params.items() if value is not None
        }
        try:
            self._mlflow.log_params(cleaned)
        except Exception:  # noqa: BLE001
            pass

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        try:
            self._mlflow.log_metrics(numeric)
        except Exception:  # noqa: BLE001
            pass

    def log_artifact(self, path: Path | str) -> None:
        if self._mlflow is None:
            return
        candidate = Path(path)
        if not candidate.exists():
            return
        try:
            self._mlflow.log_artifact(str(candidate))
        except Exception:  # noqa: BLE001
            pass

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.set_tags({k: str(v)[:MAX_PARAM_CHARS] for k, v in tags.items()})
        except Exception:  # noqa: BLE001
            pass
