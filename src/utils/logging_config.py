"""Configuracao de logging do projeto.

O pipeline se comunicava por `print("[OK] ...")` -- 96 chamadas em 14 modulos.
Isso basta enquanto um humano acompanha o terminal, mas nao serve ao piloto: o
job de lote diario roda sem supervisao, e diante de um resultado estranho nao
havia como saber quando cada etapa rodou, em que ordem, nem separar progresso
normal de condicao de erro.

Com logging o projeto ganha carimbo de tempo, nivel e destino configuravel, sem
mudar o que a operacao le no terminal: o formato padrao mantem o prefixo
`[OK]` / `[ERROR]` que a documentacao e os testes ja usavam.

Variaveis de ambiente:

    LOG_LEVEL   nivel minimo (padrao INFO)
    LOG_FILE    caminho para tambem gravar em arquivo (opcional)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

DEFAULT_LEVEL = "INFO"
LOGGER_ROOT = "vale"

# Mapeia o nivel para o prefixo que o projeto ja usava, de modo que a saida no
# terminal continue reconhecivel para quem opera o pipeline.
_PREFIX_BY_LEVEL = {
    logging.DEBUG: "[DEBUG]",
    logging.INFO: "[OK]",
    logging.WARNING: "[WARN]",
    logging.ERROR: "[ERROR]",
    logging.CRITICAL: "[ERROR]",
}


class OperationalFormatter(logging.Formatter):
    """Formata como `[NIVEL] mensagem`, preservando a convencao anterior."""

    def format(self, record: logging.LogRecord) -> str:
        prefix = _PREFIX_BY_LEVEL.get(record.levelno, f"[{record.levelname}]")
        return f"{prefix} {record.getMessage()}"


class TimestampedFormatter(logging.Formatter):
    """Formato para arquivo, onde o carimbo de tempo e o que da rastreabilidade."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


def get_logger(name: str) -> logging.Logger:
    """Logger do modulo, sob a raiz do projeto."""
    suffix = name.removeprefix("src.")
    return logging.getLogger(f"{LOGGER_ROOT}.{suffix}")


def setup_logging(level: str | None = None, log_file: str | Path | None = None) -> logging.Logger:
    """Configura os handlers uma unica vez e devolve o logger raiz do projeto.

    Chamada pelos `main()` dos entrypoints. Importar um modulo nao configura
    logging: biblioteca que mexe na configuracao global do processo surpreende
    quem a importa.
    """
    root = logging.getLogger(LOGGER_ROOT)
    resolved_level = (level or os.getenv("LOG_LEVEL", DEFAULT_LEVEL)).upper()
    root.setLevel(resolved_level)

    # Reconfigurar em nova chamada evita handlers duplicados quando um
    # entrypoint invoca outro.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(OperationalFormatter())
    root.addHandler(console)

    target = log_file or os.getenv("LOG_FILE")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(TimestampedFormatter())
        root.addHandler(file_handler)

    # A saida do projeto nao deve subir para o root logger e sair duplicada.
    root.propagate = False
    return root
