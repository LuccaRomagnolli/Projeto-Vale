"""Configuracao de logging: o que a operacao le e o que fica registrado."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.logging_config import (
    LOGGER_ROOT,
    OperationalFormatter,
    get_logger,
    setup_logging,
)


def test_console_format_preserves_the_previous_convention() -> None:
    """O terminal continua mostrando `[OK]`, como a documentacao descreve."""
    formatter = OperationalFormatter()
    record = logging.LogRecord(
        "vale.teste", logging.INFO, __file__, 1, "pipeline concluido", None, None
    )
    assert formatter.format(record) == "[OK] pipeline concluido"


def test_error_level_uses_the_error_prefix() -> None:
    formatter = OperationalFormatter()
    record = logging.LogRecord("vale.teste", logging.ERROR, __file__, 1, "falhou", None, None)
    assert formatter.format(record) == "[ERROR] falhou"


def test_module_logger_lives_under_the_project_root() -> None:
    assert get_logger("src.models.train_model").name == f"{LOGGER_ROOT}.models.train_model"
    # o prefixo `src.` sai do nome, que ja e ruido para quem le o log
    assert "src." not in get_logger("src.inference").name


def test_setup_writes_to_stdout(capsys) -> None:
    setup_logging(level="INFO")
    get_logger("src.teste").info("mensagem operacional")
    assert "[OK] mensagem operacional" in capsys.readouterr().out


def test_repeated_setup_does_not_duplicate_output(capsys) -> None:
    """Um entrypoint pode invocar outro; handlers acumulados duplicariam o log."""
    setup_logging()
    setup_logging()
    setup_logging()

    get_logger("src.teste").info("uma vez")

    assert capsys.readouterr().out.count("uma vez") == 1


def test_log_file_receives_timestamped_records(tmp_path: Path) -> None:
    """No job diario sem supervisao, o carimbo de tempo e o que da rastreabilidade."""
    log_file = tmp_path / "logs" / "pipeline.log"
    setup_logging(level="INFO", log_file=log_file)

    get_logger("src.teste").info("etapa concluida")
    logging.getLogger(LOGGER_ROOT).handlers[-1].flush()

    content = log_file.read_text(encoding="utf-8")
    assert "etapa concluida" in content
    assert "INFO" in content
    assert "vale.teste" in content
    setup_logging()  # remove o handler de arquivo para nao vazar entre testes


def test_level_can_be_raised_to_silence_info(capsys) -> None:
    setup_logging(level="ERROR")
    logger = get_logger("src.teste")
    logger.info("nao deve aparecer")
    logger.error("deve aparecer")

    out = capsys.readouterr().out
    assert "nao deve aparecer" not in out
    assert "[ERROR] deve aparecer" in out
    setup_logging()


def test_records_expose_level_for_assertions(log_records) -> None:
    """A fixture permite afirmar sobre nivel, e nao sobre o texto formatado."""
    logger = get_logger("src.teste")
    logger.info("progresso")
    logger.error("problema")

    assert log_records.messages(logging.INFO) == ["progresso"]
    assert log_records.messages(logging.ERROR) == ["problema"]
