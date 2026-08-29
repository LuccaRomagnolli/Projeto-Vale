"""Fixtures compartilhadas da suite.

O projeto nao tinha `conftest.py` nem uma unica `@pytest.fixture`. Cada arquivo
de teste trazia seu proprio construtor de dados sinteticos -- eram doze,
divergindo em colunas, tipos e periodo -- e cada teste de pipeline redirecionava
os caminhos de saida com blocos de ate onze `monkeypatch.setattr`.

O custo disso nao era so repeticao: quando o schema mudava, cada construtor
precisava ser corrigido separadamente, e um esquecido produzia um teste que
passava sobre dados que nao existem mais.

Aqui ficam apenas os blocos genuinamente comuns. Construtores especificos de um
modulo continuam no arquivo do modulo, onde o contexto ajuda a ler o teste.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

CYCLE_COLUMNS = ("Id", "Inicio", "Fim", "Tag", "Frota", "Tipo", "Classe", "target_4h")


@pytest.fixture
def cycles_frame() -> Callable[..., pd.DataFrame]:
    """Ciclos rotulados no formato produzido pela etapa de rotulacao.

    Devolve a funcao, e nao o DataFrame, porque a maioria dos testes precisa
    variar tamanho, periodo ou prevalencia.
    """

    def _build(
        n_rows: int = 200,
        start: str = "2026-01-01",
        freq: str = "h",
        positives_every: int = 4,
    ) -> pd.DataFrame:
        inicio = pd.date_range(start, periods=n_rows, freq=freq, tz="UTC")
        return pd.DataFrame(
            {
                "Id": range(n_rows),
                "Inicio": inicio,
                "Fim": inicio + pd.Timedelta(30, unit="min"),
                "Tag": [f"CA{i % 5}" for i in range(n_rows)],
                "Frota": [f"F{i % 2}" for i in range(n_rows)],
                "Tipo": ["Caminhao" if i % 2 else "Carregadeira" for i in range(n_rows)],
                "Classe": ["Operando" if i % 3 else "Parado" for i in range(n_rows)],
                "duracao_ciclo_min": [30.0] * n_rows,
                "target_4h": [int(i % positives_every == 0) for i in range(n_rows)],
            }
        )

    return _build


@pytest.fixture
def critical_events_frame() -> Callable[..., pd.DataFrame]:
    """Eventos criticos no formato de `critical_events.parquet`.

    Mantem `TAG`/`EVENT_TIME` em maiusculas, como o arquivo real -- e a
    diferenca de nomenclatura entre as duas fontes ja causou defeito antes.
    """

    def _build(n_rows: int = 20, start: str = "2026-01-01") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TAG": [f"CA{i % 5}" for i in range(n_rows)],
                "EVENT_TIME": pd.date_range(start, periods=n_rows, freq="6h", tz="UTC"),
            }
        )

    return _build


@pytest.fixture
def scored_frame() -> Callable[..., pd.DataFrame]:
    """Saida de pontuacao, no formato consumido pelo scorecard operacional."""

    def _build(n_rows: int = 120, threshold: float = 0.5, seed: int = 11) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        fim = pd.date_range("2026-02-01", periods=n_rows, freq="2h", tz="UTC")
        scores = rng.random(n_rows)
        return pd.DataFrame(
            {
                "Id": range(n_rows),
                "Tag": [f"CA{i % 6}" for i in range(n_rows)],
                "Fim": fim,
                "data": fim.date,
                "split": "test",
                "score": scores,
                "threshold": threshold,
                "prediction": (scores >= threshold).astype(int),
                "target_4h": rng.integers(0, 2, n_rows),
            }
        )

    return _build


@pytest.fixture
def redirect_paths(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, dict[str, Any]], None]:
    """Redireciona constantes de caminho de um modulo para um diretorio temporario.

    Substitui os blocos repetidos de `monkeypatch.setattr` nos testes de
    pipeline. Como as constantes sao lidas do modulo em tempo de execucao, o
    redirecionamento vale apenas enquanto o teste roda.
    """

    def _redirect(module_path: str, mapping: dict[str, Any]) -> None:
        for attribute, value in mapping.items():
            monkeypatch.setattr(f"{module_path}.{attribute}", value)

    return _redirect


@pytest.fixture
def log_records() -> Any:
    """Captura os registros do logger do projeto.

    O `caplog` do pytest instala seu handler no logger raiz, e o logger do
    projeto tem `propagate=False` para nao duplicar saida no terminal -- logo o
    caplog nao enxerga nada. Esta fixture anexa um handler diretamente em
    `vale`, permitindo afirmar sobre nivel e mensagem em vez de sobre o texto
    formatado que sai no stdout.
    """
    import logging

    from src.utils.logging_config import LOGGER_ROOT

    class _Collector(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

        def messages(self, level: int | None = None) -> list[str]:
            return [r.getMessage() for r in self.records if level is None or r.levelno == level]

    logger = logging.getLogger(LOGGER_ROOT)
    handler = _Collector()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class LinearTestModel:
    """Score proporcional a primeira feature, suficiente para ordenar.

    Definida no nivel do modulo, e nao dentro da fixture: uma classe local nao
    e serializavel, e os testes que gravam o artefato com joblib falhariam.
    """

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        positive = np.clip(x.iloc[:, 0].to_numpy(dtype=float) / 10.0, 0, 1)
        return np.column_stack([1 - positive, positive])


@pytest.fixture
def artifact_factory() -> Callable[..., dict[str, Any]]:
    """Artefato de modelo no contrato minimo exigido pela inferencia."""

    def _build(
        feature_columns: list[str] | None = None,
        threshold: float = 0.5,
        model_name: str = "modelo_de_teste",
    ) -> dict[str, Any]:
        return {
            "model": LinearTestModel(),
            "model_name": model_name,
            "feature_columns": feature_columns or ["n_alertas_4h", "duracao_ciclo_min"],
            "threshold": threshold,
        }

    return _build


@pytest.fixture
def split_dir(tmp_path: Path, cycles_frame: Callable[..., pd.DataFrame]) -> Path:
    """Diretorio com os tres parquets de split, no layout esperado por `load_splits`."""
    directory = tmp_path / "splits"
    directory.mkdir(parents=True, exist_ok=True)
    frame = cycles_frame(300)
    frame["n_alertas_4h"] = [float(i % 4) for i in range(len(frame))]
    frame.iloc[:200].to_parquet(directory / "features_train.parquet", index=False)
    frame.iloc[200:250].to_parquet(directory / "features_val.parquet", index=False)
    frame.iloc[250:].to_parquet(directory / "features_test.parquet", index=False)
    return directory
