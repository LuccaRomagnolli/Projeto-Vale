"""Encodings categoricos ajustados apenas no treino.

As estatisticas categoricas (frequencia de Tag, frequencia de Operador, media
alvo por Classe e as categorias one-hot de Frota/Tipo) eram calculadas sobre o
dataset inteiro antes do split temporal. Isso vazava a distribuicao futura para
o treino e, pior, codificava as linhas de validacao e teste usando a propria
massa de frequencia delas. `Tag_freq` era a segunda feature mais importante do
modelo promovido, entao o efeito era material.

Aqui o encoder segue o contrato fit/transform: aprende no treino e aplica em
validacao, teste e em qualquer lote novo de inferencia. Categorias nunca vistas
no treino recebem valor neutro em vez de estatistica propria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

TARGET_COL = "target_4h"
TIME_COL = "Fim"
ONE_HOT_COLUMNS = ("Frota", "Tipo")

# Blocos cronologicos usados no target encoding out-of-fold do treino. Mais
# blocos aproximam a media do historico completo; menos blocos reduzem o numero
# de valores distintos, aproximando a estrutura da inferencia.
DEFAULT_ENCODING_BLOCKS = 5


class CategoricalEncoder:
    """Aprende estatisticas categoricas no treino e as aplica sem vazamento."""

    def __init__(self, target_col: str = TARGET_COL, time_col: str = TIME_COL) -> None:
        self.target_col = target_col
        self.time_col = time_col
        self.tag_freq: dict[str, float] = {}
        self.operador_freq: dict[str, float] = {}
        self.classe_target: dict[str, float] = {}
        self.global_target_mean: float = 0.0
        self.one_hot_columns: dict[str, list[str]] = {}
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> CategoricalEncoder:
        """Aprende todas as estatisticas usando exclusivamente linhas de treino."""
        if train_df.empty:
            raise ValueError("Nao e possivel ajustar o encoder em um conjunto de treino vazio.")

        self.tag_freq = {
            str(key): float(value)
            for key, value in train_df["Tag"].value_counts(normalize=True).items()
        }

        if "Operador" in train_df.columns:
            self.operador_freq = {
                str(key): float(value)
                for key, value in train_df["Operador"].value_counts(normalize=True).items()
            }
        else:
            self.operador_freq = {}

        has_target = self.target_col in train_df.columns
        self.global_target_mean = float(train_df[self.target_col].mean()) if has_target else 0.0
        if has_target and "Classe" in train_df.columns:
            grouped = train_df.groupby("Classe", dropna=False)[self.target_col].mean()
            self.classe_target = {str(key): float(value) for key, value in grouped.items()}
        else:
            self.classe_target = {}

        self.one_hot_columns = {
            column: sorted(train_df[column].dropna().astype(str).unique().tolist())
            for column in ONE_HOT_COLUMNS
            if column in train_df.columns
        }

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica as estatisticas aprendidas, sem recalcular nada a partir de `df`."""
        if not self.fitted:
            raise ValueError("Encoder nao ajustado: chame fit() antes de transform().")

        out = df.copy()

        # Categoria ausente no treino recebe 0.0: e a leitura honesta de
        # "equipamento sem historico conhecido", nao a frequencia do proprio lote.
        out["Tag_freq"] = out["Tag"].map(self.tag_freq).fillna(0.0).astype(float)

        if "Operador" in out.columns and self.operador_freq:
            out["Operador_freq"] = out["Operador"].map(self.operador_freq).fillna(0.0).astype(float)
        else:
            out["Operador_freq"] = 0.0

        # Classe desconhecida cai para a media global do treino.
        if "Classe" in out.columns and self.classe_target:
            out["Classe_target_enc"] = (
                out["Classe"]
                .astype(str)
                .map(self.classe_target)
                .fillna(self.global_target_mean)
                .astype(float)
            )
        else:
            out["Classe_target_enc"] = self.global_target_mean

        for column, categories in self.one_hot_columns.items():
            if column not in out.columns:
                continue
            values = out[column].astype(str)
            for category in categories:
                out[f"{column}_{category}"] = (values == category).astype(bool)
            out = out.drop(columns=[column])

        return out

    def fit_transform_train(
        self,
        train_df: pd.DataFrame,
        n_blocks: int = DEFAULT_ENCODING_BLOCKS,
    ) -> pd.DataFrame:
        """Ajusta no treino e devolve o treino com target encoding out-of-fold temporal.

        O treino e dividido em `n_blocks` blocos cronologicos. Cada bloco recebe
        a media por Classe calculada apenas nos blocos ANTERIORES, o que atende
        duas exigencias ao mesmo tempo:

        1. Causalidade -- nenhuma linha enxerga o proprio rotulo nem qualquer
           informacao futura.
        2. Mesma estrutura da inferencia -- o valor e uma media por classe, e
           nao um numero diferente por linha.

        A versao anterior usava media expansiva linha a linha. Era causal, mas
        produzia 258174 valores distintos no treino contra 4 na inferencia: o
        modelo treinava numa distribuicao e recebia outra em producao. O
        monitoramento de drift flagrou essa divergencia com PSI 7.71.

        O primeiro bloco nao tem historico anterior e recebe `0.0`, valor que
        nao carrega informacao alguma sobre o alvo. A alternativa aparentemente
        inofensiva -- usar a media global do treino, que e o que `transform()`
        aplica a classe desconhecida -- vaza: quando o alvo muda de regime ao
        longo do periodo, essa media resume tambem o futuro, e as linhas do
        primeiro bloco passam a carrega-lo. O custo e um nivel a mais na
        distribuicao do treino, contra a integridade da causalidade.
        """
        self.fit(train_df)
        out = self.transform(train_df)

        if self.target_col not in train_df.columns or "Classe" not in train_df.columns:
            return out
        if train_df.empty:
            return out

        temporal = train_df.sort_values(self.time_col)
        blocks = np.array_split(np.arange(len(temporal)), min(n_blocks, len(temporal)))
        encoded = pd.Series(0.0, index=temporal.index, dtype=float)

        for position, block in enumerate(blocks):
            if position == 0 or len(block) == 0:
                continue  # primeiro bloco fica em 0.0: nao ha historico anterior
            history = temporal.iloc[: int(block[0])]
            means = history.groupby("Classe", dropna=False)[self.target_col].mean()
            history_prior = float(history[self.target_col].mean())
            block_index = temporal.index[block]
            encoded.loc[block_index] = (
                temporal.loc[block_index, "Classe"].map(means).fillna(history_prior).astype(float)
            )

        out["Classe_target_enc"] = encoded.reindex(out.index)
        return out

    def to_payload(self) -> dict[str, Any]:
        """Representacao serializavel, util para auditoria dos relatorios."""
        return {
            "tag_categories": len(self.tag_freq),
            "operador_categories": len(self.operador_freq),
            "classe_categories": len(self.classe_target),
            "global_target_mean": self.global_target_mean,
            "one_hot_columns": {k: len(v) for k, v in self.one_hot_columns.items()},
        }


def save_encoder(encoder: CategoricalEncoder, path: Path) -> Path:
    """Persiste o encoder ajustado para reuso em inferencia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)
    return path


def load_encoder(path: Path) -> CategoricalEncoder:
    """Carrega um encoder previamente ajustado."""
    if not path.exists():
        raise FileNotFoundError(
            f"Encoder categorico nao encontrado: {path}. Execute `python tasks.py train-baseline`."
        )
    encoder = joblib.load(path)
    if not isinstance(encoder, CategoricalEncoder) or not encoder.fitted:
        raise ValueError(f"Arquivo nao contem um CategoricalEncoder ajustado: {path}")
    return encoder


def align_encoded_columns(df: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    """Garante que `df` tenha exatamente as colunas de `reference_columns`."""
    out = df.copy()
    for column in reference_columns:
        if column not in out.columns:
            # Coluna nova recebe escalar; os stubs do pandas nao modelam
            # atribuicao escalar a coluna inexistente.
            out[column] = False if column.startswith(ONE_HOT_COLUMNS) else 0.0
    return out
