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

        self.tag_freq = train_df["Tag"].value_counts(normalize=True).to_dict()

        if "Operador" in train_df.columns:
            self.operador_freq = train_df["Operador"].value_counts(normalize=True).to_dict()
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

    def fit_transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta no treino e devolve o treino codificado de forma causal.

        Para as linhas de treino, `Classe_target_enc` usa media expansiva que
        exclui o proprio rotulo -- caso contrario cada linha enxergaria o
        proprio alvo. Validacao, teste e lotes novos usam `transform()`, que
        aplica a media final aprendida aqui.
        """
        self.fit(train_df)
        out = self.transform(train_df)

        if self.target_col not in train_df.columns or "Classe" not in train_df.columns:
            return out

        temporal = train_df.sort_values(self.time_col)
        target = temporal[self.target_col].astype(float)
        grouped = temporal.groupby("Classe", dropna=False)[self.target_col]

        prior_sum = grouped.cumsum() - target
        prior_count = grouped.cumcount()
        global_prior_sum = target.cumsum() - target
        global_prior_count = np.arange(len(temporal))

        global_prior = np.divide(
            global_prior_sum,
            global_prior_count,
            out=np.zeros(len(temporal), dtype=float),
            where=global_prior_count > 0,
        )
        classe_prior = np.divide(
            prior_sum,
            prior_count,
            out=np.full(len(temporal), np.nan, dtype=float),
            where=prior_count > 0,
        )
        causal = np.where(prior_count > 0, np.nan_to_num(classe_prior, nan=0.0), global_prior)
        out["Classe_target_enc"] = pd.Series(causal, index=temporal.index).reindex(out.index)
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
            out[column] = np.False_ if column.startswith(ONE_HOT_COLUMNS) else 0.0
    return out
