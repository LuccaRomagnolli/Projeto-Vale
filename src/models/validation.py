"""Validacao temporal e metricas compartilhadas de modelagem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "target_4h"
TIME_COL = "Fim"

# O alvo `target_4h` olha 4 horas para frente. Sem um intervalo de guarda, as
# ultimas horas do treino tem rotulo determinado por eventos que caem dentro da
# validacao -- e o mesmo ocorre na fronteira validacao/teste.
LABEL_HORIZON_HOURS = 4.0


def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_hours: float = LABEL_HORIZON_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Divide dados por ordem temporal, com corte por calendario e embargo.

    O corte usa o timestamp da fronteira, e nao a posicao da linha, para que
    ciclos com o mesmo `Fim` nunca fiquem dos dois lados da divisao. Em seguida
    aplica um embargo de `embargo_hours` no fim do treino e da validacao,
    descartando as linhas cujo horizonte de rotulo invade o bloco seguinte.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Frações inválidas para split temporal.")
    if embargo_hours < 0:
        raise ValueError("embargo_hours nao pode ser negativo.")

    ordered = df.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    n_rows = len(ordered)
    if n_rows == 0:
        raise ValueError("Nenhuma linha com timestamp valido para dividir.")

    times = ordered[time_col]
    train_boundary = times.iloc[min(int(n_rows * train_frac), n_rows - 1)]
    val_boundary = times.iloc[min(int(n_rows * (train_frac + val_frac)), n_rows - 1)]

    embargo = pd.Timedelta(embargo_hours, unit="h")

    # Corte por timestamp: empates ficam inteiramente do lado direito.
    train_all = times < train_boundary
    val_all = (times >= train_boundary) & (times < val_boundary)
    test_mask = times >= val_boundary

    # Embargo: remove a cauda cujo rotulo depende do bloco seguinte.
    train_mask = train_all & (times < train_boundary - embargo)
    val_mask = val_all & (times < val_boundary - embargo)

    train = ordered.loc[train_mask].copy()
    val = ordered.loc[val_mask].copy()
    test = ordered.loc[test_mask].copy()

    for name, frame in (("treino", train), ("validacao", val), ("teste", test)):
        if frame.empty:
            raise ValueError(
                f"Split temporal produziu {name} vazio "
                f"(embargo_hours={embargo_hours}). Reduza o embargo ou use mais dados."
            )

    metadata = {
        "rows_total": int(n_rows),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "embargo_hours": float(embargo_hours),
        "rows_dropped_embargo_train": int(train_all.sum() - len(train)),
        "rows_dropped_embargo_val": int(val_all.sum() - len(val)),
        "train_start": str(train[time_col].min()),
        "train_end": str(train[time_col].max()),
        "val_start": str(val[time_col].min()),
        "val_end": str(val[time_col].max()),
        "test_start": str(test[time_col].min()),
        "test_end": str(test[time_col].max()),
    }
    return train, val, test, metadata


def save_split_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Salva datasets de split temporal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "features_train.parquet",
        "val": output_dir / "features_val.parquet",
        "test": output_dir / "features_test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)
    return {key: str(value) for key, value in paths.items()}


def compute_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcula metricas binarias robustas para dados possivelmente desbalanceados."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    y_pred = (y_score_arr >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_true_arr, y_score_arr)),
    }

    if len(np.unique(y_true_arr)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


# Acima desta fracao de positivos preditos, o threshold deixou de discriminar:
# a lista vira "inspecionar tudo" e nao apoia decisao nenhuma.
DEGENERATE_ALERT_RATE = 0.99


@dataclass(frozen=True)
class ThresholdChoice:
    """Threshold calibrado e o diagnostico de como ele foi obtido."""

    threshold: float
    target_recall: float
    achieved_recall: float
    achieved_precision: float
    alert_rate: float
    target_met: bool

    @property
    def degenerate(self) -> bool:
        """Verdadeiro quando o threshold marca praticamente tudo como positivo.

        Este e o modo de falha que realmente ocorreu: o baseline heuristico foi
        publicado com threshold `0.0` e recall `1.0`. O alvo de recall estava
        formalmente atendido -- alertar sobre tudo garante recall perfeito --
        mas a decisao resultante e inutil. `target_met` sozinho nao denuncia
        esse caso, por isso a taxa de alerta e avaliada separadamente.
        """
        return self.alert_rate >= DEGENERATE_ALERT_RATE

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "threshold": self.threshold,
            "target_recall": self.target_recall,
            "achieved_recall": self.achieved_recall,
            "achieved_precision": self.achieved_precision,
            "alert_rate": self.alert_rate,
            "target_met": self.target_met,
            "degenerate": self.degenerate,
        }


def calibrate_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    min_recall: float = 0.80,
) -> ThresholdChoice:
    """Escolhe o threshold de maior precisao entre os que atingem o recall minimo.

    Quando nenhum candidato alcanca `min_recall`, a versao anterior devolvia
    silenciosamente o menor threshold -- ou seja, "alertar sobre tudo" -- sem
    aviso e sem marca nos relatorios. Isso aconteceu de fato com o baseline
    heuristico, que foi publicado com threshold `0.0` e recall `1.0`.

    Agora o fallback continua existindo (abortar no meio de uma busca de 90
    trials seria pior), mas escolhe o candidato de MAIOR recall e marca
    `target_met=False`, de modo que a condicao fique visivel a quem consome.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    if y_score_arr.size == 0:
        raise ValueError("Nao ha scores para calibrar o threshold.")

    candidates = np.unique(np.quantile(y_score_arr, np.linspace(0, 1, 101)))
    best: ThresholdChoice | None = None
    fallback: ThresholdChoice | None = None

    for threshold in candidates:
        metrics = compute_binary_metrics(y_true_arr, y_score_arr, float(threshold))
        recall = float(metrics["recall"])
        precision = float(metrics["precision"])
        choice = ThresholdChoice(
            threshold=float(threshold),
            target_recall=float(min_recall),
            achieved_recall=recall,
            achieved_precision=precision,
            alert_rate=float((y_score_arr >= threshold).mean()),
            target_met=recall >= min_recall,
        )
        if choice.target_met:
            if best is None or precision > best.achieved_precision:
                best = choice
        elif fallback is None or recall > fallback.achieved_recall:
            fallback = choice

    if best is not None:
        return best
    if fallback is not None:
        return fallback
    raise ValueError("Nenhum threshold candidato pode ser avaliado.")


def choose_threshold_for_recall(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    min_recall: float = 0.80,
) -> float:
    """Atalho que devolve apenas o valor do threshold calibrado."""
    return calibrate_threshold(y_true, y_score, min_recall=min_recall).threshold
