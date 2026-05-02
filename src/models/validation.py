"""Validacao temporal e metricas compartilhadas de modelagem."""

from __future__ import annotations

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


def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Divide dados por ordem temporal em treino, validacao e teste."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Frações inválidas para split temporal.")

    ordered = df.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))

    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    metadata = {
        "rows_total": int(n_rows),
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
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


def choose_threshold_for_recall(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    min_recall: float = 0.80,
) -> float:
    """Seleciona threshold de maior precisao entre candidatos que atingem recall minimo."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    candidates = np.unique(np.quantile(y_score_arr, np.linspace(0, 1, 101)))
    best_threshold = float(candidates.min()) if len(candidates) else 0.0
    best_precision = -1.0

    for threshold in candidates:
        metrics = compute_binary_metrics(y_true_arr, y_score_arr, float(threshold))
        if metrics["recall"] >= min_recall and metrics["precision"] > best_precision:
            best_precision = metrics["precision"]
            best_threshold = float(threshold)

    return best_threshold
