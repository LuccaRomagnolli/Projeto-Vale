"""Tuning e backtesting temporal do HistGradientBoosting regularizado."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.train_model import (
    LEAKAGE_COLUMNS,
    load_splits,
    predict_scores,
    prepare_model_matrix,
    select_feature_columns,
)
from src.models.validation import (
    TARGET_COL,
    TIME_COL,
    choose_threshold_for_recall,
    compute_binary_metrics,
)
from src.utils.config import FEATURES_DATASET_PATH, MODELS_DIR, REPORTS_DIR, SPLIT_DIR
from src.utils.metadata import build_execution_metadata, to_repo_relative_path

TUNED_MODEL_PATH = MODELS_DIR / "hist_gbdt_tuned.joblib"
TUNING_REPORT_JSON = REPORTS_DIR / "hist_gbdt_tuning_report.json"
TUNING_REPORT_CSV = REPORTS_DIR / "hist_gbdt_tuning_report.csv"
THRESHOLD_CURVE_CSV = REPORTS_DIR / "hist_gbdt_threshold_curve.csv"
BACKTEST_REPORT_CSV = REPORTS_DIR / "hist_gbdt_backtest_report.csv"


def build_param_grid() -> list[dict[str, Any]]:
    """Grade pequena e controlada para tuning recorrente."""
    return [
        {
            "learning_rate": 0.035,
            "max_iter": 450,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 80,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        {
            "learning_rate": 0.04,
            "max_iter": 350,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 80,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
        },
        {
            "learning_rate": 0.05,
            "max_iter": 320,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 120,
            "l2_regularization": 2.0,
            "class_weight": "balanced",
        },
        {
            "learning_rate": 0.04,
            "max_iter": 420,
            "max_leaf_nodes": 21,
            "min_samples_leaf": 120,
            "l2_regularization": 2.0,
            "class_weight": "balanced",
        },
        {
            "learning_rate": 0.03,
            "max_iter": 520,
            "max_leaf_nodes": 21,
            "min_samples_leaf": 160,
            "l2_regularization": 3.0,
            "class_weight": "balanced",
        },
        {
            "learning_rate": 0.06,
            "max_iter": 260,
            "max_leaf_nodes": 11,
            "min_samples_leaf": 100,
            "l2_regularization": 2.5,
            "class_weight": "balanced",
        },
    ]


def build_model(params: dict[str, Any], random_state: int = 42) -> HistGradientBoostingClassifier:
    """Instancia HistGradientBoostingClassifier com parametros auditaveis."""
    return HistGradientBoostingClassifier(
        **params,
        random_state=random_state,
    )


def flatten_metrics(
    candidate_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    fit_seconds: float,
) -> dict[str, Any]:
    """Transforma metricas aninhadas em linha tabular."""
    row: dict[str, Any] = {
        "candidate": candidate_name,
        "fit_seconds": round(fit_seconds, 3),
        "threshold": metrics["threshold"],
        **params,
    }
    for split_name in ("train", "val", "test"):
        for metric_name, value in metrics[split_name].items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def evaluate_fitted_model(
    model: HistGradientBoostingClassifier,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Avalia modelo com threshold escolhido somente na validacao."""
    matrices = {
        "train": prepare_model_matrix(train, feature_columns),
        "val": prepare_model_matrix(val, feature_columns),
        "test": prepare_model_matrix(test, feature_columns),
    }
    scores = {split: predict_scores(model, matrix) for split, matrix in matrices.items()}
    threshold = choose_threshold_for_recall(val[TARGET_COL], scores["val"], min_recall=min_recall)
    metrics = {
        "threshold": float(threshold),
        "train": compute_binary_metrics(train[TARGET_COL], scores["train"], threshold),
        "val": compute_binary_metrics(val[TARGET_COL], scores["val"], threshold),
        "test": compute_binary_metrics(test[TARGET_COL], scores["test"], threshold),
    }
    return metrics, scores


def select_best_candidate(summary: pd.DataFrame) -> pd.Series:
    """Escolhe candidato por validacao, sem olhar teste como criterio."""
    ordered = summary.sort_values(
        ["val_auc_pr", "val_recall", "val_precision"],
        ascending=[False, False, False],
    )
    return ordered.iloc[0]


def build_threshold_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_points: int = 101,
) -> pd.DataFrame:
    """Gera curva operacional de thresholds para decisao de manutencao."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    thresholds = np.unique(np.quantile(y_score_arr, np.linspace(0, 1, n_points)))
    rows = []
    for threshold in thresholds:
        y_pred = (y_score_arr >= threshold).astype(int)
        false_positive = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        false_negative = int(((y_pred == 0) & (y_true_arr == 1)).sum())
        metrics = compute_binary_metrics(y_true_arr, y_score_arr, float(threshold))
        rows.append(
            {
                **metrics,
                "predicted_positive_rate": float(y_pred.mean()),
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    return pd.DataFrame(rows)


def make_backtest_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    train_start_frac: float = 0.50,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Cria folds temporais expansivos para medir estabilidade."""
    ordered = df.copy()
    ordered[TIME_COL] = pd.to_datetime(ordered[TIME_COL], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    if len(ordered) < 30:
        raise ValueError("Dados insuficientes para backtesting temporal.")

    n_rows = len(ordered)
    val_size = max(int(n_rows * val_frac), 1)
    test_size = max(int(n_rows * test_frac), 1)
    available = 1.0 - train_start_frac - val_frac - test_frac
    step = available / max(n_folds - 1, 1)

    folds = []
    for fold_idx in range(n_folds):
        train_frac = train_start_frac + (fold_idx * step)
        train_end = int(n_rows * train_frac)
        val_end = min(train_end + val_size, n_rows - test_size)
        test_end = min(val_end + test_size, n_rows)

        train = ordered.iloc[:train_end].copy()
        val = ordered.iloc[train_end:val_end].copy()
        test = ordered.iloc[val_end:test_end].copy()
        if len(train) and len(val) and len(test):
            folds.append((train, val, test))

    if not folds:
        raise ValueError("Nao foi possivel criar folds temporais validos.")
    return folds


def run_backtest(
    features_df: pd.DataFrame,
    params: dict[str, Any],
    feature_columns: list[str],
    min_recall: float,
) -> pd.DataFrame:
    """Executa backtesting temporal usando os parametros vencedores."""
    rows = []
    for fold_idx, (train, val, test) in enumerate(make_backtest_folds(features_df), start=1):
        model = build_model(params)
        model.fit(prepare_model_matrix(train, feature_columns), train[TARGET_COL].astype(int))
        metrics, _ = evaluate_fitted_model(model, train, val, test, feature_columns, min_recall)
        row: dict[str, Any] = {
            "fold": fold_idx,
            "train_start": str(train[TIME_COL].min()),
            "train_end": str(train[TIME_COL].max()),
            "val_start": str(val[TIME_COL].min()),
            "val_end": str(val[TIME_COL].max()),
            "test_start": str(test[TIME_COL].min()),
            "test_end": str(test[TIME_COL].max()),
            "threshold": metrics["threshold"],
        }
        for split_name in ("val", "test"):
            for metric_name, value in metrics[split_name].items():
                row[f"{split_name}_{metric_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def save_tuning_outputs(
    model: HistGradientBoostingClassifier,
    feature_columns: list[str],
    best_row: pd.Series,
    summary: pd.DataFrame,
    threshold_curve: pd.DataFrame,
    backtest: pd.DataFrame,
    min_recall: float,
) -> dict[str, str]:
    """Persiste artefato, tuning, curva de threshold e backtesting."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sorted_summary = summary.sort_values(
        ["val_auc_pr", "val_recall", "val_precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    sorted_summary.to_csv(TUNING_REPORT_CSV, index=False)
    threshold_curve.to_csv(THRESHOLD_CURVE_CSV, index=False)
    backtest.to_csv(BACKTEST_REPORT_CSV, index=False)

    joblib.dump(
        {
            "model": model,
            "model_name": "hist_gbdt_tuned",
            "feature_columns": feature_columns,
            "threshold": float(best_row["threshold"]),
            "params": {
                key: best_row[key]
                for key in [
                    "learning_rate",
                    "max_iter",
                    "max_leaf_nodes",
                    "min_samples_leaf",
                    "l2_regularization",
                    "class_weight",
                ]
            },
            "selection_rule": "maior val_auc_pr; desempate por val_recall e val_precision",
        },
        TUNED_MODEL_PATH,
    )

    report = {
        "model_name": "hist_gbdt_tuned",
        "selection_rule": "maior val_auc_pr; desempate por val_recall e val_precision",
        "min_recall_calibration": min_recall,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "best_candidate": best_row.to_dict(),
        "candidates": sorted_summary.to_dict(orient="records"),
        "backtest_summary": {
            "folds": int(len(backtest)),
            "mean_test_recall": float(backtest["test_recall"].mean()),
            "mean_test_precision": float(backtest["test_precision"].mean()),
            "mean_test_auc_pr": float(backtest["test_auc_pr"].mean()),
            "std_test_recall": float(backtest["test_recall"].std(ddof=0)),
            "std_test_precision": float(backtest["test_precision"].std(ddof=0)),
        },
        "artifact_path": to_repo_relative_path(TUNED_MODEL_PATH),
        "csv_path": to_repo_relative_path(TUNING_REPORT_CSV),
        "threshold_curve_path": to_repo_relative_path(THRESHOLD_CURVE_CSV),
        "backtest_path": to_repo_relative_path(BACKTEST_REPORT_CSV),
    }
    report["metadata"] = build_execution_metadata(
        component="tune_hist_gbdt",
        feature_count=len(feature_columns),
        seed=42,
        config_payload={
            "selection_rule": report["selection_rule"],
            "min_recall_calibration": min_recall,
            "grid_candidates": int(len(summary)),
        },
        period_start=str(backtest["train_start"].min()),
        period_end=str(backtest["test_end"].max()),
    )
    TUNING_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))

    return {
        "artifact_path": str(TUNED_MODEL_PATH),
        "json_path": str(TUNING_REPORT_JSON),
        "csv_path": str(TUNING_REPORT_CSV),
        "threshold_curve_path": str(THRESHOLD_CURVE_CSV),
        "backtest_path": str(BACKTEST_REPORT_CSV),
    }


def run_tuning_pipeline(
    split_dir: Path = SPLIT_DIR,
    features_path: Path = FEATURES_DATASET_PATH,
    min_recall: float = 0.80,
) -> dict[str, Any]:
    """Executa tuning, curva de threshold e backtesting para HistGBDT."""
    train, val, test = load_splits(split_dir)
    feature_columns = select_feature_columns(train)
    x_train = prepare_model_matrix(train, feature_columns)
    y_train = train[TARGET_COL].astype(int)

    fitted_models: dict[str, HistGradientBoostingClassifier] = {}
    validation_scores: dict[str, np.ndarray] = {}
    rows = []
    for candidate_idx, params in enumerate(build_param_grid(), start=1):
        name = f"hist_gbdt_tuned_{candidate_idx:02d}"
        model = build_model(params)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        metrics, scores = evaluate_fitted_model(
            model, train, val, test, feature_columns, min_recall
        )
        fitted_models[name] = model
        validation_scores[name] = scores["val"]
        rows.append(flatten_metrics(name, params, metrics, fit_seconds))

    summary = pd.DataFrame(rows)
    best_row = select_best_candidate(summary)
    best_name = str(best_row["candidate"])
    best_model = fitted_models[best_name]
    threshold_curve = build_threshold_curve(val[TARGET_COL], validation_scores[best_name])

    features_df = (
        pd.read_parquet(features_path)
        if features_path.exists()
        else pd.concat([train, val, test], ignore_index=True)
    )
    best_params = {
        key: best_row[key]
        for key in [
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "class_weight",
        ]
    }
    backtest = run_backtest(features_df, best_params, feature_columns, min_recall)
    output_paths = save_tuning_outputs(
        model=best_model,
        feature_columns=feature_columns,
        best_row=best_row,
        summary=summary,
        threshold_curve=threshold_curve,
        backtest=backtest,
        min_recall=min_recall,
    )

    return {
        "best_candidate": best_row.to_dict(),
        "candidates_tested": int(len(summary)),
        "feature_count": len(feature_columns),
        "backtest_folds": int(len(backtest)),
        **output_paths,
    }


def main() -> None:
    result = run_tuning_pipeline()
    best = result["best_candidate"]
    print(f"[OK] Candidatos testados: {result['candidates_tested']}")
    print(f"[OK] Features usadas: {result['feature_count']}")
    print(f"[OK] Melhor candidato: {best['candidate']}")
    print(f"[OK] AUC-PR validacao: {best['val_auc_pr']:.6f}")
    print(f"[OK] Recall teste: {best['test_recall']:.6f}")
    print(f"[OK] AUC-PR teste: {best['test_auc_pr']:.6f}")
    print(f"[OK] Backtesting folds: {result['backtest_folds']}")
    print(f"[OK] Artefato: {result['artifact_path']}")
    print(f"[OK] Relatorio: {result['json_path']}")


if __name__ == "__main__":
    main()
