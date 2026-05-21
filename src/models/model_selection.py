"""Selecao robusta de modelos supervisionados sem modelo principal a priori."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.operational_scorecard import (
    DEFAULT_TOP_K_VALUES,
    PRIMARY_TOP_K,
    attach_operational_metrics,
    build_scored_frame,
    compute_split_topk_metrics,
)
from src.models.train_model import (
    LEAKAGE_COLUMNS,
    extract_feature_importance,
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
from src.utils.config import (
    FEATURES_DATASET_PATH,
    MODELS_DIR,
    REPORTS_MODEL_SELECTION_DIR,
    SPLIT_DIR,
)
from src.utils.metadata import build_execution_metadata, to_repo_relative_path

OFFICIAL_CANDIDATE_NAMES = (
    "lightgbm_optuna",
    "xgboost_optuna",
    "hist_gbdt_optuna",
)
BASELINE_MODEL_NAME = "logistic_regression_baseline"
DEFAULT_N_TRIALS = 30
DEFAULT_BACKTEST_FOLDS = 3
DEFAULT_MIN_RECALL = 0.80

SELECTED_MODEL_PATH = MODELS_DIR / "model_selected.joblib"
SELECTION_REPORT_JSON = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.json"
SELECTION_REPORT_CSV = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.csv"
SELECTION_TRIALS_CSV = REPORTS_MODEL_SELECTION_DIR / "model_selection_trials.csv"
SELECTION_SCORES_PATH = REPORTS_MODEL_SELECTION_DIR / "model_selection_scores.parquet"
SELECTION_BACKTEST_CSV = REPORTS_MODEL_SELECTION_DIR / "model_selection_backtest_report.csv"
SELECTED_THRESHOLD_CURVE_CSV = REPORTS_MODEL_SELECTION_DIR / "model_selected_threshold_curve.csv"
SELECTED_FEATURE_IMPORTANCE_CSV = (
    REPORTS_MODEL_SELECTION_DIR / "model_selected_feature_importance.csv"
)


def scale_pos_weight(y: pd.Series) -> float:
    """Calcula peso negativo/positivo para estimadores que usam scale_pos_weight."""
    positives = int(y.sum())
    negatives = len(y) - positives
    return float(negatives / max(positives, 1))


def build_baseline_model(random_state: int = 42) -> Pipeline:
    """Cria baseline linear diagnostico, fora da disputa oficial."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def suggest_candidate_params(
    trial: Any,
    candidate_name: str,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Define espacos de busca auditaveis por familia candidata."""
    if candidate_name == "lightgbm_optuna":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 180),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 8.0, log=True),
            "scale_pos_weight": scale_pos_weight(y_train),
        }
    if candidate_name == "xgboost_optuna":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 8.0, log=True),
            "scale_pos_weight": scale_pos_weight(y_train),
        }
    if candidate_name == "hist_gbdt_optuna":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
            "max_iter": trial.suggest_int("max_iter", 220, 700),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 11, 41),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 40, 180),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.05, 5.0, log=True),
            "class_weight": "balanced",
        }
    raise ValueError(f"Candidato desconhecido: {candidate_name}")


def build_candidate_model(
    candidate_name: str,
    params: dict[str, Any],
    random_state: int = 42,
) -> Any:
    """Instancia uma familia candidata oficial a partir de parametros fechados."""
    if candidate_name == "lightgbm_optuna":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if candidate_name == "xgboost_optuna":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    if candidate_name == "hist_gbdt_optuna":
        return HistGradientBoostingClassifier(random_state=random_state, **params)
    raise ValueError(f"Candidato desconhecido: {candidate_name}")


def _fit_candidate(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, float]:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - started


def evaluate_fitted_model(
    model: Any,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Avalia modelo com threshold calibrado apenas na validacao."""
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
    scored = pd.concat(
        [
            build_scored_frame(train, scores["train"], threshold, "train"),
            build_scored_frame(val, scores["val"], threshold, "val"),
            build_scored_frame(test, scores["test"], threshold, "test"),
        ],
        ignore_index=True,
    )
    return attach_operational_metrics(metrics, scored), scores


def flatten_metrics(
    model_name: str,
    role: str,
    estimator: Any,
    metrics: dict[str, Any],
    fit_seconds: float,
    eligible_for_selection: bool,
) -> dict[str, Any]:
    """Transforma metricas aninhadas em linha tabular."""
    row: dict[str, Any] = {
        "model_name": model_name,
        "role": role,
        "estimator": type(estimator).__name__,
        "fit_seconds": round(fit_seconds, 3),
        "threshold": metrics["threshold"],
        "eligible_for_selection": bool(eligible_for_selection),
    }
    for split_name in ("train", "val", "test"):
        for metric_name, value in metrics[split_name].items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def selection_columns() -> list[str]:
    """Colunas oficiais para ranking operacional."""
    return [
        f"val_top{PRIMARY_TOP_K}_recall_at_k",
        f"val_top{PRIMARY_TOP_K}_precision_at_k",
        f"val_top{PRIMARY_TOP_K}_lift_vs_random",
        "val_auc_pr",
    ]


def selection_score(row: dict[str, Any] | pd.Series) -> float:
    """Score escalar que preserva a prioridade lexicografica das metricas oficiais."""
    return (
        float(row.get(f"val_top{PRIMARY_TOP_K}_recall_at_k", 0.0)) * 1_000_000
        + float(row.get(f"val_top{PRIMARY_TOP_K}_precision_at_k", 0.0)) * 1_000
        + float(row.get(f"val_top{PRIMARY_TOP_K}_lift_vs_random", 0.0))
        + float(row.get("val_auc_pr", 0.0)) * 0.001
    )


def select_model(summary: pd.DataFrame) -> str:
    """Seleciona modelo apenas entre candidatos oficiais elegiveis."""
    eligible = summary.loc[summary["eligible_for_selection"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("Nenhum candidato elegivel para selecao.")
    cols = selection_columns()
    if set(cols).issubset(eligible.columns):
        ordered = eligible.sort_values(cols, ascending=[False] * len(cols))
    else:
        ordered = eligible.sort_values(["val_auc_pr", "val_recall"], ascending=[False, False])
    return str(ordered.iloc[0]["model_name"])


def _trial_objective(
    trial: Any,
    candidate_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float,
    random_state: int,
) -> float:
    y_train = train[TARGET_COL].astype(int)
    params = suggest_candidate_params(trial, candidate_name, y_train)
    model = build_candidate_model(candidate_name, params, random_state=random_state)
    x_train = prepare_model_matrix(train, feature_columns)
    fitted, fit_seconds = _fit_candidate(model, x_train, y_train)
    metrics, _ = evaluate_fitted_model(
        fitted,
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        min_recall=min_recall,
    )
    row = flatten_metrics(candidate_name, "official_candidate", fitted, metrics, fit_seconds, True)
    score = selection_score(row)
    for key, value in row.items():
        if isinstance(value, str | int | float | bool):
            trial.set_user_attr(key, value)
    trial.set_user_attr("params_json", json.dumps(params, ensure_ascii=False, sort_keys=True))
    trial.set_user_attr("selection_score", score)
    return score


def run_candidate_study(
    candidate_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    n_trials: int,
    min_recall: float,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, Any, dict[str, np.ndarray], float]:
    """Executa tuning Optuna de uma familia e reavalia seu melhor trial."""
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna e obrigatorio para `make model-selection`. "
            "Instale as dependencias com `make install`."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _trial_objective(
            trial,
            candidate_name,
            train,
            val,
            test,
            feature_columns,
            min_recall,
            random_state,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    trial_rows = []
    for trial in study.trials:
        row = {
            "model_name": candidate_name,
            "trial_number": trial.number,
            "state": trial.state.name,
            "selection_score": trial.user_attrs.get("selection_score", np.nan),
            "params_json": trial.user_attrs.get("params_json", "{}"),
        }
        for key, value in trial.user_attrs.items():
            if key not in row and key != "params_json":
                row[key] = value
        trial_rows.append(row)
    trials_df = pd.DataFrame(trial_rows)

    best_params = dict(study.best_params)
    if candidate_name in {"lightgbm_optuna", "xgboost_optuna"}:
        best_params["scale_pos_weight"] = scale_pos_weight(train[TARGET_COL].astype(int))
    if candidate_name == "hist_gbdt_optuna":
        best_params["class_weight"] = "balanced"
    model = build_candidate_model(candidate_name, best_params, random_state=random_state)
    fitted, fit_seconds = _fit_candidate(
        model,
        prepare_model_matrix(train, feature_columns),
        train[TARGET_COL].astype(int),
    )
    metrics, scores = evaluate_fitted_model(
        fitted,
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        min_recall=min_recall,
    )
    summary_row = flatten_metrics(
        candidate_name,
        "official_candidate",
        fitted,
        metrics,
        fit_seconds,
        eligible_for_selection=True,
    )
    summary_row["best_trial_number"] = int(study.best_trial.number)
    summary_row["best_params"] = json.dumps(best_params, ensure_ascii=False, sort_keys=True)
    return summary_row, trials_df, fitted, scores, fit_seconds


def evaluate_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float,
    random_state: int = 42,
) -> tuple[dict[str, Any], Any, dict[str, np.ndarray]]:
    """Treina baseline diagnostico fora da disputa oficial."""
    model = build_baseline_model(random_state=random_state)
    fitted, fit_seconds = _fit_candidate(
        model,
        prepare_model_matrix(train, feature_columns),
        train[TARGET_COL].astype(int),
    )
    metrics, scores = evaluate_fitted_model(
        fitted,
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        min_recall=min_recall,
    )
    row = flatten_metrics(
        BASELINE_MODEL_NAME,
        "diagnostic_baseline",
        fitted,
        metrics,
        fit_seconds,
        eligible_for_selection=False,
    )
    row["best_trial_number"] = np.nan
    row["best_params"] = "{}"
    return row, fitted, scores


def build_threshold_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_points: int = 101,
) -> pd.DataFrame:
    """Gera curva tecnica de threshold para o modelo selecionado."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    thresholds = np.unique(np.quantile(y_score_arr, np.linspace(0, 1, n_points)))
    rows = []
    for threshold in thresholds:
        y_pred = (y_score_arr >= threshold).astype(int)
        false_positive = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        false_negative = int(((y_pred == 0) & (y_true_arr == 1)).sum())
        rows.append(
            {
                **compute_binary_metrics(y_true_arr, y_score_arr, float(threshold)),
                "predicted_positive_rate": float(y_pred.mean()),
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    return pd.DataFrame(rows)


def make_backtest_folds(
    df: pd.DataFrame,
    n_folds: int = DEFAULT_BACKTEST_FOLDS,
    train_start_frac: float = 0.50,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Cria folds temporais expansivos para medir estabilidade do selecionado."""
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
    selected_model_name: str,
    selected_params: dict[str, Any],
    feature_columns: list[str],
    min_recall: float,
    n_folds: int = DEFAULT_BACKTEST_FOLDS,
) -> pd.DataFrame:
    """Executa backtesting temporal da familia selecionada."""
    rows = []
    for fold_idx, (train, val, test) in enumerate(
        make_backtest_folds(features_df, n_folds=n_folds),
        start=1,
    ):
        model = build_candidate_model(selected_model_name, selected_params)
        fitted, fit_seconds = _fit_candidate(
            model,
            prepare_model_matrix(train, feature_columns),
            train[TARGET_COL].astype(int),
        )
        metrics, _ = evaluate_fitted_model(fitted, train, val, test, feature_columns, min_recall)
        row: dict[str, Any] = {
            "fold": fold_idx,
            "model_name": selected_model_name,
            "fit_seconds": round(fit_seconds, 3),
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


def _save_scored_outputs(
    scores_by_model: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for model_name, split_scores in scores_by_model.items():
        threshold = float(summary.loc[summary["model_name"] == model_name, "threshold"].iloc[0])
        for split_name, split_df in {"train": train, "val": val, "test": test}.items():
            frame = build_scored_frame(split_df, split_scores[split_name], threshold, split_name)
            frame.insert(4, "model_name", model_name)
            frames.append(frame)
    scored = pd.concat(frames, ignore_index=True)
    scored.to_parquet(SELECTION_SCORES_PATH, index=False)
    return scored


def save_model_selection_outputs(
    trained_models: dict[str, Any],
    scores_by_model: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
    trials: pd.DataFrame,
    backtest: pd.DataFrame,
    threshold_curve: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_columns: list[str],
    selected_model_name: str,
    selected_params: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    min_recall: float,
    n_trials: int,
) -> dict[str, str]:
    """Persiste artefato neutro, reports e evidencias da selecao robusta."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

    ordered_summary = summary.sort_values(
        ["eligible_for_selection", *selection_columns()],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ordered_summary.to_csv(SELECTION_REPORT_CSV, index=False)
    trials.to_csv(SELECTION_TRIALS_CSV, index=False)
    backtest.to_csv(SELECTION_BACKTEST_CSV, index=False)
    threshold_curve.to_csv(SELECTED_THRESHOLD_CURVE_CSV, index=False)
    feature_importance.to_csv(SELECTED_FEATURE_IMPORTANCE_CSV, index=False)
    scored = _save_scored_outputs(scores_by_model, ordered_summary, train, val, test)

    selected_row = (
        ordered_summary.loc[ordered_summary["model_name"] == selected_model_name].iloc[0].to_dict()
    )
    selected_threshold = float(selected_row["threshold"])
    selection_rule = (
        f"maior val_top{PRIMARY_TOP_K}_recall_at_k; desempate por "
        f"val_top{PRIMARY_TOP_K}_precision_at_k, "
        f"val_top{PRIMARY_TOP_K}_lift_vs_random e val_auc_pr"
    )
    artifact = {
        "model": trained_models[selected_model_name],
        "model_name": selected_model_name,
        "feature_columns": feature_columns,
        "threshold": selected_threshold,
        "params": selected_params,
        "selection_rule": selection_rule,
        "primary_operational_top_k": PRIMARY_TOP_K,
        "candidate_pool": list(OFFICIAL_CANDIDATE_NAMES),
        "diagnostic_baseline": BASELINE_MODEL_NAME,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
    }
    joblib.dump(artifact, SELECTED_MODEL_PATH)

    operational_topk = []
    for model_name, group in scored.groupby("model_name", sort=False):
        topk = compute_split_topk_metrics(group, top_k_values=DEFAULT_TOP_K_VALUES)
        topk.insert(0, "model_name", model_name)
        operational_topk.extend(topk.to_dict(orient="records"))

    metadata = build_execution_metadata(
        component="model_selection",
        feature_count=len(feature_columns),
        seed=42,
        config_payload={
            "official_candidates": list(OFFICIAL_CANDIDATE_NAMES),
            "diagnostic_baseline": BASELINE_MODEL_NAME,
            "n_trials_per_candidate": n_trials,
            "min_recall_calibration": min_recall,
            "primary_operational_top_k": PRIMARY_TOP_K,
            "selection_rule": selection_rule,
        },
        period_start=str(train["Fim"].min()),
        period_end=str(test["Fim"].max()),
    )
    report = {
        "selection_name": "robust_model_selection",
        "selected_model": selected_row,
        "selection_rule": selection_rule,
        "official_candidates": list(OFFICIAL_CANDIDATE_NAMES),
        "diagnostic_baseline": BASELINE_MODEL_NAME,
        "n_trials_per_candidate": n_trials,
        "min_recall_calibration": min_recall,
        "primary_operational_top_k": PRIMARY_TOP_K,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "models": ordered_summary.to_dict(orient="records"),
        "operational_topk": operational_topk,
        "backtest_summary": {
            "folds": int(len(backtest)),
            "mean_test_recall": float(backtest["test_recall"].mean()),
            "mean_test_precision": float(backtest["test_precision"].mean()),
            f"mean_test_top{PRIMARY_TOP_K}_recall_at_k": float(
                backtest[f"test_top{PRIMARY_TOP_K}_recall_at_k"].mean()
            ),
            f"mean_test_top{PRIMARY_TOP_K}_precision_at_k": float(
                backtest[f"test_top{PRIMARY_TOP_K}_precision_at_k"].mean()
            ),
            f"std_test_top{PRIMARY_TOP_K}_recall_at_k": float(
                backtest[f"test_top{PRIMARY_TOP_K}_recall_at_k"].std(ddof=0)
            ),
            f"std_test_top{PRIMARY_TOP_K}_precision_at_k": float(
                backtest[f"test_top{PRIMARY_TOP_K}_precision_at_k"].std(ddof=0)
            ),
        },
        "outputs": {
            "artifact_path": to_repo_relative_path(SELECTED_MODEL_PATH),
            "summary_csv": to_repo_relative_path(SELECTION_REPORT_CSV),
            "trials_csv": to_repo_relative_path(SELECTION_TRIALS_CSV),
            "scores_path": to_repo_relative_path(SELECTION_SCORES_PATH),
            "backtest_csv": to_repo_relative_path(SELECTION_BACKTEST_CSV),
            "threshold_curve_csv": to_repo_relative_path(SELECTED_THRESHOLD_CURVE_CSV),
            "feature_importance_csv": to_repo_relative_path(SELECTED_FEATURE_IMPORTANCE_CSV),
        },
        "metadata": metadata,
    }
    SELECTION_REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True)
    )
    return {
        "artifact_path": str(SELECTED_MODEL_PATH),
        "json_path": str(SELECTION_REPORT_JSON),
        "csv_path": str(SELECTION_REPORT_CSV),
        "trials_path": str(SELECTION_TRIALS_CSV),
        "scores_path": str(SELECTION_SCORES_PATH),
        "backtest_path": str(SELECTION_BACKTEST_CSV),
        "threshold_curve_path": str(SELECTED_THRESHOLD_CURVE_CSV),
        "feature_importance_path": str(SELECTED_FEATURE_IMPORTANCE_CSV),
        "selected_model_name": selected_model_name,
    }


def run_model_selection_pipeline(
    split_dir: Path = SPLIT_DIR,
    features_path: Path = FEATURES_DATASET_PATH,
    n_trials: int = DEFAULT_N_TRIALS,
    n_backtest_folds: int = DEFAULT_BACKTEST_FOLDS,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> dict[str, Any]:
    """Executa selecao robusta e gera o artefato operacional neutro."""
    train, val, test = load_splits(split_dir)
    feature_columns = select_feature_columns(train)
    trained_models: dict[str, Any] = {}
    scores_by_model: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []

    baseline_row, baseline_model, baseline_scores = evaluate_baseline(
        train,
        val,
        test,
        feature_columns,
        min_recall=min_recall,
    )
    trained_models[BASELINE_MODEL_NAME] = baseline_model
    scores_by_model[BASELINE_MODEL_NAME] = baseline_scores
    summary_rows.append(baseline_row)

    for candidate_name in OFFICIAL_CANDIDATE_NAMES:
        row, trials_df, model, scores, _ = run_candidate_study(
            candidate_name,
            train,
            val,
            test,
            feature_columns,
            n_trials=n_trials,
            min_recall=min_recall,
        )
        trained_models[candidate_name] = model
        scores_by_model[candidate_name] = scores
        summary_rows.append(row)
        trial_frames.append(trials_df)

    summary = pd.DataFrame(summary_rows)
    selected_model_name = select_model(summary)
    selected_row = summary.loc[summary["model_name"] == selected_model_name].iloc[0]
    selected_params = json.loads(str(selected_row["best_params"]))
    threshold_curve = build_threshold_curve(
        val[TARGET_COL],
        scores_by_model[selected_model_name]["val"],
    )
    feature_importance = extract_feature_importance(
        trained_models[selected_model_name],
        feature_columns,
        val,
    )
    features_df = (
        pd.read_parquet(features_path)
        if features_path.exists()
        else pd.concat([train, val, test], ignore_index=True)
    )
    backtest = run_backtest(
        features_df,
        selected_model_name,
        selected_params,
        feature_columns,
        min_recall=min_recall,
        n_folds=n_backtest_folds,
    )
    trials = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
    output_paths = save_model_selection_outputs(
        trained_models=trained_models,
        scores_by_model=scores_by_model,
        summary=summary,
        trials=trials,
        backtest=backtest,
        threshold_curve=threshold_curve,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        selected_model_name=selected_model_name,
        selected_params=selected_params,
        train=train,
        val=val,
        test=test,
        min_recall=min_recall,
        n_trials=n_trials,
    )

    return {
        "models_evaluated": int(len(summary)),
        "official_candidates": list(OFFICIAL_CANDIDATE_NAMES),
        "diagnostic_baseline": BASELINE_MODEL_NAME,
        "trials_per_candidate": int(n_trials),
        "backtest_folds": int(len(backtest)),
        "feature_count": len(feature_columns),
        "selected_model": selected_row.to_dict(),
        **output_paths,
    }


def main() -> None:
    result = run_model_selection_pipeline()
    selected = result["selected_model"]
    print(f"[OK] Candidatos oficiais: {len(result['official_candidates'])}")
    print(f"[OK] Trials por candidato: {result['trials_per_candidate']}")
    print(f"[OK] Baseline diagnostico: {result['diagnostic_baseline']}")
    print(f"[OK] Features usadas: {result['feature_count']}")
    print(f"[OK] Modelo selecionado: {result['selected_model_name']}")
    val_precision_key = f"val_top{PRIMARY_TOP_K}_precision_at_k"
    val_recall_key = f"val_top{PRIMARY_TOP_K}_recall_at_k"
    val_lift_key = f"val_top{PRIMARY_TOP_K}_lift_vs_random"
    if {val_precision_key, val_recall_key, val_lift_key}.issubset(selected):
        print(
            f"[OK] Top{PRIMARY_TOP_K} Tag-dia validacao: "
            f"precision={selected[val_precision_key]:.6f}, "
            f"recall={selected[val_recall_key]:.6f}, "
            f"lift={selected[val_lift_key]:.6f}"
        )
    print(f"[OK] Backtesting folds: {result['backtest_folds']}")
    print(f"[OK] Artefato selecionado: {result['artifact_path']}")
    print(f"[OK] Relatorio JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
