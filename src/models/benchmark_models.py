"""Benchmark temporal robusto para selecao tecnica de modelos."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.operational_scorecard import (
    PRIMARY_TOP_K,
    attach_operational_metrics,
    build_scored_frame,
    compute_split_topk_metrics,
)
from src.models.train_model import (
    LEAKAGE_COLUMNS,
    load_splits,
    predict_scores,
    prepare_model_matrix,
    select_feature_columns,
)
from src.models.validation import TARGET_COL, choose_threshold_for_recall, compute_binary_metrics
from src.utils.config import MODELS_DIR, REPORTS_DIR, SPLIT_DIR
from src.utils.metadata import build_execution_metadata, to_repo_relative_path

BENCHMARK_REPORT_JSON = REPORTS_DIR / "model_benchmark_report.json"
BENCHMARK_REPORT_CSV = REPORTS_DIR / "model_benchmark_report.csv"
BENCHMARK_SCORES_PATH = REPORTS_DIR / "model_benchmark_scores.parquet"
BENCHMARK_SELECTED_PATH = MODELS_DIR / "model_benchmark_selected.joblib"
BENCHMARK_ITERATION_REPORT_CSV = REPORTS_DIR / "model_benchmark_iteration_report.csv"
DEFAULT_RANDOM_STATES = (42, 52, 62)


def build_candidate_models() -> dict[str, Callable[[int], Any]]:
    """Define candidatos tabulares para benchmark robusto."""
    candidates: dict[str, Callable[[int], Any]] = {
        "logistic_regression_balanced": lambda random_state: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.6,
                        class_weight="balanced",
                        max_iter=2000,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gbdt_regularized": lambda random_state: HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=550,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=1.2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "extra_trees_balanced": lambda random_state: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=320,
                        max_depth=16,
                        min_samples_leaf=24,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": lambda random_state: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=280,
                        max_depth=14,
                        min_samples_leaf=32,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm_balanced"] = lambda random_state: LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=80,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ModuleNotFoundError:
        pass

    return candidates


def choose_top_candidates(candidates: dict[str, Callable[[int], Any]]) -> dict[str, Callable[[int], Any]]:
    """Seleciona os 4 candidatos tecnicos mais aderentes ao problema."""
    preferred_order = [
        "lightgbm_balanced",
        "hist_gbdt_regularized",
        "extra_trees_balanced",
        "logistic_regression_balanced",
        "random_forest_balanced",
    ]
    selected_names = [name for name in preferred_order if name in candidates][:4]
    return {name: candidates[name] for name in selected_names}


def _fit_candidate(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, float]:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - started


def _evaluate_candidate(
    model: Any,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
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
    metrics = attach_operational_metrics(metrics, scored)
    return metrics, scores


def _flatten_metrics(
    name: str,
    model: Any,
    metrics: dict[str, Any],
    fit_seconds: float,
    iteration: int,
    random_state: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": name,
        "estimator": type(model).__name__,
        "fit_seconds": round(fit_seconds, 3),
        "threshold": metrics["threshold"],
        "iteration": int(iteration),
        "random_state": int(random_state),
    }
    for split_name in ("train", "val", "test"):
        for metric_name, value in metrics[split_name].items():
            row[f"{split_name}_{metric_name}"] = value
    return row


def select_winner(summary: pd.DataFrame) -> str:
    """Seleciona modelo pelo scorecard operacional de validacao."""
    operational_cols = [
        f"val_top{PRIMARY_TOP_K}_recall_at_k",
        f"val_top{PRIMARY_TOP_K}_precision_at_k",
        f"val_top{PRIMARY_TOP_K}_lift_vs_random",
        "val_auc_pr",
    ]
    if set(operational_cols).issubset(summary.columns):
        ordered = summary.sort_values(operational_cols, ascending=[False] * len(operational_cols))
    else:
        ordered = summary.sort_values(
            ["val_auc_pr", "val_precision", "val_f1"],
            ascending=[False, False, False],
        )
    return str(ordered.iloc[0]["model_name"])


def _is_better_row(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Compara duas iteracoes pela mesma regra operacional de ordenacao."""
    ranking_keys = [
        f"val_top{PRIMARY_TOP_K}_recall_at_k",
        f"val_top{PRIMARY_TOP_K}_precision_at_k",
        f"val_top{PRIMARY_TOP_K}_lift_vs_random",
        "val_auc_pr",
    ]
    for key in ranking_keys:
        if candidate.get(key, float("-inf")) > reference.get(key, float("-inf")):
            return True
        if candidate.get(key, float("-inf")) < reference.get(key, float("-inf")):
            return False
    return candidate.get("test_recall", float("-inf")) > reference.get("test_recall", float("-inf"))


def save_benchmark_outputs(
    trained_models: dict[str, Any],
    all_scores: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
    iteration_summary: pd.DataFrame,
    feature_columns: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    min_recall: float,
) -> dict[str, str]:
    """Persiste comparativo agregado e artefato selecionado."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    winner_name = select_winner(summary)
    winner_row = summary.loc[summary["model_name"] == winner_name].iloc[0].to_dict()

    scores_frames = []
    for model_name, split_scores in all_scores.items():
        threshold = float(summary.loc[summary["model_name"] == model_name, "threshold"].iloc[0])
        for split_name, split_df in {"train": train, "val": val, "test": test}.items():
            frame = build_scored_frame(
                split_df,
                split_scores[split_name],
                threshold,
                split_name,
            )
            frame.insert(4, "model_name", model_name)
            scores_frames.append(frame)

    scored_output = pd.concat(scores_frames, ignore_index=True)
    scored_output.to_parquet(BENCHMARK_SCORES_PATH, index=False)

    topk_frames = []
    for model_name, group in scored_output.groupby("model_name", sort=False):
        topk = compute_split_topk_metrics(group)
        topk.insert(0, "model_name", model_name)
        topk_frames.append(topk)
    operational_topk = pd.concat(topk_frames, ignore_index=True) if topk_frames else pd.DataFrame()

    operational_cols = [
        f"val_top{PRIMARY_TOP_K}_recall_at_k",
        f"val_top{PRIMARY_TOP_K}_precision_at_k",
        f"val_top{PRIMARY_TOP_K}_lift_vs_random",
        "val_auc_pr",
    ]
    summary = summary.sort_values(operational_cols, ascending=[False] * len(operational_cols))
    summary = summary.reset_index(drop=True)
    summary.to_csv(BENCHMARK_REPORT_CSV, index=False)
    iteration_summary.to_csv(BENCHMARK_ITERATION_REPORT_CSV, index=False)

    config_payload = {
        "selection_rule": (
            f"maior val_top{PRIMARY_TOP_K}_recall_at_k; desempate por "
            f"val_top{PRIMARY_TOP_K}_precision_at_k, "
            f"val_top{PRIMARY_TOP_K}_lift_vs_random e val_auc_pr"
        ),
        "min_recall_calibration": min_recall,
        "candidate_models": sorted(list(trained_models.keys())),
        "selection_size": 4,
        "iterations_per_model": int(iteration_summary["iteration"].nunique()),
        "primary_operational_top_k": PRIMARY_TOP_K,
    }
    metadata = build_execution_metadata(
        component="benchmark_models",
        feature_count=len(feature_columns),
        seed=min(DEFAULT_RANDOM_STATES),
        config_payload=config_payload,
        period_start=str(train["Fim"].min()),
        period_end=str(test["Fim"].max()),
    )

    report = {
        "benchmark_name": "benchmark_modelos_supervisionados",
        "selection_rule": config_payload["selection_rule"],
        "min_recall_calibration": min_recall,
        "primary_operational_top_k": PRIMARY_TOP_K,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "selected_model": winner_row,
        "winner": winner_row,
        "models": summary.to_dict(orient="records"),
        "iterations": iteration_summary.to_dict(orient="records"),
        "operational_topk": operational_topk.to_dict(orient="records"),
        "scores_path": to_repo_relative_path(BENCHMARK_SCORES_PATH),
        "csv_path": to_repo_relative_path(BENCHMARK_REPORT_CSV),
        "iteration_csv_path": to_repo_relative_path(BENCHMARK_ITERATION_REPORT_CSV),
        "selected_artifact_path": to_repo_relative_path(BENCHMARK_SELECTED_PATH),
        "winner_artifact_path": to_repo_relative_path(BENCHMARK_SELECTED_PATH),
        "metadata": metadata,
    }
    BENCHMARK_REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True)
    )

    joblib.dump(
        {
            "model": trained_models[winner_name],
            "model_name": winner_name,
            "feature_columns": feature_columns,
            "threshold": float(winner_row["threshold"]),
            "selection_rule": report["selection_rule"],
            "primary_operational_top_k": PRIMARY_TOP_K,
            "iterations_per_model": int(iteration_summary["iteration"].nunique()),
        },
        BENCHMARK_SELECTED_PATH,
    )

    return {
        "json_path": str(BENCHMARK_REPORT_JSON),
        "csv_path": str(BENCHMARK_REPORT_CSV),
        "iteration_csv_path": str(BENCHMARK_ITERATION_REPORT_CSV),
        "scores_path": str(BENCHMARK_SCORES_PATH),
        "selected_artifact_path": str(BENCHMARK_SELECTED_PATH),
        "selected_name": winner_name,
    }


def run_benchmark_pipeline(
    split_dir: Path = SPLIT_DIR,
    min_recall: float = 0.80,
    random_states: tuple[int, ...] = DEFAULT_RANDOM_STATES,
) -> dict[str, Any]:
    """Treina 4 modelos com iteracoes e compara desempenho em validacao temporal."""
    train, val, test = load_splits(split_dir)
    feature_columns = select_feature_columns(train)
    x_train = prepare_model_matrix(train, feature_columns)
    y_train = train[TARGET_COL].astype(int)

    trained_models: dict[str, Any] = {}
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    selected_candidates = choose_top_candidates(build_candidate_models())
    for name, model_factory in selected_candidates.items():
        best_iteration_row: dict[str, Any] | None = None
        for iteration, random_state in enumerate(random_states, start=1):
            model = model_factory(random_state)
            fitted_model, fit_seconds = _fit_candidate(model, x_train, y_train)
            metrics, scores = _evaluate_candidate(
                fitted_model,
                train=train,
                val=val,
                test=test,
                feature_columns=feature_columns,
                min_recall=min_recall,
            )
            row = _flatten_metrics(
                name=name,
                model=fitted_model,
                metrics=metrics,
                fit_seconds=fit_seconds,
                iteration=iteration,
                random_state=random_state,
            )
            rows.append(row)

            if best_iteration_row is None:
                best_iteration_row = row
                trained_models[name] = fitted_model
                all_scores[name] = scores
                continue

            if _is_better_row(row, best_iteration_row):
                best_iteration_row = row
                trained_models[name] = fitted_model
                all_scores[name] = scores

    iteration_summary = pd.DataFrame(rows)
    summary = (
        iteration_summary.drop(columns=["iteration", "random_state", "fit_seconds"])
        .groupby(["model_name", "estimator"], as_index=False)
        .mean(numeric_only=True)
    )
    fit_time = (
        iteration_summary.groupby("model_name", as_index=False)["fit_seconds"]
        .mean()
        .rename(columns={"fit_seconds": "fit_seconds"})
    )
    summary = summary.merge(fit_time, on="model_name", how="left")
    output_paths = save_benchmark_outputs(
        trained_models=trained_models,
        all_scores=all_scores,
        summary=summary,
        iteration_summary=iteration_summary,
        feature_columns=feature_columns,
        train=train,
        val=val,
        test=test,
        min_recall=min_recall,
    )

    winner = summary.loc[summary["model_name"] == output_paths["selected_name"]].iloc[0].to_dict()
    return {
        "models_trained": len(summary),
        "models_selected_for_benchmark": len(selected_candidates),
        "iterations_per_model": len(random_states),
        "feature_count": len(feature_columns),
        "selected_model": winner,
        "summary": summary.to_dict(orient="records"),
        **output_paths,
    }


def main() -> None:
    result = run_benchmark_pipeline()
    selected = result["selected_model"]
    print(f"[OK] Modelos benchmarkados: {result['models_trained']}")
    print(f"[OK] Modelos selecionados para comparacao: {result['models_selected_for_benchmark']}")
    print(f"[OK] Iteracoes por modelo: {result['iterations_per_model']}")
    print(f"[OK] Features usadas: {result['feature_count']}")
    print(f"[OK] Modelo selecionado (validacao): {result['selected_name']}")
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
    print(f"[OK] AUC-PR validacao tecnica: {selected['val_auc_pr']:.6f}")
    print(f"[OK] Recall teste: {selected['test_recall']:.6f}")
    test_precision_key = f"test_top{PRIMARY_TOP_K}_precision_at_k"
    test_recall_key = f"test_top{PRIMARY_TOP_K}_recall_at_k"
    test_lift_key = f"test_top{PRIMARY_TOP_K}_lift_vs_random"
    if {test_precision_key, test_recall_key, test_lift_key}.issubset(selected):
        print(
            f"[OK] Top{PRIMARY_TOP_K} Tag-dia teste: "
            f"precision={selected[test_precision_key]:.6f}, "
            f"recall={selected[test_recall_key]:.6f}, "
            f"lift={selected[test_lift_key]:.6f}"
        )
    print(f"[OK] Relatorio JSON: {result['json_path']}")
    print(f"[OK] Relatorio CSV: {result['csv_path']}")
    print(f"[OK] Relatorio iteracoes: {result['iteration_csv_path']}")
    print(f"[OK] Artefato selecionado: {result['selected_artifact_path']}")


if __name__ == "__main__":
    main()
