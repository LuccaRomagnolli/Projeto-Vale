"""Utilitarios de matriz/scoring e treino supervisionado legado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.evaluation.operational_scorecard import (
    PRIMARY_TOP_K,
    attach_operational_metrics,
    build_scored_frame,
)
from src.models.validation import (
    TARGET_COL,
    choose_threshold_for_recall,
    compute_binary_metrics,
)
from src.utils.config import MODELS_DIR, REPORTS_LEGACY_MODEL_DIR, SPLIT_DIR
from src.utils.metadata import build_execution_metadata, to_repo_relative_path

MODEL_ARTIFACT_PATH = MODELS_DIR / "legacy_supervised_model.joblib"
MODEL_REPORT_PATH = REPORTS_LEGACY_MODEL_DIR / "legacy_supervised_model_report.json"
MODEL_SCORES_PATH = REPORTS_LEGACY_MODEL_DIR / "legacy_supervised_model_scores.parquet"
FEATURE_IMPORTANCE_PATH = REPORTS_LEGACY_MODEL_DIR / "model_feature_importance.csv"

LEAKAGE_COLUMNS = {
    "Id",
    "Inicio",
    "Fim",
    "Tag",
    "Classe",
    "next_critical_event_time",
    "tte_horas",
    TARGET_COL,
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Seleciona apenas features numericas/booleanas sem colunas de vazamento."""
    candidate_columns = [column for column in df.columns if column not in LEAKAGE_COLUMNS]
    feature_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("Nenhuma feature numerica disponivel para treino.")
    return feature_columns


def prepare_model_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepara matriz de modelagem com tipos estaveis."""
    matrix = df[feature_columns].copy()
    for column in matrix.columns:
        if pd.api.types.is_bool_dtype(matrix[column]):
            matrix[column] = matrix[column].astype("int8")
    return matrix.astype("float32")


def build_estimator(y_train: pd.Series, random_state: int = 42) -> tuple[Any, str]:
    """Cria LightGBM quando disponivel; caso contrario usa fallback sklearn."""
    scale_pos_weight = (len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)
    try:
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=50,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        return estimator, "lightgbm.LGBMClassifier"
    except ModuleNotFoundError:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
            class_weight="balanced",
        )
        return estimator, "sklearn.HistGradientBoostingClassifier"


def predict_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Retorna probabilidade da classe positiva."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1 / (1 + np.exp(-decision))


def load_splits(split_dir: Path = SPLIT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega splits temporais gerados na Etapa 6."""
    paths = {
        "train": split_dir / "features_train.parquet",
        "val": split_dir / "features_val.parquet",
        "test": split_dir / "features_test.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Splits temporais ausentes: {missing}. Execute `make train`.")
    return (
        pd.read_parquet(paths["train"]),
        pd.read_parquet(paths["val"]),
        pd.read_parquet(paths["test"]),
    )


def evaluate_model(
    model: Any,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    min_recall: float = 0.80,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Avalia modelo com threshold calibrado na validacao."""
    matrices = {
        "train": prepare_model_matrix(train, feature_columns),
        "val": prepare_model_matrix(val, feature_columns),
        "test": prepare_model_matrix(test, feature_columns),
    }
    scores = {split: predict_scores(model, x) for split, x in matrices.items()}
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


def extract_feature_importance(
    model: Any,
    feature_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extrai importancia quando o estimador disponibiliza esse atributo."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_columns, "importance": values})
            .sort_values("importance", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    sample = reference_df.sample(min(len(reference_df), 5000), random_state=42)
    x_sample = prepare_model_matrix(sample, feature_columns)
    y_sample = sample[TARGET_COL].astype(int)
    result = permutation_importance(
        model,
        x_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    values = result.importances_mean
    std = result.importances_std
    return (
        pd.DataFrame({"feature": feature_columns, "importance": values, "importance_std": std})
        .sort_values("importance", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def save_model_outputs(
    model: Any,
    model_library: str,
    feature_columns: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict[str, Any],
    scores: dict[str, np.ndarray],
) -> dict[str, str]:
    """Persiste artefato, scores, metricas e importancia."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_LEGACY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "model_library": model_library,
            "feature_columns": feature_columns,
            "threshold": metrics["threshold"],
            "leakage_columns": sorted(LEAKAGE_COLUMNS),
        },
        MODEL_ARTIFACT_PATH,
    )

    frames = []
    for split_name, split_df in {"train": train, "val": val, "test": test}.items():
        output = build_scored_frame(
            split_df,
            scores[split_name],
            metrics["threshold"],
            split_name,
        )
        output["model_score"] = output["score"]
        output["model_pred"] = output["prediction"]
        frames.append(output)
    pd.concat(frames, ignore_index=True).to_parquet(MODEL_SCORES_PATH, index=False)

    importance = extract_feature_importance(model, feature_columns, val)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    config_payload = {
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
        "min_recall_calibration": 0.80,
        "model_library": model_library,
        "scorecard_operacional": f"Top{PRIMARY_TOP_K} Tag-dia por split temporal",
    }
    period_start = str(train["Fim"].min())
    period_end = str(test["Fim"].max())
    metadata = build_execution_metadata(
        component="train_model",
        feature_count=len(feature_columns),
        seed=42,
        config_payload=config_payload,
        period_start=period_start,
        period_end=period_end,
    )

    report = {
        "model_name": "legacy_supervised_model",
        "model_library": model_library,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "scorecard_operacional": {
            "primary_top_k_tags_per_day": PRIMARY_TOP_K,
            "selection_metric": (
                f"val_top{PRIMARY_TOP_K}_recall_at_k, "
                f"val_top{PRIMARY_TOP_K}_precision_at_k e "
                f"val_top{PRIMARY_TOP_K}_lift_vs_random"
            ),
        },
        "metrics": metrics,
        "artifact_path": to_repo_relative_path(MODEL_ARTIFACT_PATH),
        "scores_path": to_repo_relative_path(MODEL_SCORES_PATH),
        "feature_importance_path": to_repo_relative_path(FEATURE_IMPORTANCE_PATH),
        "metadata": metadata,
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True))

    return {
        "artifact_path": str(MODEL_ARTIFACT_PATH),
        "report_path": str(MODEL_REPORT_PATH),
        "scores_path": str(MODEL_SCORES_PATH),
        "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
    }


def run_model_pipeline(split_dir: Path = SPLIT_DIR) -> dict[str, Any]:
    """Treina e avalia um modelo supervisionado legado."""
    train, val, test = load_splits(split_dir)
    feature_columns = select_feature_columns(train)
    x_train = prepare_model_matrix(train, feature_columns)
    y_train = train[TARGET_COL].astype(int)

    model, model_library = build_estimator(y_train)
    model.fit(x_train, y_train)

    metrics, scores = evaluate_model(model, train, val, test, feature_columns)
    output_paths = save_model_outputs(
        model=model,
        model_library=model_library,
        feature_columns=feature_columns,
        train=train,
        val=val,
        test=test,
        metrics=metrics,
        scores=scores,
    )

    return {
        "model_library": model_library,
        "feature_count": len(feature_columns),
        "metrics": metrics,
        **output_paths,
    }


def main() -> None:
    result = run_model_pipeline()
    print(f"[OK] Modelo supervisionado legado: {result['model_library']}")
    print(f"[OK] Features usadas: {result['feature_count']}")
    print(f"[OK] Artefato: {result['artifact_path']}")
    print(f"[OK] Relatorio: {result['report_path']}")
    print(f"[OK] Threshold validacao: {result['metrics']['threshold']:.6f}")
    print(f"[OK] Recall teste: {result['metrics']['test']['recall']:.6f}")
    print(f"[OK] AUC-PR teste: {result['metrics']['test']['auc_pr']:.6f}")
    test_metrics = result["metrics"]["test"]
    precision_key = f"top{PRIMARY_TOP_K}_precision_at_k"
    recall_key = f"top{PRIMARY_TOP_K}_recall_at_k"
    lift_key = f"top{PRIMARY_TOP_K}_lift_vs_random"
    if {precision_key, recall_key, lift_key}.issubset(test_metrics):
        print(
            f"[OK] Top{PRIMARY_TOP_K} Tag-dia teste: "
            f"precision={test_metrics[precision_key]:.6f}, "
            f"recall={test_metrics[recall_key]:.6f}, "
            f"lift={test_metrics[lift_key]:.6f}"
        )


if __name__ == "__main__":
    main()
