"""Etapa 6: split temporal e baseline heuristico."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.validation import (
    TARGET_COL,
    choose_threshold_for_recall,
    compute_binary_metrics,
    save_split_datasets,
    temporal_train_val_test_split,
)
from src.utils.config import FEATURES_DATASET_PATH, MODELS_DIR, REPORTS_DIR, SPLIT_DIR

BASELINE_ARTIFACT_PATH = MODELS_DIR / "baseline_heuristico.joblib"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_report.json"
BASELINE_SCORES_PATH = REPORTS_DIR / "baseline_scores.parquet"
SPLIT_METADATA_PATH = SPLIT_DIR / "split_metadata.json"


def baseline_score(df: pd.DataFrame) -> pd.Series:
    """Usa frequencia historica de alertas por Tag nas ultimas 24h como score."""
    if "n_alertas_24h" not in df.columns:
        raise ValueError("Feature obrigatoria ausente: n_alertas_24h")
    return (df["n_alertas_24h"].fillna(0).astype(float) / 24.0).clip(0.0, 1.0)


def evaluate_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    min_recall: float = 0.80,
) -> dict[str, Any]:
    """Avalia baseline usando threshold calibrado na validacao temporal."""
    train_score = baseline_score(train)
    val_score = baseline_score(val)
    test_score = baseline_score(test)

    threshold = choose_threshold_for_recall(val[TARGET_COL], val_score, min_recall=min_recall)
    return {
        "threshold": threshold,
        "train": compute_binary_metrics(train[TARGET_COL], train_score, threshold),
        "val": compute_binary_metrics(val[TARGET_COL], val_score, threshold),
        "test": compute_binary_metrics(test[TARGET_COL], test_score, threshold),
    }


def save_baseline_outputs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    metrics: dict[str, Any],
    split_metadata: dict[str, Any],
) -> dict[str, str]:
    """Persiste scores, relatorio e artefato do baseline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for split_name, split_df in {"train": train, "val": val, "test": test}.items():
        scores = split_df[["Id", "Tag", "Fim", TARGET_COL]].copy()
        scores["split"] = split_name
        scores["baseline_score"] = baseline_score(split_df).to_numpy()
        scores["baseline_pred"] = (scores["baseline_score"] >= metrics["threshold"]).astype(int)
        frames.append(scores)

    pd.concat(frames, ignore_index=True).to_parquet(BASELINE_SCORES_PATH, index=False)

    report = {
        "model_name": "baseline_heuristico_24h",
        "score_definition": "n_alertas_24h / 24, clipped to [0, 1]",
        "split_metadata": split_metadata,
        "metrics": metrics,
    }
    BASELINE_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True)
    )
    SPLIT_METADATA_PATH.write_text(json.dumps(split_metadata, indent=2, ensure_ascii=False))

    artifact = {
        "model_name": "baseline_heuristico_24h",
        "threshold": metrics["threshold"],
        "score_feature": "n_alertas_24h",
        "score_formula": "clip(n_alertas_24h / 24, 0, 1)",
    }
    joblib.dump(artifact, BASELINE_ARTIFACT_PATH)

    return {
        "artifact_path": str(BASELINE_ARTIFACT_PATH),
        "report_path": str(BASELINE_REPORT_PATH),
        "scores_path": str(BASELINE_SCORES_PATH),
        "split_metadata_path": str(SPLIT_METADATA_PATH),
    }


def run_baseline_pipeline(
    features_path: Path = FEATURES_DATASET_PATH,
    split_dir: Path = SPLIT_DIR,
) -> dict[str, Any]:
    """Executa split temporal 70/15/15 e baseline heuristico."""
    if not features_path.exists():
        raise FileNotFoundError(f"Dataset de features nao encontrado: {features_path}")

    features = pd.read_parquet(features_path)
    train, val, test, split_metadata = temporal_train_val_test_split(features)
    split_paths = save_split_datasets(train, val, test, split_dir)
    metrics = evaluate_baseline(train, val, test)
    output_paths = save_baseline_outputs(train, val, test, metrics, split_metadata)

    return {
        "rows": int(len(features)),
        "split_paths": split_paths,
        "split_metadata": split_metadata,
        "metrics": metrics,
        **output_paths,
    }


def main() -> None:
    result = run_baseline_pipeline()
    print(f"[OK] Split train/val/test: {result['split_paths']}")
    print(f"[OK] Baseline artifact: {result['artifact_path']}")
    print(f"[OK] Baseline report: {result['report_path']}")
    print(f"[OK] Baseline scores: {result['scores_path']}")
    print(f"[OK] Threshold validacao: {result['metrics']['threshold']:.6f}")
    print(f"[OK] Recall teste: {result['metrics']['test']['recall']:.6f}")
    print(f"[OK] AUC-PR teste: {result['metrics']['test']['auc_pr']:.6f}")


if __name__ == "__main__":
    main()
