"""Avaliacao operacional de modelos para alerta Don't Go."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.train_model import load_splits, predict_scores, prepare_model_matrix
from src.models.validation import TARGET_COL
from src.utils.config import MODELS_DIR, REPORTS_DIR, SPLIT_DIR

TUNED_MODEL_PATH = MODELS_DIR / "hist_gbdt_tuned.joblib"
OPERATIONAL_REPORT_JSON = REPORTS_DIR / "operational_metrics_report.json"
BUDGET_METRICS_CSV = REPORTS_DIR / "operational_budget_metrics.csv"
DAILY_TOPK_METRICS_CSV = REPORTS_DIR / "operational_daily_topk_metrics.csv"
DEDUP_ALERTS_CSV = REPORTS_DIR / "operational_deduplicated_alerts.csv"


def load_model_artifact(model_path: Path = TUNED_MODEL_PATH) -> dict[str, Any]:
    """Carrega artefato de modelo com contrato de inferencia."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefato nao encontrado: {model_path}. Execute make tune-hist-gbdt."
        )
    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "threshold"}
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Artefato invalido. Chaves ausentes: {sorted(missing)}")
    return artifact


def score_split(df: pd.DataFrame, artifact: dict[str, Any], split_name: str) -> pd.DataFrame:
    """Calcula score do modelo para um split temporal."""
    feature_columns = artifact["feature_columns"]
    matrix = prepare_model_matrix(df, feature_columns)
    out = df[["Id", "Tag", "Fim", TARGET_COL]].copy()
    out["split"] = split_name
    out["score"] = predict_scores(artifact["model"], matrix)
    out["threshold"] = float(artifact["threshold"])
    out["prediction"] = (out["score"] >= out["threshold"]).astype(int)
    out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce", utc=True)
    return out


def build_scored_dataset(
    artifact: dict[str, Any],
    split_dir: Path = SPLIT_DIR,
) -> pd.DataFrame:
    """Gera scores para treino, validacao e teste usando o modelo tunado."""
    train, val, test = load_splits(split_dir)
    frames = [
        score_split(train, artifact, "train"),
        score_split(val, artifact, "val"),
        score_split(test, artifact, "test"),
    ]
    return pd.concat(frames, ignore_index=True)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_threshold_metrics(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Resume metricas no threshold calibrado do artefato."""
    rows = []
    for split_name, group in scored.groupby("split", sort=False):
        y_true = group[TARGET_COL].astype(int).to_numpy()
        y_pred = group["prediction"].astype(int).to_numpy()
        true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
        false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
        false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
        predicted_positive = int(y_pred.sum())
        positives = int(y_true.sum())
        prevalence = float(y_true.mean()) if len(y_true) else 0.0
        precision = _safe_divide(true_positive, predicted_positive)
        recall = _safe_divide(true_positive, positives)
        rows.append(
            {
                "split": split_name,
                "rows": int(len(group)),
                "prevalence": prevalence,
                "threshold": float(group["threshold"].iloc[0]),
                "predicted_positive": predicted_positive,
                "predicted_positive_rate": _safe_divide(predicted_positive, len(group)),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "lift_vs_random": _safe_divide(precision, prevalence),
            }
        )
    return rows


def compute_budget_metrics(
    scored: pd.DataFrame,
    budgets: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Calcula qualidade do ranking para diferentes orcamentos de alerta."""
    rows = []
    for split_name, group in scored.groupby("split", sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        positives = int(ordered[TARGET_COL].sum())
        prevalence = float(ordered[TARGET_COL].mean()) if len(ordered) else 0.0
        days = max((ordered["Fim"].max() - ordered["Fim"].min()).days + 1, 1)
        for budget in budgets:
            selected_n = max(int(np.ceil(len(ordered) * budget)), 1)
            selected = ordered.head(selected_n)
            true_positive = int(selected[TARGET_COL].sum())
            precision = _safe_divide(true_positive, selected_n)
            recall = _safe_divide(true_positive, positives)
            rows.append(
                {
                    "split": split_name,
                    "budget_pct": budget,
                    "selected_rows": selected_n,
                    "alerts_per_day": selected_n / days,
                    "precision_at_budget": precision,
                    "recall_at_budget": recall,
                    "lift_vs_random": _safe_divide(precision, prevalence),
                    "positives_captured": true_positive,
                    "total_positives": positives,
                }
            )
    return pd.DataFrame(rows)


def build_tag_day_panel(scored: pd.DataFrame, split_name: str = "test") -> pd.DataFrame:
    """Agrega ciclos em uma visao operacional Tag-dia."""
    split_df = scored.loc[scored["split"] == split_name].copy()
    split_df["data"] = split_df["Fim"].dt.date
    return (
        split_df.groupby(["data", "Tag"], as_index=False)
        .agg(
            score=("score", "max"),
            target_4h=(TARGET_COL, "max"),
            ciclos=("Id", "count"),
        )
        .sort_values(["data", "score"], ascending=[True, False])
    )


def compute_daily_topk_metrics(
    scored: pd.DataFrame,
    top_k_values: tuple[int, ...] = (3, 5, 10, 15, 20),
    split_name: str = "test",
) -> pd.DataFrame:
    """Avalia o cenario mais proximo do painel: top K Tags por dia."""
    panel = build_tag_day_panel(scored, split_name)
    prevalence = float(panel["target_4h"].mean()) if len(panel) else 0.0
    total_positives = int(panel["target_4h"].sum())
    n_days = max(panel["data"].nunique(), 1)

    rows = []
    for top_k in top_k_values:
        selected = panel.groupby("data", group_keys=False).head(top_k)
        selected_count = int(len(selected))
        true_positive = int(selected["target_4h"].sum())
        precision = _safe_divide(true_positive, selected_count)
        recall = _safe_divide(true_positive, total_positives)
        rows.append(
            {
                "split": split_name,
                "top_k_tags_per_day": top_k,
                "days": int(n_days),
                "selected_alerts": selected_count,
                "alerts_per_day": selected_count / n_days,
                "tag_day_prevalence": prevalence,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_vs_random": _safe_divide(precision, prevalence),
                "positives_captured": true_positive,
                "total_positives": total_positives,
            }
        )
    return pd.DataFrame(rows)


def deduplicate_alerts(
    scored: pd.DataFrame,
    split_name: str = "test",
    cooldown_hours: int = 4,
) -> pd.DataFrame:
    """Remove alertas repetidos por Tag dentro de uma janela de cooldown."""
    split_df = scored.loc[
        (scored["split"] == split_name) & (scored["prediction"] == 1)
    ].sort_values(["Tag", "Fim", "score"], ascending=[True, True, False])

    kept_rows = []
    cooldown = pd.Timedelta(hours=cooldown_hours)
    for _, group in split_df.groupby("Tag", sort=False):
        last_kept_time = pd.Timestamp.min.tz_localize("UTC")
        for _, row in group.iterrows():
            if row["Fim"] >= last_kept_time + cooldown:
                kept_rows.append(row)
                last_kept_time = row["Fim"]

    if not kept_rows:
        return split_df.head(0).copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def summarize_deduplicated_alerts(
    scored: pd.DataFrame,
    split_name: str = "test",
    cooldown_hours: int = 4,
) -> dict[str, Any]:
    """Resume volume e precisao depois de anti-spam por Tag."""
    dedup = deduplicate_alerts(scored, split_name, cooldown_hours)
    split_df = scored.loc[scored["split"] == split_name]
    positives = int(split_df[TARGET_COL].sum())
    true_positive = int(dedup[TARGET_COL].sum()) if len(dedup) else 0
    days = max((split_df["Fim"].max() - split_df["Fim"].min()).days + 1, 1)
    return {
        "split": split_name,
        "cooldown_hours": cooldown_hours,
        "raw_predicted_positive": int(split_df["prediction"].sum()),
        "deduplicated_alerts": int(len(dedup)),
        "deduplicated_alerts_per_day": len(dedup) / days,
        "deduplicated_precision": _safe_divide(true_positive, len(dedup)),
        "row_positive_capture_after_dedup": _safe_divide(true_positive, positives),
    }


def run_operational_evaluation(
    model_path: Path = TUNED_MODEL_PATH,
    split_dir: Path = SPLIT_DIR,
) -> dict[str, Any]:
    """Executa avaliacao operacional completa."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_model_artifact(model_path)
    scored = build_scored_dataset(artifact, split_dir)

    threshold_metrics = summarize_threshold_metrics(scored)
    budget_metrics = compute_budget_metrics(scored)
    daily_topk_metrics = compute_daily_topk_metrics(scored)
    dedup_alerts = deduplicate_alerts(scored)
    dedup_summary = summarize_deduplicated_alerts(scored)

    budget_metrics.to_csv(BUDGET_METRICS_CSV, index=False)
    daily_topk_metrics.to_csv(DAILY_TOPK_METRICS_CSV, index=False)
    dedup_alerts.to_csv(DEDUP_ALERTS_CSV, index=False)

    report = {
        "model_path": str(model_path),
        "model_name": artifact.get("model_name", "unknown"),
        "threshold": float(artifact["threshold"]),
        "feature_count": len(artifact["feature_columns"]),
        "threshold_metrics": threshold_metrics,
        "test_budget_metrics": budget_metrics.loc[budget_metrics["split"] == "test"].to_dict(
            orient="records"
        ),
        "test_daily_topk_metrics": daily_topk_metrics.to_dict(orient="records"),
        "deduplicated_alerts": dedup_summary,
        "outputs": {
            "budget_metrics_csv": str(BUDGET_METRICS_CSV),
            "daily_topk_metrics_csv": str(DAILY_TOPK_METRICS_CSV),
            "deduplicated_alerts_csv": str(DEDUP_ALERTS_CSV),
        },
    }
    OPERATIONAL_REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True)
    )
    return {
        "json_path": str(OPERATIONAL_REPORT_JSON),
        "budget_metrics_csv": str(BUDGET_METRICS_CSV),
        "daily_topk_metrics_csv": str(DAILY_TOPK_METRICS_CSV),
        "deduplicated_alerts_csv": str(DEDUP_ALERTS_CSV),
        "report": report,
    }


def main() -> None:
    result = run_operational_evaluation()
    report = result["report"]
    test_threshold = next(item for item in report["threshold_metrics"] if item["split"] == "test")
    best_topk = report["test_daily_topk_metrics"][0]
    print(f"[OK] Relatorio operacional: {result['json_path']}")
    print(f"[OK] Modelo avaliado: {report['model_name']}")
    print(f"[OK] Threshold: {report['threshold']:.6f}")
    print(f"[OK] Test lift no threshold: {test_threshold['lift_vs_random']:.3f}")
    print(
        "[OK] Top "
        f"{best_topk['top_k_tags_per_day']} Tags/dia: "
        f"precision={best_topk['precision_at_k']:.3f}, "
        f"recall={best_topk['recall_at_k']:.3f}, "
        f"lift={best_topk['lift_vs_random']:.3f}"
    )
    print(f"[OK] Budget metrics: {result['budget_metrics_csv']}")
    print(f"[OK] Daily top-k metrics: {result['daily_topk_metrics_csv']}")


if __name__ == "__main__":
    main()
