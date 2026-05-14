"""Analise segmentada das metricas operacionais por frota, tipo, turno e Tag."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.evaluation.evaluate_model import _safe_divide, score_split
from src.evaluation.operational_scorecard import (
    build_segment_tag_day_panel as build_operational_segment_tag_day_panel,
)
from src.evaluation.operational_scorecard import (
    decode_one_hot_prefix as decode_operational_one_hot_prefix,
)
from src.evaluation.operational_scorecard import (
    topk_metrics_by_segment as operational_topk_metrics_by_segment,
)
from src.models.model_selection import SELECTED_MODEL_PATH
from src.models.train_model import load_splits
from src.models.validation import TARGET_COL
from src.utils.config import REPORTS_SEGMENT_DIR, SPLIT_DIR
from src.utils.metadata import to_repo_relative_path

SEGMENT_REPORT_JSON = REPORTS_SEGMENT_DIR / "segment_operational_report.json"
SEGMENT_THRESHOLD_CSV = REPORTS_SEGMENT_DIR / "segment_threshold_metrics.csv"
SEGMENT_TOPK_CSV = REPORTS_SEGMENT_DIR / "segment_topk_tag_day_metrics.csv"
TAG_HOTSPOTS_CSV = REPORTS_SEGMENT_DIR / "segment_tag_hotspots.csv"


def decode_one_hot_prefix(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Reconstrói categoria a partir de colunas one-hot de um prefixo."""
    return decode_operational_one_hot_prefix(df, prefix)


def load_artifact(model_path: Path = SELECTED_MODEL_PATH) -> dict[str, Any]:
    """Carrega modelo tunado para analise segmentada."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefato nao encontrado: {model_path}. Execute make model-selection."
        )
    return joblib.load(model_path)


def score_test_with_segments(
    model_path: Path = SELECTED_MODEL_PATH,
    split_dir: Path = SPLIT_DIR,
) -> pd.DataFrame:
    """Calcula scores do teste e adiciona segmentos de negocio."""
    artifact = load_artifact(model_path)
    _, _, test = load_splits(split_dir)
    scored = score_split(test, artifact, "test")

    context_cols = ["Classe", "turno"]
    for column in context_cols:
        scored[column] = test[column].astype(str) if column in test.columns else "desconhecido"

    scored["Frota"] = decode_one_hot_prefix(test, "Frota")
    scored["Tipo"] = decode_one_hot_prefix(test, "Tipo")
    scored["semana_iso"] = scored["Fim"].dt.isocalendar().week.astype(int)
    return scored


def threshold_metrics_by_segment(
    scored: pd.DataFrame,
    segment_cols: tuple[str, ...] = ("Frota", "Tipo", "turno", "Classe"),
    min_rows: int = 50,
    min_positives: int = 10,
) -> pd.DataFrame:
    """Calcula metricas no threshold por segmento."""
    rows = []
    for segment_col in segment_cols:
        for segment_value, group in scored.groupby(segment_col, dropna=False):
            y_true = group[TARGET_COL].astype(int)
            y_pred = group["prediction"].astype(int)
            positives = int(y_true.sum())
            status = "ok"
            recommendation = "manter threshold global e monitorar"
            if len(group) < min_rows:
                status = "inconclusivo_baixa_amostra"
                recommendation = "coletar mais dados antes de decidir por segmento"
            elif positives < min_positives:
                status = "inconclusivo_baixa_prevalencia"
                recommendation = "avaliar regra heuristica ou janela de avaliacao maior"
            predicted_positive = int(y_pred.sum())
            true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
            false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
            false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
            prevalence = float(y_true.mean()) if len(group) else 0.0
            precision = _safe_divide(true_positive, predicted_positive)
            recall = _safe_divide(true_positive, positives)

            rows.append(
                {
                    "segment_col": segment_col,
                    "segment_value": str(segment_value),
                    "rows": int(len(group)),
                    "prevalence": prevalence,
                    "predicted_positive": predicted_positive,
                    "predicted_positive_rate": _safe_divide(predicted_positive, len(group)),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "precision": precision,
                    "recall": recall,
                    "lift_vs_random": _safe_divide(precision, prevalence),
                    "status": status,
                    "recommendation": recommendation,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["segment_col", "lift_vs_random"], ascending=[True, False]
    )


def build_segment_tag_day_panel(scored: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Agrega score e target no nivel segmento-dia-Tag."""
    return build_operational_segment_tag_day_panel(scored, segment_col)


def topk_metrics_by_segment(
    scored: pd.DataFrame,
    segment_cols: tuple[str, ...] = ("Frota", "Tipo", "turno", "Classe"),
    top_k_values: tuple[int, ...] = (3, 5, 10, 15),
    min_tag_days: int = 20,
    min_positives: int = 10,
) -> pd.DataFrame:
    """Calcula Precision/Recall/Lift@TopK Tag-dia dentro de cada segmento."""
    return operational_topk_metrics_by_segment(
        scored,
        segment_cols=segment_cols,
        top_k_values=top_k_values,
        min_tag_days=min_tag_days,
        min_positives=min_positives,
    )


def tag_hotspots(scored: pd.DataFrame, top_k: int = 15, min_tag_days: int = 5) -> pd.DataFrame:
    """Identifica Tags com maior volume de erros e oportunidades de ajuste."""
    frame = scored.copy()
    frame["data"] = frame["Fim"].dt.date
    panel = (
        frame.groupby(["data", "Tag"], as_index=False)
        .agg(
            score=("score", "max"),
            target_4h=(TARGET_COL, "max"),
            ciclos=("Id", "count"),
        )
        .sort_values(["data", "score"], ascending=[True, False])
    )
    panel["selected_topk"] = (panel.groupby("data", group_keys=False).cumcount() < top_k).astype(
        int
    )
    panel["selected_true_positive"] = panel["selected_topk"] * panel["target_4h"]
    panel["selected_false_positive"] = panel["selected_topk"] * (1 - panel["target_4h"])
    panel["missed_positive"] = (1 - panel["selected_topk"]) * panel["target_4h"]

    out = (
        panel.groupby("Tag", as_index=False)
        .agg(
            tag_days=("data", "nunique"),
            avg_score=("score", "mean"),
            max_score=("score", "max"),
            positive_days=("target_4h", "sum"),
            selected_days=("selected_topk", "sum"),
            selected_true_positive_days=("selected_true_positive", "sum"),
            selected_false_positive_days=("selected_false_positive", "sum"),
            missed_positive_days=("missed_positive", "sum"),
        )
        .query("tag_days >= @min_tag_days")
    )
    out["selected_precision"] = out.apply(
        lambda row: _safe_divide(row["selected_true_positive_days"], row["selected_days"]), axis=1
    )
    out["tag_positive_rate"] = out["positive_days"] / out["tag_days"]
    return out.sort_values(
        ["missed_positive_days", "selected_false_positive_days", "positive_days"],
        ascending=[False, False, False],
    )


def run_segment_analysis(
    model_path: Path = SELECTED_MODEL_PATH,
    split_dir: Path = SPLIT_DIR,
) -> dict[str, Any]:
    """Executa analise segmentada completa no split de teste."""
    REPORTS_SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    scored = score_test_with_segments(model_path, split_dir)
    threshold = threshold_metrics_by_segment(scored)
    topk = topk_metrics_by_segment(scored)
    hotspots = tag_hotspots(scored)

    threshold.to_csv(SEGMENT_THRESHOLD_CSV, index=False)
    topk.to_csv(SEGMENT_TOPK_CSV, index=False)
    hotspots.to_csv(TAG_HOTSPOTS_CSV, index=False)

    top15 = topk.loc[topk["top_k_tags_per_day"] == 15].copy()
    inconclusive_segments = (
        top15.loc[top15["status"] != "ok"]
        .sort_values(["segment_col", "segment_value"])
        .to_dict(orient="records")
    )
    weakest_top15 = (
        top15.sort_values(["recall_at_k", "precision_at_k"], ascending=[True, True])
        .head(10)
        .to_dict(orient="records")
    )
    strongest_top15 = (
        top15.sort_values(["lift_vs_random", "precision_at_k"], ascending=[False, False])
        .head(10)
        .to_dict(orient="records")
    )

    report = {
        "model_path": to_repo_relative_path(model_path),
        "split": "test",
        "rows": int(len(scored)),
        "segment_columns": ["Frota", "Tipo", "turno", "Classe"],
        "outputs": {
            "threshold_metrics_csv": to_repo_relative_path(SEGMENT_THRESHOLD_CSV),
            "topk_metrics_csv": to_repo_relative_path(SEGMENT_TOPK_CSV),
            "tag_hotspots_csv": to_repo_relative_path(TAG_HOTSPOTS_CSV),
        },
        "weakest_top15_segments": weakest_top15,
        "strongest_top15_segments": strongest_top15,
        "inconclusive_segments": inconclusive_segments,
        "top_tag_hotspots": hotspots.head(20).to_dict(orient="records"),
    }
    SEGMENT_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {
        "json_path": str(SEGMENT_REPORT_JSON),
        "threshold_metrics_csv": str(SEGMENT_THRESHOLD_CSV),
        "topk_metrics_csv": str(SEGMENT_TOPK_CSV),
        "tag_hotspots_csv": str(TAG_HOTSPOTS_CSV),
        "report": report,
    }


def main() -> None:
    result = run_segment_analysis()
    report = result["report"]
    weakest = report["weakest_top15_segments"][0] if report["weakest_top15_segments"] else {}
    strongest = report["strongest_top15_segments"][0] if report["strongest_top15_segments"] else {}
    print(f"[OK] Relatorio segmentado: {result['json_path']}")
    print(f"[OK] Threshold por segmento: {result['threshold_metrics_csv']}")
    print(f"[OK] TopK por segmento: {result['topk_metrics_csv']}")
    print(f"[OK] Hotspots por Tag: {result['tag_hotspots_csv']}")
    if strongest:
        print(
            "[OK] Melhor segmento Top15: "
            f"{strongest['segment_col']}={strongest['segment_value']} "
            f"precision={strongest['precision_at_k']:.3f} "
            f"recall={strongest['recall_at_k']:.3f}"
        )
    if weakest:
        print(
            "[OK] Segmento mais fragil Top15: "
            f"{weakest['segment_col']}={weakest['segment_value']} "
            f"precision={weakest['precision_at_k']:.3f} "
            f"recall={weakest['recall_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()
