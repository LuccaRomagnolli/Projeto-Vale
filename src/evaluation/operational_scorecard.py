"""Scorecard operacional compartilhado para treino, benchmark e avaliacao."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.validation import TARGET_COL

DEFAULT_TOP_K_VALUES = (3, 5, 10, 15, 20)
PRIMARY_TOP_K = 15
SEGMENT_COLUMNS = ("Frota", "Tipo", "turno", "Classe")


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide protegendo contra denominador zero."""
    return float(numerator / denominator) if denominator else 0.0


def decode_one_hot_prefix(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Reconstrói categoria a partir de colunas one-hot de um prefixo."""
    columns = [column for column in df.columns if column.startswith(f"{prefix}_")]
    if not columns:
        return pd.Series(["desconhecido"] * len(df), index=df.index)

    matrix = df[columns].fillna(False).astype(bool)
    decoded = matrix.idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)
    decoded.loc[~matrix.any(axis=1)] = "desconhecido"
    return decoded


def add_segment_context(scored: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas de setor/segmento usadas na avaliacao operacional."""
    out = scored.copy()
    for column in ("Classe", "turno"):
        values = (
            source_df[column].astype(str).to_numpy()
            if column in source_df.columns
            else np.repeat("desconhecido", len(source_df))
        )
        out[column] = values

    out["Frota"] = decode_one_hot_prefix(source_df, "Frota").astype(str).to_numpy()
    out["Tipo"] = decode_one_hot_prefix(source_df, "Tipo").astype(str).to_numpy()
    return out


def build_scored_frame(
    source_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    split_name: str,
    include_segments: bool = True,
) -> pd.DataFrame:
    """Monta frame auditavel de scores com o contrato operacional comum."""
    base_cols = [
        column for column in ["Id", "Tag", "Fim", TARGET_COL] if column in source_df.columns
    ]
    out = source_df[base_cols].copy()
    out["split"] = split_name
    out["score"] = np.asarray(scores, dtype=float)
    out["threshold"] = float(threshold)
    out["prediction"] = (out["score"] >= out["threshold"]).astype(int)
    out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce", utc=True)
    if include_segments:
        out = add_segment_context(out, source_df)
    return out


def build_tag_day_panel(scored: pd.DataFrame, split_name: str = "test") -> pd.DataFrame:
    """Agrega ciclos em uma visao operacional Tag-dia."""
    split_df = scored.loc[scored["split"] == split_name].copy()
    split_df["Fim"] = pd.to_datetime(split_df["Fim"], errors="coerce", utc=True)
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
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
    split_name: str = "test",
) -> pd.DataFrame:
    """Avalia o cenario mais proximo do painel: Top K Tags por dia."""
    panel = build_tag_day_panel(scored, split_name)
    prevalence = float(panel[TARGET_COL].mean()) if len(panel) else 0.0
    total_positives = int(panel[TARGET_COL].sum()) if len(panel) else 0
    n_days = max(panel["data"].nunique(), 1) if len(panel) else 1

    rows = []
    for top_k in top_k_values:
        selected = panel.groupby("data", group_keys=False).head(top_k)
        selected_count = int(len(selected))
        true_positive = int(selected[TARGET_COL].sum()) if len(selected) else 0
        precision = safe_divide(true_positive, selected_count)
        recall = safe_divide(true_positive, total_positives)
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
                "lift_vs_random": safe_divide(precision, prevalence),
                "positives_captured": true_positive,
                "total_positives": total_positives,
            }
        )
    return pd.DataFrame(rows)


def compute_split_topk_metrics(
    scored: pd.DataFrame,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
) -> pd.DataFrame:
    """Calcula TopK Tag-dia para todos os splits presentes."""
    frames = [
        compute_daily_topk_metrics(scored, top_k_values=top_k_values, split_name=str(split_name))
        for split_name in scored["split"].drop_duplicates()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_segment_tag_day_panel(scored: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Agrega score e target no nivel segmento-dia-Tag."""
    frame = scored.copy()
    frame["Fim"] = pd.to_datetime(frame["Fim"], errors="coerce", utc=True)
    frame["data"] = frame["Fim"].dt.date
    return (
        frame.groupby([segment_col, "data", "Tag"], as_index=False)
        .agg(
            score=("score", "max"),
            target_4h=(TARGET_COL, "max"),
            ciclos=("Id", "count"),
        )
        .sort_values([segment_col, "data", "score"], ascending=[True, True, False])
    )


def topk_metrics_by_segment(
    scored: pd.DataFrame,
    segment_cols: tuple[str, ...] = SEGMENT_COLUMNS,
    top_k_values: tuple[int, ...] = (3, 5, 10, 15),
    min_tag_days: int = 20,
    min_positives: int = 10,
) -> pd.DataFrame:
    """Calcula Precision/Recall/Lift@TopK Tag-dia dentro de cada segmento."""
    rows = []
    for segment_col in segment_cols:
        if segment_col not in scored.columns:
            continue
        panel = build_segment_tag_day_panel(scored, segment_col)
        for segment_value, group in panel.groupby(segment_col, dropna=False):
            prevalence = float(group[TARGET_COL].mean()) if len(group) else 0.0
            positives = int(group[TARGET_COL].sum())
            n_days = max(group["data"].nunique(), 1)
            status = "ok"
            recommendation = "usar TopK como canal principal"
            if len(group) < min_tag_days:
                status = "inconclusivo_baixa_amostra"
                recommendation = "coletar mais dias de observacao"
            elif positives < min_positives:
                status = "inconclusivo_baixa_prevalencia"
                recommendation = "tratar segmento em trilha separada"
            for top_k in top_k_values:
                selected = group.groupby("data", group_keys=False).head(top_k)
                selected_count = int(len(selected))
                true_positive = int(selected[TARGET_COL].sum())
                precision = safe_divide(true_positive, selected_count)
                recall = safe_divide(true_positive, positives)

                rows.append(
                    {
                        "segment_col": segment_col,
                        "segment_value": str(segment_value),
                        "top_k_tags_per_day": top_k,
                        "tag_days": int(len(group)),
                        "days": int(n_days),
                        "selected_alerts": selected_count,
                        "alerts_per_day": selected_count / n_days,
                        "tag_day_prevalence": prevalence,
                        "precision_at_k": precision,
                        "recall_at_k": recall,
                        "lift_vs_random": safe_divide(precision, prevalence),
                        "positives_captured": true_positive,
                        "total_positives": positives,
                        "status": status,
                        "recommendation": recommendation,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["segment_col", "top_k_tags_per_day", "lift_vs_random"],
        ascending=[True, True, False],
    )


def primary_topk_by_split(
    topk_metrics: pd.DataFrame,
    top_k: int = PRIMARY_TOP_K,
) -> dict[str, dict[str, Any]]:
    """Indexa a metrica operacional primaria por split."""
    if topk_metrics.empty:
        return {}
    primary = topk_metrics.loc[topk_metrics["top_k_tags_per_day"] == top_k]
    return {
        str(row["split"]): row
        for row in primary.to_dict(orient="records")
        if "split" in row
    }


def attach_operational_metrics(
    metrics: dict[str, Any],
    scored: pd.DataFrame,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
    primary_top_k: int = PRIMARY_TOP_K,
) -> dict[str, Any]:
    """Anexa scorecard operacional ao dicionario de metricas tecnicas."""
    enriched = metrics.copy()
    topk = compute_split_topk_metrics(scored, top_k_values=top_k_values)
    enriched["operational_primary_top_k"] = primary_top_k
    enriched["operational_topk"] = topk.to_dict(orient="records")
    for split_name, row in primary_topk_by_split(topk, top_k=primary_top_k).items():
        if split_name not in enriched:
            continue
        enriched[split_name] = enriched[split_name].copy()
        enriched[split_name][f"top{primary_top_k}_precision_at_k"] = float(
            row["precision_at_k"]
        )
        enriched[split_name][f"top{primary_top_k}_recall_at_k"] = float(row["recall_at_k"])
        enriched[split_name][f"top{primary_top_k}_lift_vs_random"] = float(
            row["lift_vs_random"]
        )
        enriched[split_name][f"top{primary_top_k}_alerts_per_day"] = float(
            row["alerts_per_day"]
        )
    return enriched
