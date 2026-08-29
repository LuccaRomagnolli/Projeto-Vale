"""Etapa 5: engenharia de features sem vazamento temporal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import (
    CRITICAL_EVENTS_PATH,
    FEATURES_DATASET_PATH,
    FEATURES_DIR,
    LABELED_DATASET_PATH,
)
from src.utils.logging_config import get_logger, setup_logging
from src.utils.timeutils import NS_PER_DAY, NS_PER_HOUR, to_epoch_ns

logger = get_logger(__name__)

ROLLING_WINDOWS_HOURS = (4, 8, 24)
REFERENCE_COL = "Fim"
TAG_COL = "Tag"
TARGET_COL = "target_4h"


def load_feature_inputs(
    labeled_path: Path = LABELED_DATASET_PATH,
    critical_events_path: Path = CRITICAL_EVENTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dataset rotulado e eventos criticos extraidos na etapa 3b."""
    if not labeled_path.exists():
        raise FileNotFoundError(f"Dataset rotulado nao encontrado: {labeled_path}")
    if not critical_events_path.exists():
        raise FileNotFoundError(f"Eventos criticos nao encontrados: {critical_events_path}")
    return pd.read_parquet(labeled_path), pd.read_parquet(critical_events_path)


def _assign_turn(hour: float) -> str:
    if pd.isna(hour):
        return "desconhecido"
    if 6 <= int(hour) <= 13:
        return "manha"
    if 14 <= int(hour) <= 21:
        return "tarde"
    return "noite"


def create_temporal_features(df: pd.DataFrame, dt_col: str = "Inicio") -> pd.DataFrame:
    """Cria features temporais basicas a partir de Inicio/Fim."""
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce", utc=True)

    out["hora_do_dia"] = out[dt_col].dt.hour
    out["dia_da_semana"] = out[dt_col].dt.dayofweek
    out["mes"] = out[dt_col].dt.month
    out["turno"] = out["hora_do_dia"].map(_assign_turn)
    out["is_fim_de_semana"] = out["dia_da_semana"].isin([5, 6]).astype(int)

    if "Fim" in out.columns and "Inicio" in out.columns:
        out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce", utc=True)
        out["duracao_ciclo_min"] = (out["Fim"] - out["Inicio"]).dt.total_seconds() / 60.0

    # Compatibilidade com testes e notebooks iniciais.
    out["hora"] = out["hora_do_dia"]
    out["dia_semana"] = out["dia_da_semana"]
    return out


def add_cycle_rolling_features(
    df: pd.DataFrame,
    windows_hours: tuple[int, ...] = ROLLING_WINDOWS_HOURS,
    reference_col: str = REFERENCE_COL,
    tag_col: str = TAG_COL,
) -> pd.DataFrame:
    """Adiciona features rolling de ciclos por equipamento."""
    out = df.copy().sort_values([tag_col, reference_col]).reset_index(drop=True)
    out[reference_col] = pd.to_datetime(out[reference_col], errors="coerce", utc=True)

    for window in windows_hours:
        out[f"n_ciclos_{window}h"] = 0.0
        out[f"duracao_media_ciclo_{window}h"] = np.nan
        out[f"duracao_std_ciclo_{window}h"] = np.nan
        out[f"freq_classe_atividade_{window}h"] = 0.0
        out[f"n_classes_distintas_{window}h"] = 0.0

    for _, group in out.groupby(tag_col, sort=False):
        idx = group.index
        temporal = group.set_index(reference_col).sort_index()
        ones = pd.Series(1.0, index=temporal.index)
        duration = temporal["duracao_ciclo_min"]

        for window in windows_hours:
            n_ciclos = ones.rolling(f"{window}h", closed="left").sum()
            duration_mean = duration.rolling(f"{window}h", closed="left").mean()
            duration_std = duration.rolling(f"{window}h", closed="left").std()
            out.loc[idx, f"n_ciclos_{window}h"] = n_ciclos.to_numpy()
            out.loc[idx, f"duracao_media_ciclo_{window}h"] = duration_mean.to_numpy()
            out.loc[idx, f"duracao_std_ciclo_{window}h"] = duration_std.to_numpy()

            row_times = to_epoch_ns(pd.DatetimeIndex(temporal.index))
            left_times = row_times - (window * NS_PER_HOUR)
            left_positions = np.searchsorted(row_times, left_times, side="left")
            class_values = temporal["Classe"].astype(str).to_numpy()
            diversity = [
                float(len(set(class_values[left_pos:pos])))
                for pos, left_pos in enumerate(left_positions)
            ]
            out.loc[idx, f"n_classes_distintas_{window}h"] = diversity

        for _, class_group in group.groupby("Classe", dropna=False, sort=False):
            class_idx = class_group.index
            class_temporal = class_group.set_index(reference_col).sort_index()
            class_ones = pd.Series(1.0, index=class_temporal.index)
            for window in windows_hours:
                class_counts = class_ones.rolling(f"{window}h", closed="left").sum()
                denominators = out.loc[class_idx, f"n_ciclos_{window}h"].replace(0, np.nan)
                frequency = class_counts.to_numpy() / denominators.to_numpy()
                out.loc[class_idx, f"freq_classe_atividade_{window}h"] = np.nan_to_num(frequency)

    return out


def add_alert_history_features(
    df: pd.DataFrame,
    critical_events_df: pd.DataFrame,
    windows_hours: tuple[int, ...] = ROLLING_WINDOWS_HOURS,
    reference_col: str = REFERENCE_COL,
    tag_col: str = TAG_COL,
) -> pd.DataFrame:
    """Adiciona features historicas de alertas usando apenas eventos anteriores ao ciclo."""
    out = df.copy().sort_values([tag_col, reference_col]).reset_index(drop=True)
    out[reference_col] = pd.to_datetime(out[reference_col], errors="coerce", utc=True)

    critical = critical_events_df.copy()
    critical["TAG"] = critical["TAG"].astype(str).str.strip().str.upper()
    critical["EVENT_TIME"] = pd.to_datetime(critical["EVENT_TIME"], errors="coerce", utc=True)
    critical = critical.dropna(subset=["EVENT_TIME"]).sort_values(["TAG", "EVENT_TIME"])

    for window in windows_hours:
        out[f"n_alertas_{window}h"] = 0
        out[f"alertas_por_hora_{window}h"] = 0.0
    out["dias_desde_ultimo_alerta"] = np.nan

    events_by_tag = {
        tag: to_epoch_ns(group["EVENT_TIME"]) for tag, group in critical.groupby("TAG", sort=False)
    }

    for tag, group in out.groupby(tag_col, sort=False):
        row_idx = group.index
        row_times = to_epoch_ns(group[reference_col])
        event_times = events_by_tag.get(str(tag).strip().upper())
        if event_times is None or len(event_times) == 0:
            continue

        right_positions = np.searchsorted(event_times, row_times, side="left")
        for window in windows_hours:
            left_times = row_times - (window * NS_PER_HOUR)
            left_positions = np.searchsorted(event_times, left_times, side="left")
            alert_counts = right_positions - left_positions
            out.loc[row_idx, f"n_alertas_{window}h"] = alert_counts
            out.loc[row_idx, f"alertas_por_hora_{window}h"] = alert_counts / window

        has_prior = right_positions > 0
        prior_times = np.full(len(row_times), np.nan)
        prior_times[has_prior] = event_times[right_positions[has_prior] - 1]
        delta_days = (row_times - prior_times) / NS_PER_DAY
        out.loc[row_idx, "dias_desde_ultimo_alerta"] = delta_days

    out["n_precondicoes_satisfeitas_4h"] = out["n_alertas_4h"]
    out["nivel_maximo_evento_recente"] = np.where(out["n_alertas_24h"] > 0, 4, 0)
    return out


def add_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria sinais de degradacao recente comparando janelas curtas e longas."""
    out = df.copy()

    out["delta_duracao_ciclo_4h_24h"] = (
        out["duracao_media_ciclo_4h"] - out["duracao_media_ciclo_24h"]
    )
    out["ratio_duracao_ciclo_4h_24h"] = out["duracao_media_ciclo_4h"] / out[
        "duracao_media_ciclo_24h"
    ].replace(0, np.nan)
    out["delta_alertas_por_hora_4h_24h"] = out["alertas_por_hora_4h"] - out["alertas_por_hora_24h"]
    out["ratio_alertas_4h_24h"] = out["n_alertas_4h"] / out["n_alertas_24h"].replace(0, np.nan)
    out["delta_n_ciclos_4h_24h_norm"] = (out["n_ciclos_4h"] / 4.0) - (out["n_ciclos_24h"] / 24.0)
    out["delta_freq_classe_4h_24h"] = (
        out["freq_classe_atividade_4h"] - out["freq_classe_atividade_24h"]
    )
    out["delta_classes_distintas_4h_24h"] = (
        out["n_classes_distintas_4h"] - out["n_classes_distintas_24h"]
    )

    degradation_cols = [
        "delta_duracao_ciclo_4h_24h",
        "ratio_duracao_ciclo_4h_24h",
        "delta_alertas_por_hora_4h_24h",
        "ratio_alertas_4h_24h",
        "delta_n_ciclos_4h_24h_norm",
        "delta_freq_classe_4h_24h",
        "delta_classes_distintas_4h_24h",
    ]
    out[degradation_cols] = out[degradation_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_categorical_encodings(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Encoding categorico global -- LEGADO, fora do pipeline oficial.

    Calcula as estatisticas sobre todo o dataframe recebido. Quando aplicado
    antes do split temporal, como era feito ate a Fase 2, vaza a distribuicao
    futura para o treino e codifica validacao e teste com a propria massa de
    frequencia deles.

    Mantido exclusivamente como braco de controle em
    `src/evaluation/leakage_ablation.py`, que mede o custo da correcao. Para o
    pipeline use `src.features.encoders.CategoricalEncoder`, que segue o
    contrato fit/transform.
    """
    out = df.copy()
    total_rows = len(out)

    tag_freq = out["Tag"].value_counts(normalize=True)
    out["Tag_freq"] = out["Tag"].map(tag_freq).fillna(0.0)

    if "Operador" in out.columns:
        operador_freq = out["Operador"].value_counts(normalize=True)
        out["Operador_freq"] = out["Operador"].map(operador_freq).fillna(0.0)
    else:
        out["Operador_freq"] = 0.0

    if total_rows and target_col in out.columns:
        temporal = out.sort_values("Fim").copy()
        prior_sum = (
            temporal.groupby("Classe", dropna=False)[target_col].cumsum() - temporal[target_col]
        )
        prior_count = temporal.groupby("Classe", dropna=False).cumcount()
        global_prior_sum = temporal[target_col].cumsum() - temporal[target_col]
        global_prior_count = np.arange(len(temporal))
        global_prior = np.divide(
            global_prior_sum,
            global_prior_count,
            out=np.zeros(len(temporal), dtype=float),
            where=global_prior_count > 0,
        )
        classe_prior = np.divide(
            prior_sum,
            prior_count,
            out=np.full(len(temporal), np.nan, dtype=float),
            where=prior_count > 0,
        )
        temporal["Classe_target_enc"] = np.nan_to_num(classe_prior, nan=0.0)
        temporal.loc[prior_count == 0, "Classe_target_enc"] = global_prior[prior_count == 0]
        out["Classe_target_enc"] = temporal.sort_index()["Classe_target_enc"]
    else:
        out["Classe_target_enc"] = 0.0

    out = pd.get_dummies(out, columns=["Frota", "Tipo"], prefix=["Frota", "Tipo"], dummy_na=False)
    return out


def build_feature_dataset(
    labeled_df: pd.DataFrame,
    critical_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Executa o pipeline de feature engineering independente de split.

    Os encodings categoricos NAO sao aplicados aqui. Eles dependem de
    estatisticas que precisam ser aprendidas apenas no treino e por isso vivem
    em `src/features/encoders.py`, aplicados depois do split temporal. Este
    dataset preserva as colunas categoricas cruas (`Tag`, `Classe`, `Frota`,
    `Tipo`) para que o encoder possa ajusta-las sem vazamento.
    """
    features = create_temporal_features(labeled_df, dt_col="Inicio")
    features = add_cycle_rolling_features(features)
    features = add_alert_history_features(features, critical_events_df)
    features = add_degradation_features(features)
    return features


def write_feature_report(features_df: pd.DataFrame, output_dir: Path = FEATURES_DIR) -> Path:
    """Persiste relatorio simples de features para auditoria."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "feature_report.json"
    payload: dict[str, Any] = {
        "rows": int(len(features_df)),
        "columns": int(features_df.shape[1]),
        "target_4h_positives": int(features_df[TARGET_COL].sum()),
        "target_4h_positive_rate_pct": round(float(features_df[TARGET_COL].mean() * 100), 6),
        "feature_columns": list(features_df.columns),
        "null_pct_top10": (
            (features_df.isna().mean() * 100.0)
            .sort_values(ascending=False)
            .head(10)
            .round(6)
            .to_dict()
        ),
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return report_path


def save_feature_dataset(
    features_df: pd.DataFrame,
    output_path: Path = FEATURES_DATASET_PATH,
) -> Path:
    """Salva dataset de features para modelagem."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    return output_path


def run_feature_pipeline(
    labeled_path: Path = LABELED_DATASET_PATH,
    critical_events_path: Path = CRITICAL_EVENTS_PATH,
    output_path: Path = FEATURES_DATASET_PATH,
) -> dict[str, Any]:
    """Executa feature engineering de ponta a ponta."""
    labeled_df, critical_events_df = load_feature_inputs(labeled_path, critical_events_path)
    features_df = build_feature_dataset(labeled_df, critical_events_df)
    dataset_path = save_feature_dataset(features_df, output_path)
    report_path = write_feature_report(features_df, output_path.parent)
    return {
        "dataset_path": str(dataset_path),
        "report_path": str(report_path),
        "rows": int(len(features_df)),
        "columns": int(features_df.shape[1]),
        "target_4h_positives": int(features_df[TARGET_COL].sum()),
    }


def main() -> None:
    setup_logging()
    result = run_feature_pipeline()
    logger.info(f"Feature dataset: {result['dataset_path']}")
    logger.info(f"Feature report: {result['report_path']}")
    logger.info(f"Linhas: {result['rows']}")
    logger.info(f"Colunas: {result['columns']}")
    logger.info(f"Positivos target_4h: {result['target_4h_positives']}")


if __name__ == "__main__":
    main()
