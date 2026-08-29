"""Inferencia operacional com contrato de schema para artefatos treinados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.evaluation.operational_scorecard import PRIMARY_TOP_K, decode_one_hot_prefix
from src.features.encoders import load_encoder
from src.inference_contract import check_feature_coverage, validate_batch_values
from src.models.model_selection import SELECTED_MODEL_PATH
from src.models.train_baseline import ENCODER_FILENAME
from src.models.train_model import predict_scores, prepare_model_matrix
from src.utils.config import REPORTS_DIR, REPORTS_INFERENCE_DIR, SPLIT_DIR

DEFAULT_MODEL_PATH = SELECTED_MODEL_PATH
DEFAULT_ENCODER_PATH = SPLIT_DIR / ENCODER_FILENAME
DEFAULT_INPUT_PATH = SPLIT_DIR / "features_test.parquet"
DEFAULT_OUTPUT_PATH = REPORTS_INFERENCE_DIR / "inference_scores.parquet"
DEFAULT_PRIORITY_OUTPUT_PATH = REPORTS_DIR / f"daily_priority_top{PRIMARY_TOP_K}.csv"
DEFAULT_OPERATIONAL_TOP_K = PRIMARY_TOP_K


def load_model_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Carrega artefato de modelo e valida chaves obrigatorias."""
    if not model_path.exists():
        raise FileNotFoundError(f"Artefato nao encontrado: {model_path}")
    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "threshold"}
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Artefato invalido. Chaves ausentes: {sorted(missing)}")
    return artifact


def read_inference_input(input_path: Path) -> pd.DataFrame:
    """Carrega dataset de inferencia em csv ou parquet."""
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Formato nao suportado para inferencia: {suffix}")


def align_feature_schema(
    df: pd.DataFrame,
    feature_columns: list[str],
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Alinha colunas para inferencia.

    Por padrao falha quando o lote nao traz alguma feature esperada. O
    comportamento anterior -- preencher com `0.0` e apenas registrar em
    metadados -- pontuava lotes com schema alterado sem qualquer sinal.
    `allow_missing=True` mantem o preenchimento, mas como escolha explicita.
    """
    out = df.copy()
    coverage = check_feature_coverage(out, feature_columns, allow_missing=allow_missing)
    missing_columns = coverage["missing_feature_columns"]
    for column in missing_columns:
        out[column] = 0.0
    aligned = out[feature_columns]
    return aligned, missing_columns, coverage["extra_columns_ignored"]


def apply_categorical_encoder(
    features_df: pd.DataFrame,
    encoder_path: Path | None = DEFAULT_ENCODER_PATH,
) -> tuple[pd.DataFrame, bool]:
    """Aplica o encoder ajustado no treino, quando o lote traz categoricas cruas.

    Sem este passo, um lote com `Tag`/`Classe`/`Frota` crus chegaria ao modelo
    sem `Tag_freq`, `Classe_target_enc` e os indicadores one-hot -- exatamente
    as colunas que o preenchimento silencioso com zero mascarava.
    """
    needs_encoding = (
        any(column in features_df.columns for column in ("Frota", "Tipo"))
        or "Tag_freq" not in features_df.columns
    )
    if not needs_encoding or encoder_path is None:
        return features_df, False
    encoder = load_encoder(encoder_path)
    return encoder.transform(features_df), True


def score_features(
    features_df: pd.DataFrame,
    artifact: dict[str, Any],
    allow_missing_features: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gera score e predicao a partir de um dataframe de features."""
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    aligned_features, missing_columns, extra_columns = align_feature_schema(
        features_df,
        feature_columns,
        allow_missing=allow_missing_features,
    )
    model_matrix = prepare_model_matrix(aligned_features, feature_columns)
    scores = predict_scores(artifact["model"], model_matrix)

    passthrough_cols = [column for column in ("Id", "Tag", "Fim") if column in features_df.columns]
    if passthrough_cols:
        scored = features_df[passthrough_cols].copy()
    else:
        scored = pd.DataFrame(index=features_df.index)
    scored["score"] = scores
    scored["threshold"] = threshold
    scored["prediction"] = (scored["score"] >= threshold).astype(int)

    metadata = {
        "rows": int(len(features_df)),
        "feature_count": len(feature_columns),
        "missing_feature_columns": missing_columns,
        "extra_columns_ignored": extra_columns,
    }
    return scored, metadata


def _context_series(features_df: pd.DataFrame, column: str) -> pd.Series:
    """Retorna coluna de contexto quando existir, senao valor padrao."""
    if column in features_df.columns:
        return features_df[column].astype(str)
    return pd.Series(["desconhecido"] * len(features_df), index=features_df.index)


def _build_priority_reason(row: pd.Series) -> str:
    """Resume o principal sinal operacional disponivel para a Tag-dia."""
    if float(row.get("n_alertas_4h", 0.0) or 0.0) > 0:
        return "alertas recentes na janela de 4h"
    if float(row.get("n_alertas_24h", 0.0) or 0.0) > 0:
        return "historico recente de alertas em 24h"
    if float(row.get("dias_desde_ultimo_alerta", 9999.0) or 9999.0) <= 1.0:
        return "alerta critico observado no ultimo dia"
    if float(row.get("delta_duracao_ciclo_4h_24h", 0.0) or 0.0) > 0:
        return "aumento recente na duracao do ciclo"
    if float(row.get("Tag_freq", 0.0) or 0.0) >= 0.03:
        return "Tag com alta recorrencia operacional"
    return "maior score diario do modelo"


def _risk_band(score: float, threshold: float, rank: int) -> str:
    """Classifica a prioridade em linguagem operacional simples."""
    if score >= threshold:
        return "alto_acima_threshold"
    if rank <= 5:
        return "alto_por_ranking"
    if rank <= 10:
        return "medio_por_ranking"
    return "monitorar_por_ranking"


def _recommended_action(risk_segment: str) -> str:
    """Traduz faixa de risco em acao recomendada para a lista diaria."""
    if risk_segment in {"alto_acima_threshold", "alto_por_ranking"}:
        return "inspecionar primeiro e registrar achado operacional"
    if risk_segment == "medio_por_ranking":
        return "verificar na ronda diaria e acompanhar tendencia"
    return "monitorar no painel e reavaliar no proximo ciclo"


def build_daily_priority_ranking(
    features_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    top_k: int = DEFAULT_OPERATIONAL_TOP_K,
) -> pd.DataFrame:
    """Gera ranking operacional TopK Tag-dia a partir dos scores de inferencia."""
    required_context = {"Tag", "Fim"}
    missing_context = sorted(required_context - set(scored_df.columns))
    if missing_context:
        raise ValueError(f"Contexto operacional ausente para ranking: {missing_context}")

    frame = features_df.copy()
    frame["Tag"] = scored_df["Tag"].astype(str)
    frame["Fim"] = pd.to_datetime(scored_df["Fim"], errors="coerce", utc=True)
    frame["score"] = scored_df["score"].astype(float)
    frame["threshold"] = scored_df["threshold"].astype(float)
    frame["data"] = frame["Fim"].dt.date
    frame["Frota"] = (
        _context_series(frame, "Frota")
        if "Frota" in frame.columns
        else decode_one_hot_prefix(frame, "Frota").astype(str)
    )
    frame["Tipo"] = (
        _context_series(frame, "Tipo")
        if "Tipo" in frame.columns
        else decode_one_hot_prefix(frame, "Tipo").astype(str)
    )
    frame["turno"] = _context_series(frame, "turno")
    frame["motivo_principal"] = frame.apply(_build_priority_reason, axis=1)

    frame = frame.dropna(subset=["data", "Tag"])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "data",
                "rank",
                "Tag",
                "score",
                "Frota",
                "Tipo",
                "turno",
                "motivo_principal",
                "risco_segmento",
                "acao_recomendada",
            ]
        )

    idx = frame.groupby(["data", "Tag"], sort=False)["score"].idxmax()
    tag_day = frame.loc[idx].sort_values(["data", "score"], ascending=[True, False]).copy()
    tag_day["rank"] = tag_day.groupby("data").cumcount() + 1
    top = tag_day.loc[tag_day["rank"] <= top_k].copy()
    top["risco_segmento"] = [
        _risk_band(score, threshold, rank)
        for score, threshold, rank in zip(top["score"], top["threshold"], top["rank"], strict=False)
    ]
    top["acao_recomendada"] = top["risco_segmento"].map(_recommended_action)

    output_columns = [
        "data",
        "rank",
        "Tag",
        "score",
        "Frota",
        "Tipo",
        "turno",
        "motivo_principal",
        "risco_segmento",
        "acao_recomendada",
    ]
    top["score"] = top["score"].round(6)
    return top[output_columns].reset_index(drop=True)


def save_inference_output(scored_df: pd.DataFrame, output_path: Path) -> Path:
    """Persiste saida da inferencia em csv ou parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        scored_df.to_parquet(output_path, index=False)
        return output_path
    if suffix == ".csv":
        scored_df.to_csv(output_path, index=False)
        return output_path
    raise ValueError(f"Formato de saida nao suportado: {suffix}")


def save_daily_priority_ranking(priority_df: pd.DataFrame, output_path: Path) -> Path:
    """Persiste ranking operacional diario em CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    priority_df.to_csv(output_path, index=False)
    return output_path


def run_inference(
    input_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    priority_output_path: Path = DEFAULT_PRIORITY_OUTPUT_PATH,
    top_k: int = DEFAULT_OPERATIONAL_TOP_K,
    encoder_path: Path | None = DEFAULT_ENCODER_PATH,
    validate_values: bool = True,
    allow_missing_features: bool = False,
) -> dict[str, Any]:
    """Executa inferencia ponta a ponta usando artefato e dataset de entrada."""
    artifact = load_model_artifact(model_path=model_path)
    features_df = read_inference_input(input_path)

    if validate_values:
        features_df = validate_batch_values(features_df)
    features_df, encoder_applied = apply_categorical_encoder(features_df, encoder_path)

    scored_df, metadata = score_features(
        features_df,
        artifact,
        allow_missing_features=allow_missing_features,
    )
    metadata["encoder_applied"] = encoder_applied
    metadata["values_validated"] = validate_values
    output = save_inference_output(scored_df, output_path=output_path)
    priority_df = build_daily_priority_ranking(features_df, scored_df, top_k=top_k)
    priority_output = save_daily_priority_ranking(priority_df, priority_output_path)
    return {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "output_path": str(output),
        "priority_output_path": str(priority_output),
        "threshold": float(artifact["threshold"]),
        "priority_rows": int(len(priority_df)),
        "priority_days": int(priority_df["data"].nunique()) if len(priority_df) else 0,
        "priority_top_k": int(top_k),
        **metadata,
    }


def main() -> None:
    result = run_inference(input_path=DEFAULT_INPUT_PATH)
    print(f"[OK] Input inferencia: {result['input_path']}")
    print(f"[OK] Artefato: {result['model_path']}")
    print(f"[OK] Saida: {result['output_path']}")
    print(f"[OK] Ranking operacional: {result['priority_output_path']}")
    print(f"[OK] Linhas processadas: {result['rows']}")
    print(
        "[OK] TopK Tag-dia: "
        f"top_k={result['priority_top_k']} "
        f"dias={result['priority_days']} "
        f"linhas={result['priority_rows']}"
    )
    print(f"[OK] Threshold aplicado: {result['threshold']:.6f}")
    print(f"[OK] Encoder categorico aplicado: {result['encoder_applied']}")
    print(f"[OK] Features ausentes: {len(result['missing_feature_columns'])}")


if __name__ == "__main__":
    main()
