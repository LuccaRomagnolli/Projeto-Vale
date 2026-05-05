"""Inferencia operacional com contrato de schema para artefatos treinados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.model_selection import SELECTED_MODEL_PATH
from src.models.train_model import predict_scores, prepare_model_matrix
from src.utils.config import REPORTS_DIR

DEFAULT_MODEL_PATH = SELECTED_MODEL_PATH
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "inference_scores.parquet"
SUPPORTED_EXTENSIONS = {".parquet", ".csv"}


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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Alinha colunas para inferencia, preenchendo ausentes com zero."""
    out = df.copy()
    missing_columns = [column for column in feature_columns if column not in out.columns]
    for column in missing_columns:
        out[column] = 0.0
    extra_columns = [column for column in out.columns if column not in feature_columns]
    aligned = out[feature_columns]
    return aligned, missing_columns, extra_columns


def score_features(
    features_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gera score e predicao a partir de um dataframe de features."""
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    aligned_features, missing_columns, extra_columns = align_feature_schema(
        features_df,
        feature_columns,
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


def run_inference(
    input_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Executa inferencia ponta a ponta usando artefato e dataset de entrada."""
    artifact = load_model_artifact(model_path=model_path)
    features_df = read_inference_input(input_path)
    scored_df, metadata = score_features(features_df, artifact)
    output = save_inference_output(scored_df, output_path=output_path)
    return {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "output_path": str(output),
        "threshold": float(artifact["threshold"]),
        **metadata,
    }


def main() -> None:
    input_path = Path("data/processed/features/splits/features_test.parquet")
    result = run_inference(input_path=input_path)
    print(f"[OK] Input inferencia: {result['input_path']}")
    print(f"[OK] Artefato: {result['model_path']}")
    print(f"[OK] Saida: {result['output_path']}")
    print(f"[OK] Linhas processadas: {result['rows']}")
    print(f"[OK] Threshold aplicado: {result['threshold']:.6f}")
    print(f"[OK] Features ausentes preenchidas: {len(result['missing_feature_columns'])}")


if __name__ == "__main__":
    main()
