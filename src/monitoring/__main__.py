"""Entradas de linha de comando do monitoramento operacional.

python -m src.monitoring baseline   constroi o perfil de referencia
python -m src.monitoring check      compara o ultimo lote com a referencia
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.model_selection import SELECTED_MODEL_PATH
from src.models.train_model import load_splits, predict_scores, prepare_model_matrix
from src.monitoring.drift import (
    DRIFT_REPORT_PATH,
    REFERENCE_PROFILE_PATH,
    build_reference_profile,
    compute_drift,
    load_reference_profile,
    save_reference_profile,
    should_retrain,
)
from src.utils.config import REPORTS_INFERENCE_DIR, SPLIT_DIR
from src.utils.metadata import to_repo_relative_path

DEFAULT_SCORES_PATH = REPORTS_INFERENCE_DIR / "inference_scores.parquet"


def _load_artifact(model_path: Path = SELECTED_MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefato promovido ausente: {model_path}. Execute `python tasks.py train`."
        )
    return joblib.load(model_path)


def run_baseline(
    split_dir: Path = SPLIT_DIR,
    model_path: Path = SELECTED_MODEL_PATH,
    output_path: Path = REFERENCE_PROFILE_PATH,
) -> dict[str, Any]:
    """Congela o perfil de referencia a partir do treino do modelo promovido."""
    artifact = _load_artifact(model_path)
    train, _, _ = load_splits(split_dir)
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    scores = predict_scores(artifact["model"], prepare_model_matrix(train, feature_columns))
    alert_rate = float((scores >= threshold).mean())

    profile = build_reference_profile(
        train,
        feature_columns=feature_columns,
        scores=scores,
        alert_rate=alert_rate,
    )
    path = save_reference_profile(profile, output_path)
    return {
        "rows": profile.rows,
        "features_perfiladas": len(profile.feature_reference),
        "alert_rate": alert_rate,
        "profile_path": str(path),
    }


def run_check(
    scores_path: Path = DEFAULT_SCORES_PATH,
    profile_path: Path = REFERENCE_PROFILE_PATH,
    split_dir: Path = SPLIT_DIR,
    model_path: Path = SELECTED_MODEL_PATH,
    output_path: Path = DRIFT_REPORT_PATH,
) -> dict[str, Any]:
    """Compara o lote mais recente pontuado contra a referencia."""
    profile = load_reference_profile(profile_path)
    artifact = _load_artifact(model_path)
    feature_columns = list(artifact["feature_columns"])

    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores de inferencia ausentes: {scores_path}. Execute `python tasks.py infer`."
        )
    scored = pd.read_parquet(scores_path)

    # As features do lote vem do mesmo conjunto usado na inferencia.
    _, _, test = load_splits(split_dir)
    batch = test

    observed_alert_rate = (
        float(scored["prediction"].mean()) if "prediction" in scored.columns else None
    )
    drift = compute_drift(
        batch,
        profile,
        scores=scored["score"] if "score" in scored.columns else None,
        observed_alert_rate=observed_alert_rate,
    )
    drift["feature_count"] = len(feature_columns)
    drift.update(should_retrain(drift))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(drift, indent=2, ensure_ascii=False), encoding="utf-8")
    drift["report_path"] = str(output_path)
    return drift


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monitoramento operacional de drift.")
    parser.add_argument("acao", choices=["baseline", "check"])
    args = parser.parse_args(argv)

    if args.acao == "baseline":
        result = run_baseline()
        print(f"[OK] Linhas de referencia: {result['rows']}")
        print(f"[OK] Features perfiladas: {result['features_perfiladas']}")
        print(f"[OK] Taxa de alerta esperada: {result['alert_rate']:.6f}")
        print(f"[OK] Perfil salvo em: {to_repo_relative_path(Path(result['profile_path']))}")
        return

    drift = run_check()
    print(f"[OK] Linhas avaliadas: {drift['rows']}")
    print(f"[OK] PSI do score: {drift['score_psi']:.6f} ({drift['score_faixa']})")
    volume = drift["volume_de_alertas"]
    if volume.get("observado") is not None:
        print(
            f"[OK] Volume de alertas: esperado {volume['esperado']:.4f}, "
            f"observado {volume['observado']:.4f}"
        )
    significativos = drift["features_com_drift_significativo"]
    print(f"[OK] Features com drift significativo: {len(significativos)}")
    for feature in significativos[:5]:
        print(f"[OK]   - {feature}")
    print(f"[OK] Relatorio: {to_repo_relative_path(Path(drift['report_path']))}")
    if drift["retreinar"]:
        for reason in drift["motivos"]:
            print(f"[ERROR] {reason}")
        raise SystemExit("[ERROR] Gatilho de retreino acionado.")


if __name__ == "__main__":
    main()
