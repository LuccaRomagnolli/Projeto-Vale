"""Ablacao controlada do vazamento nos encodings categoricos.

Quantifica quanto do desempenho reportado vinha de vazamento temporal, em vez
de deixar a questao em aberto. Dois bracos partem do MESMO dataset de features
-- ja com as features de historico de alertas reparadas -- e diferem apenas em
como os encodings categoricos e o split temporal sao construidos:

    braco `com_vazamento`  encoding global antes do split, sem embargo
                           (o comportamento anterior do projeto)
    braco `sem_vazamento`  encoder ajustado apenas no treino, com embargo
                           (o comportamento atual)

O estimador e os hiperparametros sao identicos nos dois bracos, para que a
busca do Optuna nao entre como fator de confusao, e sao lidos do relatorio de
selecao vigente. O conjunto de teste e verificado como identico entre os bracos
-- o embargo corta apenas a cauda de treino e validacao -- de modo que as
metricas sao diretamente comparaveis.

Limitacao registrada: os hiperparametros usados foram selecionados sobre os
dados sem vazamento, o que desfavorece levemente o braco com vazamento. O
otimismo real pode ser um pouco maior que o medido aqui.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.operational_scorecard import (
    PRIMARY_TOP_K,
    build_scored_frame,
    compute_daily_topk_metrics,
)
from src.features.build_features import add_categorical_encodings
from src.features.encoders import CategoricalEncoder
from src.models.model_selection import (
    DEFAULT_MIN_RECALL,
    SELECTION_REPORT_JSON,
    build_candidate_model,
)
from src.models.train_model import predict_scores, prepare_model_matrix, select_feature_columns
from src.models.validation import (
    LABEL_HORIZON_HOURS,
    TARGET_COL,
    TIME_COL,
    choose_threshold_for_recall,
    compute_binary_metrics,
    temporal_train_val_test_split,
)
from src.utils.config import FEATURES_DATASET_PATH, REPORTS_MODEL_SELECTION_DIR
from src.utils.logging_config import get_logger, setup_logging
from src.utils.metadata import to_repo_relative_path

logger = get_logger(__name__)

ABLATION_REPORT_PATH = REPORTS_MODEL_SELECTION_DIR / "leakage_ablation_report.json"


def load_selected_configuration(report_path: Path = SELECTION_REPORT_JSON) -> tuple[str, dict]:
    """Le familia e hiperparametros do modelo promovido vigente."""
    if not report_path.exists():
        raise FileNotFoundError(
            f"Relatorio de selecao nao encontrado: {report_path}. "
            "Execute `python tasks.py model-selection` antes da ablacao."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_model") or {}
    model_name = selected.get("model_name")
    if not model_name:
        raise ValueError(f"Relatorio de selecao sem `selected_model.model_name`: {report_path}")
    params = selected.get("best_params") or "{}"
    return str(model_name), json.loads(params) if isinstance(params, str) else dict(params)


def evaluate_arm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    params: dict[str, Any],
    min_recall: float = DEFAULT_MIN_RECALL,
) -> dict[str, Any]:
    """Treina o estimador fixo e devolve as metricas operacionais do teste."""
    feature_columns = select_feature_columns(train)
    model = build_candidate_model(model_name, params)
    model.fit(prepare_model_matrix(train, feature_columns), train[TARGET_COL].astype(int))

    val_scores = predict_scores(model, prepare_model_matrix(val, feature_columns))
    test_scores = predict_scores(model, prepare_model_matrix(test, feature_columns))
    threshold = choose_threshold_for_recall(val[TARGET_COL], val_scores, min_recall=min_recall)

    scored = build_scored_frame(test, test_scores, threshold, "test")
    topk = compute_daily_topk_metrics(scored, (PRIMARY_TOP_K,), "test").iloc[0]
    binary = compute_binary_metrics(test[TARGET_COL], test_scores, threshold)

    return {
        "rows_train": int(len(train)),
        "rows_val": int(len(val)),
        "rows_test": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "threshold": float(threshold),
        f"test_top{PRIMARY_TOP_K}_precision_at_k": float(topk["precision_at_k"]),
        f"test_top{PRIMARY_TOP_K}_recall_at_k": float(topk["recall_at_k"]),
        f"test_top{PRIMARY_TOP_K}_lift_vs_random": float(topk["lift_vs_random"]),
        "test_auc_pr": float(binary["auc_pr"]),
    }


def run_leakage_ablation(
    features_path: Path = FEATURES_DATASET_PATH,
    report_path: Path = SELECTION_REPORT_JSON,
    output_path: Path = ABLATION_REPORT_PATH,
    embargo_hours: float = LABEL_HORIZON_HOURS,
) -> dict[str, Any]:
    """Executa os dois bracos e persiste a decomposicao."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Dataset de features nao encontrado: {features_path}. "
            "Execute `python tasks.py features` antes da ablacao."
        )
    model_name, params = load_selected_configuration(report_path)
    raw = pd.read_parquet(features_path)

    # Braco com vazamento: encoding global sobre o dataset inteiro, sem embargo.
    leaky_train, leaky_val, leaky_test, _ = temporal_train_val_test_split(
        add_categorical_encodings(raw.copy()), embargo_hours=0
    )
    with_leakage = evaluate_arm(leaky_train, leaky_val, leaky_test, model_name, params)

    # Braco sem vazamento: embargo e encoder ajustado apenas no treino.
    clean_train, clean_val, clean_test, _ = temporal_train_val_test_split(
        raw.copy(), embargo_hours=embargo_hours
    )
    encoder = CategoricalEncoder()
    clean_train = encoder.fit_transform_train(clean_train)
    clean_val = encoder.transform(clean_val)
    clean_test = encoder.transform(clean_test)
    without_leakage = evaluate_arm(clean_train, clean_val, clean_test, model_name, params)

    # A comparabilidade depende de o teste ser o mesmo nos dois bracos.
    same_test_set = bool(
        len(leaky_test) == len(clean_test)
        and leaky_test[TIME_COL].min() == clean_test[TIME_COL].min()
        and leaky_test[TIME_COL].max() == clean_test[TIME_COL].max()
    )
    if not same_test_set:
        raise ValueError(
            "Conjuntos de teste divergem entre os bracos; a comparacao nao seria valida."
        )

    compared = [
        f"test_top{PRIMARY_TOP_K}_precision_at_k",
        f"test_top{PRIMARY_TOP_K}_recall_at_k",
        f"test_top{PRIMARY_TOP_K}_lift_vs_random",
        "test_auc_pr",
    ]
    payload: dict[str, Any] = {
        "objetivo": (
            "Medir quanto do desempenho vinha de vazamento nos encodings categoricos, "
            "com estimador e hiperparametros fixos e o mesmo conjunto de teste."
        ),
        "model_name": model_name,
        "params": params,
        "embargo_hours": float(embargo_hours),
        "same_test_set": same_test_set,
        "com_vazamento": with_leakage,
        "sem_vazamento": without_leakage,
        "custo_do_fix": {key: without_leakage[key] - with_leakage[key] for key in compared},
        "limitacao": (
            "Os hiperparametros foram selecionados sobre os dados sem vazamento, o que "
            "desfavorece levemente o braco com vazamento. O otimismo real pode ser "
            "um pouco maior que o medido."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["report_path"] = str(output_path)
    return payload


def main() -> None:
    setup_logging()
    result = run_leakage_ablation()
    precision_key = f"test_top{PRIMARY_TOP_K}_precision_at_k"
    logger.info(f"Modelo fixado: {result['model_name']}")
    logger.info(f"Conjunto de teste identico entre bracos: {result['same_test_set']}")
    logger.info(
        f"Precision@{PRIMARY_TOP_K} com vazamento: " f"{result['com_vazamento'][precision_key]:.6f}"
    )
    logger.info(
        f"Precision@{PRIMARY_TOP_K} sem vazamento: " f"{result['sem_vazamento'][precision_key]:.6f}"
    )
    logger.info(f"Custo do fix: {result['custo_do_fix'][precision_key]:+.6f}")
    logger.info(f"Relatorio: {to_repo_relative_path(Path(result['report_path']))}")


if __name__ == "__main__":
    main()
