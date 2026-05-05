import pandas as pd
from src.models.benchmark_models import build_candidate_models, select_winner


def test_select_winner_uses_operational_validation_scorecard() -> None:
    summary = pd.DataFrame(
        {
            "model_name": ["a", "b"],
            "val_top15_recall_at_k": [0.8, 0.7],
            "val_top15_precision_at_k": [0.6, 0.9],
            "val_top15_lift_vs_random": [1.8, 2.5],
            "val_auc_pr": [0.2, 0.4],
            "val_precision": [0.1, 0.9],
            "val_f1": [0.8, 0.2],
        }
    )
    assert select_winner(summary) == "a"


def test_build_candidate_models_includes_lightgbm_when_installed() -> None:
    candidates = build_candidate_models()

    assert "lightgbm_optuna" in candidates
