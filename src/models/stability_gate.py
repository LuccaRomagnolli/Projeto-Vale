"""Gate de estabilidade temporal para promocao de modelo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.operational_scorecard import PRIMARY_TOP_K
from src.utils.config import REPORTS_MODEL_SELECTION_DIR
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

BACKTEST_REPORT_CSV = REPORTS_MODEL_SELECTION_DIR / "model_selection_backtest_report.csv"
DEFAULT_MAX_RECALL_STD = 0.03
DEFAULT_MAX_PRECISION_STD = 0.05
DEFAULT_MIN_FOLDS = 3


def _resolve_stability_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    operational_recall = f"test_top{PRIMARY_TOP_K}_recall_at_k"
    operational_precision = f"test_top{PRIMARY_TOP_K}_precision_at_k"
    if {operational_recall, operational_precision}.issubset(df.columns):
        return operational_recall, operational_precision, f"Top{PRIMARY_TOP_K} Tag-dia"
    return "test_recall", "test_precision", "ciclo-a-ciclo"


def run_stability_gate(
    backtest_path: Path = BACKTEST_REPORT_CSV,
    max_recall_std: float = DEFAULT_MAX_RECALL_STD,
    max_precision_std: float = DEFAULT_MAX_PRECISION_STD,
    min_folds: int = DEFAULT_MIN_FOLDS,
) -> dict[str, Any]:
    """Valida estabilidade minima entre folds temporais."""
    if not backtest_path.exists():
        raise FileNotFoundError(
            f"Backtesting ausente: {backtest_path}. Execute make model-selection antes do gate."
        )
    df = pd.read_csv(backtest_path)
    if len(df) < min_folds:
        raise ValueError(f"Folds insuficientes: {len(df)} < {min_folds}")

    recall_col, precision_col, metric_family = _resolve_stability_columns(df)
    recall_std = float(df[recall_col].std(ddof=0))
    precision_std = float(df[precision_col].std(ddof=0))
    passed = recall_std <= max_recall_std and precision_std <= max_precision_std
    return {
        "passed": passed,
        "folds": int(len(df)),
        "metric_family": metric_family,
        "recall_column": recall_col,
        "precision_column": precision_col,
        "recall_std": recall_std,
        "precision_std": precision_std,
        "max_recall_std": float(max_recall_std),
        "max_precision_std": float(max_precision_std),
    }


def main() -> None:
    setup_logging()
    result = run_stability_gate()
    logger.info(
        "Stability gate: "
        f"folds={result['folds']} "
        f"metric={result['metric_family']} "
        f"recall_std={result['recall_std']:.4f}/{result['max_recall_std']:.4f} "
        f"precision_std={result['precision_std']:.4f}/{result['max_precision_std']:.4f}"
    )
    if not result["passed"]:
        raise SystemExit("[ERROR] Stability gate failed. Modelo nao apto para promocao.")


if __name__ == "__main__":
    main()
