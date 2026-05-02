"""Gate de estabilidade temporal para promocao de modelo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import REPORTS_DIR

BACKTEST_REPORT_CSV = REPORTS_DIR / "hist_gbdt_backtest_report.csv"
DEFAULT_MAX_RECALL_STD = 0.03
DEFAULT_MAX_PRECISION_STD = 0.05
DEFAULT_MIN_FOLDS = 3


def run_stability_gate(
    backtest_path: Path = BACKTEST_REPORT_CSV,
    max_recall_std: float = DEFAULT_MAX_RECALL_STD,
    max_precision_std: float = DEFAULT_MAX_PRECISION_STD,
    min_folds: int = DEFAULT_MIN_FOLDS,
) -> dict[str, float | int | bool]:
    """Valida estabilidade minima entre folds temporais."""
    if not backtest_path.exists():
        raise FileNotFoundError(
            f"Backtesting ausente: {backtest_path}. Execute make tune-hist-gbdt antes do gate."
        )
    df = pd.read_csv(backtest_path)
    if len(df) < min_folds:
        raise ValueError(f"Folds insuficientes: {len(df)} < {min_folds}")

    recall_std = float(df["test_recall"].std(ddof=0))
    precision_std = float(df["test_precision"].std(ddof=0))
    passed = recall_std <= max_recall_std and precision_std <= max_precision_std
    return {
        "passed": passed,
        "folds": int(len(df)),
        "recall_std": recall_std,
        "precision_std": precision_std,
        "max_recall_std": float(max_recall_std),
        "max_precision_std": float(max_precision_std),
    }


def main() -> None:
    result = run_stability_gate()
    print(
        "[OK] Stability gate: "
        f"folds={result['folds']} "
        f"recall_std={result['recall_std']:.4f}/{result['max_recall_std']:.4f} "
        f"precision_std={result['precision_std']:.4f}/{result['max_precision_std']:.4f}"
    )
    if not result["passed"]:
        raise SystemExit("[ERROR] Stability gate failed. Modelo nao apto para promocao.")


if __name__ == "__main__":
    main()
