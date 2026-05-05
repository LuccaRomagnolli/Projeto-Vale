"""Compatibilidade: tuning especifico substituido por selecao robusta."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.model_selection import (
    build_threshold_curve,
    make_backtest_folds,
    run_model_selection_pipeline,
    select_model,
)

__all__ = [
    "build_threshold_curve",
    "make_backtest_folds",
    "run_tuning_pipeline",
    "select_best_candidate",
]


def select_best_candidate(summary: pd.DataFrame) -> pd.Series:
    """Compatibilidade para testes/chamadas antigas sobre uma tabela de candidatos."""
    frame = summary.rename(columns={"candidate": "model_name"}).copy()
    if "eligible_for_selection" not in frame.columns:
        frame["eligible_for_selection"] = True
    selected = select_model(frame)
    return summary.loc[frame["model_name"] == selected].iloc[0]


def run_tuning_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias legado para o pipeline oficial de selecao robusta."""
    return run_model_selection_pipeline(*args, **kwargs)


def main() -> None:
    result = run_tuning_pipeline()
    print("[OK] Alias de tuning executou selecao robusta")
    print(f"[OK] Modelo selecionado: {result['selected_model_name']}")
    print(f"[OK] Backtesting folds: {result['backtest_folds']}")
    print(f"[OK] Artefato selecionado: {result['artifact_path']}")
    print(f"[OK] Relatorio JSON: {result['json_path']}")


if __name__ == "__main__":
    main()
