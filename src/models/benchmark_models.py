"""Compatibilidade: benchmark simples substituido por selecao robusta."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.model_selection import (
    BASELINE_MODEL_NAME,
    OFFICIAL_CANDIDATE_NAMES,
    run_model_selection_pipeline,
    select_model,
)


def build_candidate_models() -> tuple[str, ...]:
    """Retorna as familias oficiais da selecao robusta."""
    return OFFICIAL_CANDIDATE_NAMES


def select_winner(summary: pd.DataFrame) -> str:
    """Compatibilidade para chamadas antigas: usa a regra oficial atual."""
    frame = summary.copy()
    if "eligible_for_selection" not in frame.columns:
        frame["eligible_for_selection"] = frame["model_name"] != BASELINE_MODEL_NAME
    return select_model(frame)


def run_benchmark_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias legado para o pipeline oficial de selecao robusta."""
    return run_model_selection_pipeline(*args, **kwargs)


def main() -> None:
    result = run_benchmark_pipeline()
    print("[OK] Alias benchmark executou selecao robusta")
    print(f"[OK] Modelo selecionado: {result['selected_model_name']}")
    print(f"[OK] Relatorio JSON: {result['json_path']}")
    print(f"[OK] Artefato selecionado: {result['artifact_path']}")


if __name__ == "__main__":
    main()
