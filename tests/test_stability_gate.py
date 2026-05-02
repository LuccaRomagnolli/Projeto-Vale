from pathlib import Path

import pandas as pd
from src.models.stability_gate import run_stability_gate


def test_stability_gate_prefers_operational_topk_columns(tmp_path: Path) -> None:
    path = tmp_path / "backtest.csv"
    pd.DataFrame(
        {
            "test_recall": [0.2, 0.8, 0.1],
            "test_precision": [0.1, 0.9, 0.2],
            "test_top15_recall_at_k": [0.72, 0.73, 0.74],
            "test_top15_precision_at_k": [0.62, 0.64, 0.63],
        }
    ).to_csv(path, index=False)

    result = run_stability_gate(path, max_recall_std=0.03, max_precision_std=0.05)

    assert result["passed"] is True
    assert result["metric_family"] == "Top15 Tag-dia"


def test_stability_gate_falls_back_to_cycle_metrics(tmp_path: Path) -> None:
    path = tmp_path / "backtest.csv"
    pd.DataFrame(
        {
            "test_recall": [0.78, 0.79, 0.80],
            "test_precision": [0.24, 0.25, 0.26],
        }
    ).to_csv(path, index=False)

    result = run_stability_gate(path, max_recall_std=0.03, max_precision_std=0.05)

    assert result["passed"] is True
    assert result["metric_family"] == "ciclo-a-ciclo"
