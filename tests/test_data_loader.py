from pathlib import Path

import pandas as pd
import pytest
from src.data_loader import (
    build_quality_report,
    run_ingestion_contract,
    standardize_datetime_columns,
    validate_required_columns,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tag": ["CAM-001", "CAM-001", "CAM-002"],
            "Frota": ["SUL-01", "SUL-01", "SUDE-03"],
            "Tipo": ["CAM", "CAM", "CAR"],
            "Inicio": [
                "2026-01-01 08:00:00",
                "2026-01-01 08:00:00",
                "2026-01-01 10:00:00",
            ],
            "Fim": [
                "2026-01-01 09:00:00",
                "2026-01-01 09:00:00",
                "2026-01-02 18:30:00",
            ],
        }
    )


def test_validate_required_columns_raises_when_missing() -> None:
    df = pd.DataFrame({"Tag": ["CAM-001"], "Inicio": ["2026-01-01 08:00:00"]})
    with pytest.raises(ValueError):
        validate_required_columns(df)


def test_standardize_datetime_columns_converts_to_utc() -> None:
    df = _sample_df()
    out = standardize_datetime_columns(df)
    assert str(out["Inicio"].dtype).startswith("datetime64[ns, UTC]")
    assert str(out["Fim"].dtype).startswith("datetime64[ns, UTC]")


def test_build_quality_report_counts_duplicates_and_outliers() -> None:
    df = standardize_datetime_columns(_sample_df())
    report = build_quality_report(df, max_cycle_hours=24)

    assert report["total_records"] == 3
    assert report["duplicate_full_rows"] == 1
    assert report["duplicate_tag_inicio_fim"] == 1
    assert report["duration_gt_max_cycle_hours_count"] == 1
    assert report["duration_negative_count"] == 0


def test_run_ingestion_contract_writes_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    input_path = raw_dir / "desenvolver_apontamentos.csv"
    _sample_df().to_csv(input_path, index=False)

    output_dir = tmp_path / "processed" / "labeled"
    result = run_ingestion_contract(raw_dir=raw_dir, output_dir=output_dir)

    assert Path(result["snapshot_path"]).exists()
    assert Path(result["quality_report_json"]).exists()
    assert Path(result["quality_report_md"]).exists()
    assert result["total_records"] == 3
