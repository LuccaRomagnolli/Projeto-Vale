from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.inference import align_feature_schema, run_inference


class _FakeModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        base = x.iloc[:, 0].to_numpy(dtype=float)
        scores = np.clip(base, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


def test_align_feature_schema_adds_missing_columns() -> None:
    df = pd.DataFrame({"feature_a": [0.2, 0.8], "other": [1, 2]})
    aligned, missing, extra = align_feature_schema(df, ["feature_a", "feature_b"])

    assert missing == ["feature_b"]
    assert "other" in extra
    assert list(aligned.columns) == ["feature_a", "feature_b"]
    assert aligned["feature_b"].tolist() == [0.0, 0.0]


def test_run_inference_scores_and_writes_output(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.joblib"
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    joblib.dump(
        {
            "model": _FakeModel(),
            "feature_columns": ["feature_a", "feature_b"],
            "threshold": 0.5,
        },
        artifact_path,
    )

    pd.DataFrame(
        {
            "Id": [1, 2],
            "Tag": ["A", "B"],
            "Fim": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "feature_a": [0.3, 0.9],
        }
    ).to_parquet(input_path, index=False)

    result = run_inference(
        input_path=input_path,
        model_path=artifact_path,
        output_path=output_path,
    )

    scored = pd.read_parquet(output_path)
    assert result["rows"] == 2
    assert result["missing_feature_columns"] == ["feature_b"]
    assert list(scored["prediction"]) == [0, 1]
