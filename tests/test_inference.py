from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.inference import align_feature_schema, build_daily_priority_ranking, run_inference


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
    priority_output_path = tmp_path / "daily_priority_top15.csv"

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
            "turno": ["manha", "tarde"],
            "feature_a": [0.3, 0.9],
            "Frota_X": [True, False],
            "Frota_Y": [False, True],
            "Tipo_Caminhao": [True, True],
        }
    ).to_parquet(input_path, index=False)

    result = run_inference(
        input_path=input_path,
        model_path=artifact_path,
        output_path=output_path,
        priority_output_path=priority_output_path,
    )

    scored = pd.read_parquet(output_path)
    priority = pd.read_csv(priority_output_path)
    assert result["rows"] == 2
    assert result["priority_rows"] == 2
    assert result["missing_feature_columns"] == ["feature_b"]
    assert list(scored["prediction"]) == [0, 1]
    assert list(priority["Tag"]) == ["A", "B"]
    assert set(
        [
            "data",
            "rank",
            "Tag",
            "score",
            "Frota",
            "Tipo",
            "turno",
            "motivo_principal",
            "risco_segmento",
            "acao_recomendada",
        ]
    ) <= set(priority.columns)


def test_build_daily_priority_ranking_keeps_best_tag_day_and_top_k() -> None:
    features = pd.DataFrame(
        {
            "Tag": ["A", "A", "B", "C"],
            "Fim": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 10:00",
                    "2026-01-01 09:00",
                    "2026-01-01 11:00",
                ],
                utc=True,
            ),
            "turno": ["manha", "manha", "manha", "tarde"],
            "n_alertas_4h": [0, 2, 0, 0],
            "n_alertas_24h": [0, 2, 1, 0],
            "Frota_793-D": [True, True, False, True],
            "Frota_LeTourneau": [False, False, True, False],
            "Tipo_Caminhao": [True, True, False, True],
            "Tipo_Escavadeira": [False, False, True, False],
        }
    )
    scored = pd.DataFrame(
        {
            "Tag": ["A", "A", "B", "C"],
            "Fim": features["Fim"],
            "score": [0.3, 0.9, 0.7, 0.6],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
    )

    ranking = build_daily_priority_ranking(features, scored, top_k=2)

    assert ranking["Tag"].tolist() == ["A", "B"]
    assert ranking["rank"].tolist() == [1, 2]
    assert ranking.loc[0, "score"] == 0.9
    assert ranking.loc[0, "motivo_principal"] == "alertas recentes na janela de 4h"
    assert ranking.loc[0, "Frota"] == "793-D"
    assert ranking.loc[1, "Tipo"] == "Escavadeira"
