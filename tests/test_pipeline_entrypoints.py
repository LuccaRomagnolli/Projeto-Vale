from src import inference
from src.data import make_dataset
from src.evaluation import evaluate_model, segment_analysis
from src.features import build_features
from src.models import model_selection, train_baseline, train_model


def test_make_dataset_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        make_dataset.pd, "read_parquet", lambda _: make_dataset.pd.DataFrame({"Tag": []})
    )
    monkeypatch.setattr(
        make_dataset,
        "run_ingestion_contract",
        lambda: {
            "source_path": "/tmp/input.parquet",
            "snapshot_path": "/tmp/output.parquet",
            "quality_report_json": "/tmp/report.json",
            "quality_report_md": "/tmp/report.md",
            "total_records": 10,
        },
    )
    monkeypatch.setattr(
        make_dataset,
        "run_labeling_pipeline",
        lambda **_: {
            "rules_path": "/tmp/rules.xlsx",
            "events_source": "/tmp/events/",
            "critical_events_path": "/tmp/critical_events.parquet",
            "labeled_path": "/tmp/apontamentos_labeled.parquet",
            "labeling_report_json": "/tmp/labeling_report.json",
            "labeling_report_md": "/tmp/labeling_report.md",
            "files_processed": 6,
            "rows_critical_events_final": 20,
            "target_4h_positives": 1,
            "total_rows": 10,
        },
    )
    make_dataset.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_evaluate_model_main_runs(capsys, monkeypatch) -> None:
    # Atribuicao direta de atributo de modulo nao e desfeita ao fim do teste:
    # o stub ficava valendo para o resto da sessao, e qualquer teste posterior
    # que importasse `run_operational_evaluation` receberia o falso. Era o unico
    # caso assim no projeto -- todos os outros usam monkeypatch.
    monkeypatch.setattr(
        evaluate_model,
        "run_operational_evaluation",
        lambda: {
            "json_path": "/tmp/operational_metrics_report.json",
            "budget_metrics_csv": "/tmp/budget.csv",
            "daily_topk_metrics_csv": "/tmp/topk.csv",
            "report": {
                "model_name": "test_model",
                "threshold": 0.2,
                "threshold_metrics": [{"split": "test", "lift_vs_random": 1.5}],
                "test_daily_topk_metrics": [
                    {
                        "top_k_tags_per_day": 3,
                        "precision_at_k": 0.5,
                        "recall_at_k": 0.4,
                        "lift_vs_random": 2.0,
                    }
                ],
            },
        },
    )
    evaluate_model.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_build_features_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        build_features,
        "run_feature_pipeline",
        lambda: {
            "dataset_path": "/tmp/features.parquet",
            "report_path": "/tmp/feature_report.json",
            "rows": 10,
            "columns": 5,
            "target_4h_positives": 2,
        },
    )
    build_features.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_train_baseline_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        train_baseline,
        "run_baseline_pipeline",
        lambda: {
            "split_paths": {
                "train": "/tmp/train.parquet",
                "val": "/tmp/val.parquet",
                "test": "/tmp/test.parquet",
            },
            "artifact_path": "/tmp/baseline.joblib",
            "report_path": "/tmp/baseline_report.json",
            "scores_path": "/tmp/baseline_scores.parquet",
            "metrics": {
                "threshold": 0.2,
                "test": {
                    "recall": 0.8,
                    "auc_pr": 0.3,
                },
            },
        },
    )
    train_baseline.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_train_model_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        train_model,
        "run_model_pipeline",
        lambda: {
            "model_library": "test.Model",
            "feature_count": 3,
            "artifact_path": "/tmp/model.joblib",
            "report_path": "/tmp/model_report.json",
            "metrics": {
                "threshold": 0.4,
                "test": {
                    "recall": 0.7,
                    "auc_pr": 0.5,
                },
            },
        },
    )
    train_model.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_model_selection_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        model_selection,
        "run_model_selection_pipeline",
        lambda: {
            "official_candidates": [
                "lightgbm_optuna",
                "xgboost_optuna",
                "hist_gbdt_optuna",
            ],
            "diagnostic_baseline": "logistic_regression_baseline",
            "trials_per_candidate": 30,
            "feature_count": 5,
            "backtest_folds": 3,
            "artifact_path": "/tmp/model_selected.joblib",
            "json_path": "/tmp/model_selection_report.json",
            "selected_model_name": "hist_gbdt_optuna",
            "selected_model": {
                "val_auc_pr": 0.4,
                "test_recall": 0.8,
                "test_auc_pr": 0.3,
            },
        },
    )
    model_selection.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_segment_analysis_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        segment_analysis,
        "run_segment_analysis",
        lambda: {
            "json_path": "/tmp/segment.json",
            "threshold_metrics_csv": "/tmp/threshold.csv",
            "topk_metrics_csv": "/tmp/topk.csv",
            "tag_hotspots_csv": "/tmp/hotspots.csv",
            "report": {
                "strongest_top15_segments": [
                    {
                        "segment_col": "Tipo",
                        "segment_value": "Caminhao",
                        "precision_at_k": 0.7,
                        "recall_at_k": 0.8,
                    }
                ],
                "weakest_top15_segments": [
                    {
                        "segment_col": "Tipo",
                        "segment_value": "Escavadeira",
                        "precision_at_k": 0.4,
                        "recall_at_k": 0.2,
                    }
                ],
            },
        },
    )
    segment_analysis.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out


def test_inference_main_runs(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        inference,
        "run_inference",
        lambda **_: {
            "input_path": "/tmp/features_test.parquet",
            "model_path": "/tmp/model.joblib",
            "output_path": "/tmp/inference_scores.parquet",
            "priority_output_path": "/tmp/daily_priority_top15.csv",
            "rows": 10,
            "priority_rows": 5,
            "priority_days": 1,
            "priority_top_k": 15,
            "threshold": 0.3,
            "missing_feature_columns": [],
            "encoder_applied": True,
        },
    )
    inference.main()
    captured = capsys.readouterr()
    assert "[OK]" in captured.out
