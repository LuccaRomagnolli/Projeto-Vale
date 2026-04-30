from __future__ import annotations

import argparse
from pathlib import Path

from projeto_vale.baseline import FrequencyBaseline
from projeto_vale.evaluate import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_metrics_json,
    save_roc_pr_curves,
)
from projeto_vale.features import build_apontamentos_features
from projeto_vale.io import load_alarm_rules, read_apontamentos, read_telemetria_dir
from projeto_vale.rules import apply_critical_rules, extract_critical_alert_events
from projeto_vale.split import temporal_split
from projeto_vale.target import build_next_alert_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline baseline do Projeto Vale",
    )
    parser.add_argument(
        "--apontamentos",
        required=True,
        help="Arquivo de apontamentos (.parquet/.xlsx)",
    )
    parser.add_argument(
        "--telemetria-dir",
        required=True,
        help="Diretorio com arquivos de telemetria",
    )
    parser.add_argument(
        "--regras",
        required=True,
        help="Arquivo de regras de negocio (.xlsx)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Diretorio de saida processado",
    )
    parser.add_argument("--reports-dir", default="reports", help="Diretorio de relatarios")
    parser.add_argument("--horizon-hours", type=int, default=4, help="Janela de predicao do target")
    parser.add_argument("--critical-level", default="muito alto", help="Nivel critico da regra")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de classificacao")
    parser.add_argument(
        "--telemetry-max-rows",
        type=int,
        default=None,
        help="Limite opcional de linhas de telemetria para execucao rapida",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    apontamentos = read_apontamentos(args.apontamentos)
    telemetria = read_telemetria_dir(args.telemetria_dir)

    if args.telemetry_max_rows is not None and args.telemetry_max_rows > 0:
        telemetria = telemetria.sort_values("Data_Evento").head(args.telemetry_max_rows)

    regras = load_alarm_rules(args.regras)
    labeled_telemetria = apply_critical_rules(
        telemetria_df=telemetria,
        rules_df=regras,
        critical_level=args.critical_level,
    )
    alert_events = extract_critical_alert_events(labeled_telemetria)

    feats = build_apontamentos_features(apontamentos)
    feats["target_critical_4h"] = build_next_alert_target(
        apontamentos_df=feats,
        alert_events_df=alert_events,
        horizon_hours=args.horizon_hours,
        tag_col="Tag",
        event_time_col="Inicio",
    )

    train_df, val_df, test_df = temporal_split(
        feats,
        time_col="Inicio",
        train_size=0.70,
        val_size=0.15,
    )

    model = FrequencyBaseline(tag_col="Tag")
    model.fit(train_df, train_df["target_critical_4h"])

    val_prob = model.predict_proba(val_df)
    test_prob = model.predict_proba(test_df)

    val_metrics = compute_classification_metrics(
        y_true=val_df["target_critical_4h"],
        y_prob=val_prob,
        threshold=args.threshold,
    )
    test_metrics = compute_classification_metrics(
        y_true=test_df["target_critical_4h"],
        y_prob=test_prob,
        threshold=args.threshold,
    )

    metrics = {
        "horizon_hours": args.horizon_hours,
        "critical_level": args.critical_level,
        "threshold": args.threshold,
        "rows": {
            "train": int(train_df.shape[0]),
            "val": int(val_df.shape[0]),
            "test": int(test_df.shape[0]),
        },
        "val": val_metrics,
        "test": test_metrics,
    }

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_out = val_df.copy()
    val_out["prediction_proba"] = val_prob.to_numpy()
    val_out.to_parquet(processed_dir / "val.parquet", index=False)

    test_out = test_df.copy()
    test_out["prediction_proba"] = test_prob.to_numpy()
    test_out.to_parquet(processed_dir / "test.parquet", index=False)

    save_metrics_json(metrics, reports_dir / "metrics_baseline.json")
    save_roc_pr_curves(val_df["target_critical_4h"], val_prob, figures_dir)
    save_confusion_matrix_plot(
        y_true=val_df["target_critical_4h"],
        y_prob=val_prob,
        threshold=args.threshold,
        output_path=figures_dir / "confusion_matrix.png",
    )

    print("Pipeline executado com sucesso.")
    print(f"Processed: {processed_dir}")
    print(f"Reports: {reports_dir}")


if __name__ == "__main__":
    main()
