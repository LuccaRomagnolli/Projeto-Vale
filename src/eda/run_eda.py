"""Etapa 4: EDA orientada a decisao com saidas reproduziveis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import (
    FIGURES_DIR,
    LABELED_DATASET_PATH,
    REPORTS_EDA_DIR,
    REPORTS_MODEL_SELECTION_DIR,
    REPORTS_OPERATIONAL_DIR,
    REPORTS_SEGMENT_DIR,
)

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def load_labeled_dataset(dataset_path: Path = LABELED_DATASET_PATH) -> pd.DataFrame:
    """Carrega dataset rotulado usado na EDA."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset rotulado nao encontrado em "
            f"{dataset_path}. Execute `make label` antes de `make eda`."
        )
    return pd.read_parquet(dataset_path)


def enrich_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas auxiliares para analise exploratoria."""
    out = df.copy()
    out["Inicio"] = pd.to_datetime(out["Inicio"], errors="coerce", utc=True)
    out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce", utc=True)
    out["duracao_ciclo_min"] = (out["Fim"] - out["Inicio"]).dt.total_seconds() / 60.0
    out["hora_do_dia"] = out["Inicio"].dt.hour
    out["dia_da_semana"] = out["Inicio"].dt.dayofweek
    out["is_fim_de_semana"] = out["dia_da_semana"].isin([5, 6]).astype(int)
    return out


def build_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula sumario tecnico para suporte a decisao."""
    total_rows = int(len(df))
    positives = int(df["target_4h"].sum())
    positive_rate_pct = round((positives / total_rows) * 100.0, 6) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "total_tags": int(df["Tag"].nunique()),
        "total_frotas": int(df["Frota"].nunique()),
        "total_tipos": int(df["Tipo"].nunique()),
        "target_4h_positives": positives,
        "target_4h_positive_rate_pct": positive_rate_pct,
        "inicio_min": str(df["Inicio"].min()),
        "inicio_max": str(df["Inicio"].max()),
        "duracao_ciclo_min_mean": float(df["duracao_ciclo_min"].mean()),
        "duracao_ciclo_min_median": float(df["duracao_ciclo_min"].median()),
        "duracao_ciclo_min_p95": float(df["duracao_ciclo_min"].quantile(0.95)),
        "missing_pct_top5": (
            (df.isna().mean() * 100.0).sort_values(ascending=False).head(5).round(4).to_dict()
        ),
    }


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _target_rate_by_category(
    df: pd.DataFrame,
    column: str,
    top: int = 12,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Calcula volume e taxa positiva por categoria, filtrando grupos pequenos."""
    grouped = (
        df.groupby(column, dropna=False)
        .agg(ciclos=("target_4h", "size"), taxa_alerta=("target_4h", "mean"))
        .reset_index()
        .rename(columns={column: "categoria"})
    )
    grouped["categoria"] = grouped["categoria"].fillna("MISSING").astype(str)
    return grouped.loc[grouped["ciclos"].ge(min_rows)].nlargest(top, "taxa_alerta")


def _format_count_axis(ax: plt.Axes, axis: str = "y") -> None:
    ax.ticklabel_format(axis=axis, style="plain")
    ax.grid(axis=axis, alpha=0.25)


def generate_eda_figures(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Gera figuras da EDA para o relatorio."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1) Distribuicao target
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="target_4h", ax=ax, color="#3b82f6")
    ax.set_title("Distribuicao do target_4h")
    ax.set_xlabel("target_4h")
    ax.set_ylabel("Quantidade de registros")
    path = figures_dir / "eda_target_distribution.png"
    _save_figure(fig, path)
    generated.append(path)

    # 2) Ciclos por hora do dia
    fig, ax = plt.subplots(figsize=(9, 4))
    hour_counts = (
        df["hora_do_dia"].value_counts(dropna=True).sort_index().reindex(range(24), fill_value=0)
    )
    sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax, color="#10b981")
    ax.set_title("Ciclos por hora do dia")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Quantidade de ciclos")
    path = figures_dir / "eda_ciclos_por_hora.png"
    _save_figure(fig, path)
    generated.append(path)

    # 3) Top frotas
    top_frotas = df["Frota"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=top_frotas.index, y=top_frotas.values, ax=ax, color="#f59e0b")
    ax.set_title("Top 15 Frotas por volume de ciclos")
    ax.set_xlabel("Frota")
    ax.set_ylabel("Quantidade de ciclos")
    ax.tick_params(axis="x", rotation=45)
    path = figures_dir / "eda_top_frotas.png"
    _save_figure(fig, path)
    generated.append(path)

    # 4) Duracao de ciclo
    sample = df["duracao_ciclo_min"].dropna()
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(sample, bins=60, ax=ax, color="#6366f1")
    ax.set_title("Distribuicao da duracao de ciclo (min)")
    ax.set_xlabel("Duracao de ciclo (min)")
    ax.set_ylabel("Frequencia")
    path = figures_dir / "eda_duracao_ciclo_hist.png"
    _save_figure(fig, path)
    generated.append(path)

    # 5) Classe de atividade
    top_classes = df["Classe"].fillna("MISSING").value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(x=top_classes.index, y=top_classes.values, ax=ax, color="#ef4444")
    ax.set_title("Top classes de atividade")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de ciclos")
    ax.tick_params(axis="x", rotation=40)
    path = figures_dir / "eda_top_classes.png"
    _save_figure(fig, path)
    generated.append(path)

    # 6) Serie temporal: volume e prevalencia diaria
    daily = (
        df.assign(dia=df["Inicio"].dt.floor("D"))
        .groupby("dia")
        .agg(ciclos=("target_4h", "size"), taxa_alerta=("target_4h", "mean"))
        .reset_index()
        .sort_values("dia")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["dia"], daily["ciclos"], color="#2563eb", linewidth=1.8, label="Ciclos")
    ax.set_title("Volume diário de ciclos e prevalência de target_4h")
    ax.set_xlabel("Data")
    ax.set_ylabel("Ciclos")
    _format_count_axis(ax)
    ax2 = ax.twinx()
    ax2.plot(
        daily["dia"],
        daily["taxa_alerta"],
        color="#dc2626",
        linewidth=1.6,
        label="Taxa target_4h",
    )
    ax2.set_ylabel("Taxa target_4h")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
    path = figures_dir / "eda_daily_volume_target_rate.png"
    _save_figure(fig, path)
    generated.append(path)

    # 7) Heatmap de risco por dia da semana e hora
    heatmap = (
        df.pivot_table(
            index="dia_da_semana",
            columns="hora_do_dia",
            values="target_4h",
            aggfunc="mean",
            fill_value=0,
        )
        .reindex(index=range(7), columns=range(24), fill_value=0)
        .rename(index={0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"})
    )
    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="YlOrRd",
        cbar_kws={"label": "Taxa target_4h"},
        linewidths=0.2,
        linecolor="#ffffff",
    )
    ax.set_title("Taxa de target_4h por hora e dia da semana")
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Dia da semana")
    path = figures_dir / "eda_target_rate_heatmap_hora_dia.png"
    _save_figure(fig, path)
    generated.append(path)

    # 8) Tags com mais positivos
    positive_tags = (
        df.loc[df["target_4h"].eq(1), "Tag"]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .head(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    if positive_tags.empty:
        ax.text(0.5, 0.5, "Sem positivos target_4h", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=positive_tags.values, y=positive_tags.index, ax=ax, color="#dc2626")
        ax.set_xlabel("Ciclos positivos target_4h")
        ax.set_ylabel("Tag")
        _format_count_axis(ax, axis="x")
    ax.set_title("Top 15 Tags por volume de alertas antecipáveis")
    path = figures_dir / "eda_top_tags_target_positives.png"
    _save_figure(fig, path)
    generated.append(path)

    # 9) Taxa de alerta por frota
    frota_rate = _target_rate_by_category(df, "Frota", top=12, min_rows=20).sort_values(
        "taxa_alerta"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frota_rate, x="taxa_alerta", y="categoria", ax=ax, color="#0f766e")
    ax.set_title("Frotas com maior taxa de target_4h")
    ax.set_xlabel("Taxa target_4h")
    ax.set_ylabel("Frota")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    path = figures_dir / "eda_target_rate_by_frota.png"
    _save_figure(fig, path)
    generated.append(path)

    # 10) Taxa de alerta por tipo e classe
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, column, title in [
        (axes[0], "Tipo", "Taxa target_4h por Tipo"),
        (axes[1], "Classe", "Taxa target_4h por Classe"),
    ]:
        rate_table = _target_rate_by_category(df, column, top=10, min_rows=20).sort_values(
            "taxa_alerta"
        )
        sns.barplot(data=rate_table, x="taxa_alerta", y="categoria", ax=ax, color="#7c3aed")
        ax.set_title(title)
        ax.set_xlabel("Taxa target_4h")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    path = figures_dir / "eda_target_rate_by_tipo_classe.png"
    _save_figure(fig, path)
    generated.append(path)

    # 11) Duracao por classe
    duration_cols = ["Classe", "duracao_ciclo_min"]
    duration_sample = df[duration_cols].dropna()
    if len(duration_sample) > 50000:
        duration_sample = duration_sample.sample(50000, random_state=42)
    top_duration_classes = duration_sample["Classe"].value_counts().head(8).index
    duration_sample = duration_sample.loc[duration_sample["Classe"].isin(top_duration_classes)]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=duration_sample,
        x="Classe",
        y="duracao_ciclo_min",
        ax=ax,
        color="#38bdf8",
        showfliers=False,
    )
    ax.set_title("Duração de ciclo por classe de atividade")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Duração de ciclo (min)")
    ax.tick_params(axis="x", rotation=35)
    path = figures_dir / "eda_duracao_por_classe_boxplot.png"
    _save_figure(fig, path)
    generated.append(path)

    # 12) Tempo ate evento critico para positivos
    if "tte_horas" in df.columns:
        tte_positive = df.loc[df["target_4h"].eq(1), "tte_horas"].dropna()
    else:
        tte_positive = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    if tte_positive.empty:
        ax.text(0.5, 0.5, "Sem tte_horas para positivos", ha="center", va="center")
        ax.axis("off")
    else:
        sns.histplot(tte_positive.clip(lower=0, upper=4), bins=24, ax=ax, color="#f97316")
        ax.set_xlabel("Horas até evento crítico")
        ax.set_ylabel("Frequência")
        _format_count_axis(ax)
    ax.set_title("Distribuição de antecedência dos ciclos positivos")
    path = figures_dir / "eda_tte_horas_positivos_hist.png"
    _save_figure(fig, path)
    generated.append(path)

    # 13) Nulos por coluna
    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#64748b")
    ax.set_title("Percentual de nulos por coluna")
    ax.set_xlabel("% nulos")
    ax.set_ylabel("Coluna")
    path = figures_dir / "eda_missing_values.png"
    _save_figure(fig, path)
    generated.append(path)

    # 14) Heatmap de Correlação
    # Seleciona colunas numéricas, excluindo identificadores e colunas temporais brutas se possível
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols_to_drop = {"Id"}
    valid_cols = [c for c in numeric_cols if c not in cols_to_drop]
    if valid_cols:
        corr_matrix = df[valid_cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            ax=ax, 
            cmap="coolwarm", 
            center=0,
            annot=False, 
            cbar_kws={"label": "Spearman Correlation"},
            linewidths=0.5
        )
        ax.set_title("Heatmap de Correlação (Spearman) - Variáveis Numéricas")
        path = figures_dir / "eda_correlation_heatmap.png"
        _save_figure(fig, path)
        generated.append(path)

    return generated


def generate_project_artifact_figures(figures_dir: Path = FIGURES_DIR) -> list[Path]:
    """Gera figuras de modelagem/operação quando os artefatos já existem."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    selection_csv = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.csv"
    if selection_csv.exists():
        selection_df = pd.read_csv(selection_csv)
        candidates = selection_df.loc[selection_df["role"].eq("official_candidate")].copy()
        if not candidates.empty:
            metric_cols = [
                "test_top15_precision_at_k",
                "test_top15_recall_at_k",
                "test_top15_lift_vs_random",
            ]
            plot_df = candidates.melt(
                id_vars="model_name",
                value_vars=[col for col in metric_cols if col in candidates.columns],
                var_name="metric",
                value_name="value",
            )
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.barplot(data=plot_df, x="model_name", y="value", hue="metric", ax=ax)
            ax.set_title("Comparação dos candidatos oficiais no teste temporal")
            ax.set_xlabel("Modelo")
            ax.set_ylabel("Valor")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Métrica", fontsize=8)
            path = figures_dir / "model_selection_test_top15_metrics.png"
            _save_figure(fig, path)
            generated.append(path)

            auc_cols = ["test_auc_pr", "test_auc_roc"]
            auc_df = candidates.melt(
                id_vars="model_name",
                value_vars=[col for col in auc_cols if col in candidates.columns],
                var_name="metric",
                value_name="value",
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=auc_df, x="model_name", y="value", hue="metric", ax=ax)
            ax.set_title("AUC dos candidatos oficiais no teste temporal")
            ax.set_xlabel("Modelo")
            ax.set_ylabel("AUC")
            ax.set_ylim(0, min(1.0, max(auc_df["value"].max() * 1.15, 0.1)))
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Métrica", fontsize=8)
            path = figures_dir / "model_selection_test_auc.png"
            _save_figure(fig, path)
            generated.append(path)

    topk_csv = REPORTS_OPERATIONAL_DIR / "operational_daily_topk_metrics.csv"
    if topk_csv.exists():
        topk_df = pd.read_csv(topk_csv)
        test_topk = topk_df.loc[topk_df["split"].eq("test")].sort_values("top_k_tags_per_day")
        if not test_topk.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                test_topk["top_k_tags_per_day"],
                test_topk["precision_at_k"],
                marker="o",
                label="Precision@K",
                color="#2563eb",
            )
            ax.plot(
                test_topk["top_k_tags_per_day"],
                test_topk["recall_at_k"],
                marker="o",
                label="Recall@K",
                color="#dc2626",
            )
            ax.set_title("Curva operacional TopK Tag-dia no teste")
            ax.set_xlabel("Top K Tags por dia")
            ax.set_ylabel("Métrica")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.25)
            path = figures_dir / "operational_topk_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    budget_csv = REPORTS_OPERATIONAL_DIR / "operational_budget_metrics.csv"
    if budget_csv.exists():
        budget_df = pd.read_csv(budget_csv)
        test_budget = budget_df.loc[budget_df["split"].eq("test")].sort_values("budget_pct")
        if not test_budget.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                test_budget["budget_pct"],
                test_budget["precision_at_budget"],
                marker="o",
                label="Precision",
                color="#0f766e",
            )
            ax.plot(
                test_budget["budget_pct"],
                test_budget["recall_at_budget"],
                marker="o",
                label="Recall",
                color="#f97316",
            )
            ax.set_title("Trade-off de orçamento operacional no teste")
            ax.set_xlabel("% orçamento de inspeção")
            ax.set_ylabel("Métrica")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(alpha=0.25)
            path = figures_dir / "operational_budget_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    segment_csv = REPORTS_SEGMENT_DIR / "segment_topk_tag_day_metrics.csv"
    if segment_csv.exists():
        segment_df = pd.read_csv(segment_csv)
        segment_top15 = segment_df.loc[segment_df["top_k_tags_per_day"].eq(15)].copy()
        segment_top15 = segment_top15.loc[segment_top15["status"].ne("inconclusivo")]
        if not segment_top15.empty:
            segment_top15["segmento"] = (
                segment_top15["segment_col"].astype(str)
                + ": "
                + segment_top15["segment_value"].astype(str)
            )
            segment_top15 = segment_top15.sort_values("recall_at_k").tail(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=segment_top15,
                x="precision_at_k",
                y="recall_at_k",
                size="total_positives",
                hue="segment_col",
                sizes=(60, 260),
                ax=ax,
            )
            for _, row in segment_top15.iterrows():
                ax.text(
                    row["precision_at_k"] + 0.005,
                    row["recall_at_k"],
                    row["segmento"],
                    fontsize=7,
                )
            ax.set_title("Desempenho Top15 por segmento operacional")
            ax.set_xlabel("Precision@15")
            ax.set_ylabel("Recall@15")
            ax.set_xlim(0, min(1.05, max(segment_top15["precision_at_k"].max() * 1.2, 0.1)))
            ax.set_ylim(0, min(1.05, max(segment_top15["recall_at_k"].max() * 1.2, 0.1)))
            ax.legend(fontsize=7, loc="lower right")
            path = figures_dir / "segments_top15_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    return generated


def write_eda_report(
    summary: dict[str, Any],
    figure_paths: list[Path],
    report_path: Path = REPORTS_EDA_DIR / "eda_report.md",
) -> Path:
    """Escreve relatorio markdown da EDA com principais achados."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    imbalance_note = ""
    if summary["target_4h_positives"] == 0:
        imbalance_note = (
            "- **Alerta de dados**: não há positivos em `target_4h` no dataset atual; "
            "isso bloqueia treino supervisionado útil e exige revisão da fonte/janela de eventos.\n"
        )

    lines = [
        "# Relatório de EDA - Etapa 4",
        "",
        "## Escopo",
        "",
        "Análise exploratória orientada a decisão para o dataset rotulado "
        "`apontamentos_labeled.parquet`.",
        "",
        "## Sumário Executivo",
        "",
        f"- Registros analisados: `{summary['total_rows']}`",
        f"- Tags únicas: `{summary['total_tags']}`",
        f"- Frotas únicas: `{summary['total_frotas']}`",
        f"- Tipos únicos: `{summary['total_tipos']}`",
        f"- Positivos target_4h: `{summary['target_4h_positives']}`",
        f"- Taxa de positivos: `{summary['target_4h_positive_rate_pct']}%`",
        imbalance_note,
        "## Cobertura temporal",
        "",
        f"- Início mínimo: `{summary['inicio_min']}`",
        f"- Início máximo: `{summary['inicio_max']}`",
        "",
        "## Duração de ciclo",
        "",
        f"- Média: `{round(summary['duracao_ciclo_min_mean'], 4)} min`",
        f"- Mediana: `{round(summary['duracao_ciclo_min_median'], 4)} min`",
        f"- P95: `{round(summary['duracao_ciclo_min_p95'], 4)} min`",
        "",
        "## Qualidade de dados (top 5 colunas com mais nulos)",
        "",
        "| Coluna | % nulos |",
        "|---|---:|",
    ]
    for column, pct in summary["missing_pct_top5"].items():
        lines.append(f"| {column} | {pct}% |")

    lines.extend(["", "## Figuras geradas", ""])
    for path in figure_paths:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Decisões e próximos passos",
            "",
            "1. Revisar cobertura de eventos críticos para destravar positivos no `target_4h`.",
            "2. Validar sincronização temporal entre apontamentos e telemetria.",
            "3. Seguir para Etapa 5 apenas após estabilizar a rotulação com "
            "taxa de positivos não nula.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
    return report_path


def run_eda_pipeline(
    dataset_path: Path = LABELED_DATASET_PATH,
    figures_dir: Path = FIGURES_DIR,
    report_path: Path = REPORTS_EDA_DIR / "eda_report.md",
    include_project_artifacts: bool = True,
) -> dict[str, Any]:
    """Executa EDA completa com saídas persistidas."""
    df = load_labeled_dataset(dataset_path=dataset_path)
    eda_df = enrich_for_eda(df)
    summary = build_eda_summary(eda_df)
    figures = generate_eda_figures(eda_df, figures_dir=figures_dir)
    if include_project_artifacts:
        figures.extend(generate_project_artifact_figures(figures_dir=figures_dir))
    report = write_eda_report(summary, figures, report_path=report_path)
    return {
        "dataset_path": str(dataset_path),
        "report_path": str(report),
        "figures": [str(item) for item in figures],
        "total_rows": summary["total_rows"],
        "target_4h_positives": summary["target_4h_positives"],
    }


def main() -> None:
    result = run_eda_pipeline()
    print(f"[OK] EDA dataset: {result['dataset_path']}")
    print(f"[OK] EDA report: {result['report_path']}")
    print(f"[OK] EDA figures: {len(result['figures'])}")
    print(f"[OK] Registros analisados: {result['total_rows']}")
    print(f"[OK] Positivos target_4h: {result['target_4h_positives']}")


if __name__ == "__main__":
    main()
