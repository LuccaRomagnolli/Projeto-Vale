"""Etapa 4: EDA orientada a decisao com saidas reproduziveis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR, LABELED_DATASET_PATH, REPORTS_DIR

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

    return generated


def write_eda_report(
    summary: dict[str, Any],
    figure_paths: list[Path],
    report_path: Path = REPORTS_DIR / "eda_report.md",
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
    report_path: Path = REPORTS_DIR / "eda_report.md",
) -> dict[str, Any]:
    """Executa EDA completa com saídas persistidas."""
    df = load_labeled_dataset(dataset_path=dataset_path)
    eda_df = enrich_for_eda(df)
    summary = build_eda_summary(eda_df)
    figures = generate_eda_figures(eda_df, figures_dir=figures_dir)
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
