"""Etapa 4: EDA orientada a decisao com saidas reproduziveis e graficos profissionais."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
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
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

matplotlib.use("Agg")


# ─────────────────────────────────────────────
# TEMA VISUAL
# ─────────────────────────────────────────────

PALETTE = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "blue": "#58a6ff",
    "teal": "#3fb950",
    "orange": "#f0883e",
    "red": "#f85149",
    "purple": "#a371f7",
    "yellow": "#e3b341",
    "navy": "#1f6feb",
    "pink": "#db61a2",
    "cyan": "#39d3f0",
}

# Gradientes customizados
CMAP_HEAT = mcolors.LinearSegmentedColormap.from_list("heat", ["#0d1117", "#f0883e", "#f85149"])
CMAP_TEAL = mcolors.LinearSegmentedColormap.from_list("teal", ["#0d1117", "#3fb950"])
CMAP_BLUE = mcolors.LinearSegmentedColormap.from_list("blue", ["#0d1117", "#58a6ff"])
CMAP_CM = mcolors.LinearSegmentedColormap.from_list("cm", ["#0d1117", "#1f6feb"])
CMAP_CORR = mcolors.LinearSegmentedColormap.from_list("corr", ["#f85149", "#161b22", "#58a6ff"])

FONT_TITLE = {
    "fontsize": 13,
    "fontweight": "bold",
    "color": PALETTE["text"],
    "fontfamily": "monospace",
}
FONT_LABEL = {"fontsize": 10, "color": PALETTE["muted"], "fontfamily": "monospace"}
FONT_TICK = {"labelsize": 9, "colors": PALETTE["muted"]}
FONT_ANNOT = {"fontsize": 8, "color": PALETTE["muted"], "fontfamily": "monospace"}


def _apply_theme(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Aplica tema escuro consistente a figura e eixos."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(axis="x", **FONT_TICK)
        ax.tick_params(axis="y", **FONT_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.xaxis.label.set_color(PALETTE["muted"])
        ax.yaxis.label.set_color(PALETTE["muted"])
        ax.title.set_color(PALETTE["text"])
        ax.grid(color=PALETTE["border"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)


def _styled_bar(
    ax: plt.Axes,
    x,
    y,
    color: str,
    orientation: str = "v",
    alpha_gradient: bool = True,
) -> None:
    """Barras com gradiente de opacidade e borda sutil."""
    n = len(x) if orientation == "v" else len(y)
    alphas = np.linspace(0.55, 1.0, n) if alpha_gradient else [1.0] * n
    if orientation == "v":
        for xi, yi, a in zip(x, y, alphas, strict=False):
            ax.bar(
                xi, yi, color=color, alpha=a, edgecolor=PALETTE["border"], linewidth=0.6, zorder=3
            )
    else:
        for xi, yi, a in zip(x, y, alphas, strict=False):
            ax.barh(
                xi, yi, color=color, alpha=a, edgecolor=PALETTE["border"], linewidth=0.6, zorder=3
            )


def add_source_note(ax: plt.Axes, source: str) -> None:
    ax.annotate(
        source,
        xy=(1, -0.13),
        xycoords="axes fraction",
        ha="right",
        va="top",
        **FONT_ANNOT,
    )


def _save_figure(fig: plt.Figure, output_path: Path, apply_tight_layout: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────
# CARREGAMENTO E ENRIQUECIMENTO
# ─────────────────────────────────────────────


def load_labeled_dataset(dataset_path: Path = LABELED_DATASET_PATH) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset rotulado nao encontrado em {dataset_path}. "
            "Execute `make label` antes de `make eda`."
        )
    return pd.read_parquet(dataset_path)


def enrich_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Inicio"] = pd.to_datetime(out["Inicio"], errors="coerce", utc=True)
    out["Fim"] = pd.to_datetime(out["Fim"], errors="coerce", utc=True)
    out["duracao_ciclo_min"] = (out["Fim"] - out["Inicio"]).dt.total_seconds() / 60.0
    out["hora_do_dia"] = out["Inicio"].dt.hour
    out["dia_da_semana"] = out["Inicio"].dt.dayofweek
    out["is_fim_de_semana"] = out["dia_da_semana"].isin([5, 6]).astype(int)
    return out


def build_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    total_rows = int(len(df))
    positives = int(df["target_4h"].sum())
    positive_rate = round((positives / total_rows) * 100.0, 6) if total_rows else 0.0
    return {
        "total_rows": total_rows,
        "total_tags": int(df["Tag"].nunique()),
        "total_frotas": int(df["Frota"].nunique()),
        "total_tipos": int(df["Tipo"].nunique()),
        "target_4h_positives": positives,
        "target_4h_positive_rate_pct": positive_rate,
        "inicio_min": str(df["Inicio"].min()),
        "inicio_max": str(df["Inicio"].max()),
        "duracao_ciclo_min_mean": float(df["duracao_ciclo_min"].mean()),
        "duracao_ciclo_min_median": float(df["duracao_ciclo_min"].median()),
        "duracao_ciclo_min_p95": float(df["duracao_ciclo_min"].quantile(0.95)),
        "missing_pct_top5": (
            (df.isna().mean() * 100.0).sort_values(ascending=False).head(5).round(4).to_dict()
        ),
    }


def _target_rate_by_category(
    df: pd.DataFrame,
    column: str,
    top: int = 12,
    min_rows: int = 20,
) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(ciclos=("target_4h", "size"), taxa_alerta=("target_4h", "mean"))
        .reset_index()
        .rename(columns={column: "categoria"})
    )
    grouped["categoria"] = grouped["categoria"].fillna("MISSING").astype(str)
    return grouped.loc[grouped["ciclos"].ge(min_rows)].nlargest(top, "taxa_alerta")


def _pct_fmt(value, _):
    return f"{value:.0%}"


def _plain_fmt(ax: plt.Axes, axis: str = "y") -> None:
    ax.ticklabel_format(axis=axis, style="plain")


# ─────────────────────────────────────────────
# FIGURAS DA EDA
# ─────────────────────────────────────────────


def generate_eda_figures(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1) Distribuicao do target
    fig, ax = plt.subplots(figsize=(6, 4.2))
    counts = df["target_4h"].value_counts().sort_index()
    colors = [PALETTE["blue"], PALETTE["red"]]
    bars = ax.bar(
        [f"Classe {int(k)}" for k in counts.index],
        counts.values,
        color=colors,
        edgecolor=PALETTE["border"],
        linewidth=0.8,
        zorder=3,
    )
    for bar, val in zip(bars, counts.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PALETTE["text"],
            fontfamily="monospace",
        )
    total = counts.sum()
    pct_labels = [f"{v/total:.1%}" for v in counts.values]
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(
        [
            f"{lbl}\n{pct}"
            for lbl, pct in zip(["Sem alerta\n(0)", "Com alerta\n(1)"], pct_labels, strict=False)
        ],
        color=PALETTE["muted"],
        fontsize=9,
    )
    ax.set_title("Distribuição do target_4h", **FONT_TITLE)
    ax.set_ylabel("Registros", **FONT_LABEL)
    _plain_fmt(ax, "y")
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_target_distribution.png"
    _save_figure(fig, path)
    generated.append(path)

    # 2) Ciclos por hora do dia
    fig, ax = plt.subplots(figsize=(11, 4.2))
    hour_counts = (
        df["hora_do_dia"].value_counts(dropna=True).sort_index().reindex(range(24), fill_value=0)
    )
    norm = plt.Normalize(hour_counts.min(), hour_counts.max())
    bar_colors = [CMAP_TEAL(norm(v)) for v in hour_counts.values]
    ax.bar(
        hour_counts.index,
        hour_counts.values,
        color=bar_colors,
        edgecolor=PALETTE["border"],
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], **FONT_ANNOT)
    ax.set_title("Ciclos por hora do dia", **FONT_TITLE)
    ax.set_xlabel("Hora", **FONT_LABEL)
    ax.set_ylabel("Ciclos", **FONT_LABEL)
    _plain_fmt(ax, "y")
    sm = plt.cm.ScalarMappable(cmap=CMAP_TEAL, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
    cb.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
    cb.outline.set_edgecolor(PALETTE["border"])
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_ciclos_por_hora.png"
    _save_figure(fig, path)
    generated.append(path)

    # 3) Top frotas
    top_frotas = df["Frota"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = plt.Normalize(top_frotas.min(), top_frotas.max())
    bar_colors = [CMAP_BLUE(norm(v)) for v in top_frotas.values]
    ax.barh(
        top_frotas.index.astype(str),
        top_frotas.values,
        color=bar_colors,
        edgecolor=PALETTE["border"],
        linewidth=0.5,
        zorder=3,
    )
    for val, name in zip(top_frotas.values, top_frotas.index, strict=False):
        ax.text(
            val + top_frotas.max() * 0.01,
            name,
            f"{val:,}",
            va="center",
            fontsize=8,
            color=PALETTE["muted"],
            fontfamily="monospace",
        )
    ax.set_title("Top 15 Frotas — volume de ciclos", **FONT_TITLE)
    ax.set_xlabel("Ciclos", **FONT_LABEL)
    ax.set_ylabel("")
    _plain_fmt(ax, "x")
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_top_frotas.png"
    _save_figure(fig, path)
    generated.append(path)

    # 4) Distribuicao de duracao de ciclo
    sample = df["duracao_ciclo_min"].dropna()
    if len(sample) > 50_000:
        sample = sample.sample(50_000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n, bins, patches = ax.hist(
        sample, bins=60, edgecolor=PALETTE["border"], linewidth=0.4, zorder=3
    )
    norm = plt.Normalize(n.min(), n.max())
    for patch, count in zip(patches, n, strict=False):
        patch.set_facecolor(CMAP_BLUE(norm(count)))
    p50 = sample.median()
    p95 = sample.quantile(0.95)
    ax.axvline(
        p50, color=PALETTE["yellow"], linestyle="--", linewidth=1.4, label=f"Mediana: {p50:.1f} min"
    )
    ax.axvline(p95, color=PALETTE["red"], linestyle=":", linewidth=1.4, label=f"P95: {p95:.1f} min")
    ax.legend(
        framealpha=0.25,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=9,
    )
    ax.set_title("Distribuição da duração de ciclo (min)", **FONT_TITLE)
    ax.set_xlabel("Duração (min)", **FONT_LABEL)
    ax.set_ylabel("Frequência", **FONT_LABEL)
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_duracao_ciclo_hist.png"
    _save_figure(fig, path)
    generated.append(path)

    # 5) Top classes de atividade
    top_classes = df["Classe"].fillna("MISSING").value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    norm = plt.Normalize(top_classes.min(), top_classes.max())
    bar_colors = [CMAP_HEAT(norm(v)) for v in top_classes.values]
    ax.barh(
        top_classes.index.astype(str),
        top_classes.values,
        color=bar_colors,
        edgecolor=PALETTE["border"],
        linewidth=0.5,
        zorder=3,
    )
    for val, name in zip(top_classes.values, top_classes.index, strict=False):
        ax.text(
            val + top_classes.max() * 0.01,
            name,
            f"{val:,}",
            va="center",
            fontsize=8,
            color=PALETTE["muted"],
            fontfamily="monospace",
        )
    ax.set_title("Top classes de atividade", **FONT_TITLE)
    ax.set_xlabel("Ciclos", **FONT_LABEL)
    _plain_fmt(ax, "x")
    _apply_theme(fig, [ax])
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
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(daily["dia"], daily["ciclos"], alpha=0.18, color=PALETTE["blue"])
    ax.plot(daily["dia"], daily["ciclos"], color=PALETTE["blue"], linewidth=1.8, label="Ciclos/dia")
    ax.set_ylabel("Ciclos", **FONT_LABEL)
    _plain_fmt(ax, "y")
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.fill_between(daily["dia"], daily["taxa_alerta"], alpha=0.12, color=PALETTE["red"])
    ax2.plot(
        daily["dia"],
        daily["taxa_alerta"],
        color=PALETTE["red"],
        linewidth=1.6,
        linestyle="--",
        label="Taxa target_4h",
    )
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax2.set_ylabel("Taxa target_4h", color=PALETTE["red"], fontsize=10, fontfamily="monospace")
    ax2.tick_params(axis="y", colors=PALETTE["red"], labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="upper left",
        framealpha=0.25,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=9,
    )
    ax.set_title("Volume diário de ciclos e prevalência de target_4h", **FONT_TITLE)
    ax.set_xlabel("Data", **FONT_LABEL)
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_daily_volume_target_rate.png"
    _save_figure(fig, path)
    generated.append(path)

    # 7) Heatmap de risco por dia da semana e hora
    heatmap_data = (
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
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=CMAP_HEAT,
        cbar_kws={"label": "Taxa target_4h", "shrink": 0.8},
        linewidths=0.3,
        linecolor=PALETTE["bg"],
        xticklabels=[f"{h:02d}h" for h in range(24)],
    )
    ax.set_title("Taxa de target_4h por hora e dia da semana", **FONT_TITLE)
    ax.set_xlabel("Hora", **FONT_LABEL)
    ax.set_ylabel("")
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
    cbar.ax.set_facecolor(PALETTE["surface"])
    cbar.outline.set_edgecolor(PALETTE["border"])
    cbar.set_label("Taxa target_4h", color=PALETTE["muted"], fontsize=9)
    _apply_theme(fig, [ax])
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
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if positive_tags.empty:
        ax.text(
            0.5,
            0.5,
            "Sem positivos target_4h",
            ha="center",
            va="center",
            color=PALETTE["muted"],
            fontsize=12,
            fontfamily="monospace",
        )
        ax.axis("off")
    else:
        norm = plt.Normalize(positive_tags.min(), positive_tags.max())
        bar_colors = [CMAP_HEAT(norm(v)) for v in positive_tags.values]
        ax.barh(
            positive_tags.index,
            positive_tags.values,
            color=bar_colors,
            edgecolor=PALETTE["border"],
            linewidth=0.5,
            zorder=3,
        )
        for val, name in zip(positive_tags.values, positive_tags.index, strict=False):
            ax.text(
                val + positive_tags.max() * 0.01,
                name,
                f"{val:,}",
                va="center",
                fontsize=8,
                color=PALETTE["muted"],
                fontfamily="monospace",
            )
        ax.set_xlabel("Ciclos positivos target_4h", **FONT_LABEL)
        _plain_fmt(ax, "x")
    ax.set_title("Top 15 Tags — volume de alertas antecipáveis", **FONT_TITLE)
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_top_tags_target_positives.png"
    _save_figure(fig, path)
    generated.append(path)

    # 9) Taxa de alerta por frota
    frota_rate = _target_rate_by_category(df, "Frota", top=12, min_rows=20).sort_values(
        "taxa_alerta"
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    norm = plt.Normalize(frota_rate["taxa_alerta"].min(), frota_rate["taxa_alerta"].max())
    bar_colors = [CMAP_TEAL(norm(v)) for v in frota_rate["taxa_alerta"]]
    ax.barh(
        frota_rate["categoria"],
        frota_rate["taxa_alerta"],
        color=bar_colors,
        edgecolor=PALETTE["border"],
        linewidth=0.5,
        zorder=3,
    )
    for val, cat in zip(frota_rate["taxa_alerta"], frota_rate["categoria"], strict=False):
        ax.text(
            val + frota_rate["taxa_alerta"].max() * 0.01,
            cat,
            f"{val:.1%}",
            va="center",
            fontsize=8,
            color=PALETTE["muted"],
            fontfamily="monospace",
        )
    ax.set_title("Frotas com maior taxa de target_4h", **FONT_TITLE)
    ax.set_xlabel("Taxa target_4h", **FONT_LABEL)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_target_rate_by_frota.png"
    _save_figure(fig, path)
    generated.append(path)

    # 10) Taxa de alerta por Tipo e Classe
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, column, title, cmap in [
        (axes[0], "Tipo", "Taxa target_4h por Tipo", CMAP_BLUE),
        (axes[1], "Classe", "Taxa target_4h por Classe", CMAP_HEAT),
    ]:
        rate_table = _target_rate_by_category(df, column, top=10, min_rows=20).sort_values(
            "taxa_alerta"
        )
        if rate_table.empty:
            ax.text(
                0.5,
                0.5,
                "Sem dados",
                ha="center",
                va="center",
                color=PALETTE["muted"],
                fontsize=11,
                fontfamily="monospace",
            )
            ax.axis("off")
            ax.set_title(title, **FONT_TITLE)
            continue
        norm = plt.Normalize(rate_table["taxa_alerta"].min(), rate_table["taxa_alerta"].max())
        bar_colors = [cmap(norm(v)) for v in rate_table["taxa_alerta"]]
        ax.barh(
            rate_table["categoria"],
            rate_table["taxa_alerta"],
            color=bar_colors,
            edgecolor=PALETTE["border"],
            linewidth=0.5,
            zorder=3,
        )
        for val, cat in zip(rate_table["taxa_alerta"], rate_table["categoria"], strict=False):
            ax.text(
                val + rate_table["taxa_alerta"].max() * 0.02,
                cat,
                f"{val:.1%}",
                va="center",
                fontsize=8,
                color=PALETTE["muted"],
                fontfamily="monospace",
            )
        ax.set_title(title, **FONT_TITLE)
        ax.set_xlabel("Taxa target_4h", **FONT_LABEL)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    _apply_theme(fig, list(axes))
    path = figures_dir / "eda_target_rate_by_tipo_classe.png"
    _save_figure(fig, path)
    generated.append(path)

    # 11) Duracao por classe (boxplot)
    dur_cols = ["Classe", "duracao_ciclo_min"]
    dur_sample = df[dur_cols].dropna()
    if len(dur_sample) > 50_000:
        dur_sample = dur_sample.sample(50_000, random_state=42)
    top_dur_classes = dur_sample["Classe"].value_counts().head(8).index
    dur_sample = dur_sample.loc[dur_sample["Classe"].isin(top_dur_classes)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bp = ax.boxplot(
        [
            dur_sample.loc[dur_sample["Classe"] == c, "duracao_ciclo_min"].values
            for c in top_dur_classes
        ],
        tick_labels=[str(c) for c in top_dur_classes],
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops={"color": PALETTE["orange"], "linewidth": 2},
        whiskerprops={"color": PALETTE["muted"], "linewidth": 1},
        capprops={"color": PALETTE["muted"], "linewidth": 1.5},
        boxprops={"linewidth": 0.8},
    )
    box_colors = [
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["purple"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["yellow"],
        PALETTE["cyan"],
        PALETTE["pink"],
    ]
    for patch, color in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(PALETTE["border"])
    ax.set_title("Duração de ciclo por classe de atividade", **FONT_TITLE)
    ax.set_xlabel("Classe", **FONT_LABEL)
    ax.set_ylabel("Duração (min)", **FONT_LABEL)
    ax.tick_params(axis="x", rotation=30)
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_duracao_por_classe_boxplot.png"
    _save_figure(fig, path)
    generated.append(path)

    # 12) TTE para positivos
    if "tte_horas" in df.columns:
        tte_positive = df.loc[df["target_4h"].eq(1), "tte_horas"].dropna()
    else:
        tte_positive = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if tte_positive.empty:
        ax.text(
            0.5,
            0.5,
            "Sem tte_horas para positivos",
            ha="center",
            va="center",
            color=PALETTE["muted"],
            fontsize=11,
            fontfamily="monospace",
        )
        ax.axis("off")
    else:
        clipped = tte_positive.clip(lower=0, upper=4)
        n, bins, patches = ax.hist(
            clipped, bins=24, edgecolor=PALETTE["border"], linewidth=0.4, zorder=3
        )
        norm = plt.Normalize(n.min(), n.max())
        for patch, count in zip(patches, n, strict=False):
            patch.set_facecolor(CMAP_HEAT(norm(count)))
        ax.set_xlabel("Horas até evento crítico", **FONT_LABEL)
        ax.set_ylabel("Frequência", **FONT_LABEL)
    ax.set_title("Distribuição de antecedência dos ciclos positivos", **FONT_TITLE)
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_tte_horas_positivos_hist.png"
    _save_figure(fig, path)
    generated.append(path)

    # 13) Nulos por coluna
    missing = (df.isna().mean() * 100).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    norm = plt.Normalize(0, max(missing.max(), 1))
    bar_colors = [CMAP_HEAT(norm(v)) for v in missing.values]
    ax.barh(
        missing.index.astype(str),
        missing.values,
        color=bar_colors,
        edgecolor=PALETTE["border"],
        linewidth=0.5,
        zorder=3,
    )
    for val, name in zip(missing.values, missing.index, strict=False):
        ax.text(
            val + 0.3,
            name,
            f"{val:.1f}%",
            va="center",
            fontsize=8,
            color=PALETTE["muted"],
            fontfamily="monospace",
        )
    ax.set_title("Percentual de nulos por coluna (top 15)", **FONT_TITLE)
    ax.set_xlabel("% nulos", **FONT_LABEL)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _apply_theme(fig, [ax])
    path = figures_dir / "eda_missing_values.png"
    _save_figure(fig, path)
    generated.append(path)

    # 14) Heatmap de Correlacao Spearman
    numeric_cols = df.select_dtypes(include=["number"]).columns
    valid_cols = [c for c in numeric_cols if c not in {"Id"}]
    if valid_cols:
        corr = df[valid_cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True  # mostra so triangulo inferior
        sns.heatmap(
            corr,
            ax=ax,
            cmap=CMAP_CORR,
            center=0,
            mask=mask,
            annot=len(valid_cols) <= 15,
            fmt=".2f",
            annot_kws={"size": 7, "color": PALETTE["text"]},
            cbar_kws={"label": "Spearman ρ", "shrink": 0.7},
            linewidths=0.3,
            linecolor=PALETTE["bg"],
            square=True,
        )
        ax.set_title("Heatmap de Correlação Spearman — variáveis numéricas", **FONT_TITLE)
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
        cbar.outline.set_edgecolor(PALETTE["border"])
        cbar.set_label("Spearman ρ", color=PALETTE["muted"], fontsize=9)
        _apply_theme(fig, [ax])
        path = figures_dir / "eda_correlation_heatmap.png"
        _save_figure(fig, path)
        generated.append(path)

    return generated


# ─────────────────────────────────────────────
# FIGURAS DE MODELAGEM / OPERACAO
# ─────────────────────────────────────────────


def _plot_threshold_diagnostic(
    curve: pd.DataFrame,
    threshold_test: dict[str, Any],
    output_path: Path,
) -> None:
    required = {"threshold", "recall", "precision"}
    missing = required.difference(curve.columns)
    if missing:
        raise ValueError(f"Curva sem colunas: {sorted(missing)}")

    curve = curve.copy()
    for col in required:
        curve[col] = pd.to_numeric(curve[col], errors="coerce")
    curve = curve.dropna(subset=list(required)).sort_values("threshold")
    if curve.empty:
        raise ValueError("Curva vazia apos limpeza.")

    threshold = float(threshold_test["threshold"])
    closest_idx = (curve["threshold"] - threshold).abs().idxmin()
    sel_row = curve.loc[closest_idx]

    rows = int(threshold_test["rows"])
    tp = int(threshold_test["true_positive"])
    fp = int(threshold_test["false_positive"])
    fn = int(threshold_test["false_negative"])
    tn = rows - tp - fp - fn
    cm = [[tn, fp], [fn, tp]]
    total = max(sum(sum(r) for r in cm), 1)
    max_cell = max(max(r) for r in cm)

    fig = plt.figure(figsize=(15, 5.5), facecolor=PALETTE["bg"])
    gs = fig.add_gridspec(1, 2, wspace=0.38)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # — curva precision x recall —
    ax0.fill_between(curve["threshold"], curve["recall"], alpha=0.10, color=PALETTE["teal"])
    ax0.fill_between(curve["threshold"], curve["precision"], alpha=0.10, color=PALETTE["orange"])
    ax0.plot(
        curve["threshold"], curve["recall"], label="Recall", color=PALETTE["teal"], linewidth=2.2
    )
    ax0.plot(
        curve["threshold"],
        curve["precision"],
        label="Precision",
        color=PALETTE["orange"],
        linewidth=2.2,
    )
    ax0.axvline(
        threshold,
        color=PALETTE["red"],
        linestyle="--",
        linewidth=1.6,
        label=f"Threshold = {threshold:.4f}",
    )
    R = float(sel_row["recall"])
    P = float(sel_row["precision"])
    ax0.scatter([threshold], [R], color=PALETTE["teal"], s=70, zorder=5)
    ax0.scatter([threshold], [P], color=PALETTE["orange"], s=70, zorder=5)
    ax0.annotate(
        f"R={R:.2f}",
        xy=(threshold, R),
        xytext=(threshold + 0.04, R - 0.07),
        fontsize=8.5,
        color=PALETTE["teal"],
        arrowprops={"arrowstyle": "-", "color": PALETTE["teal"], "lw": 1},
    )
    ax0.annotate(
        f"P={P:.2f}",
        xy=(threshold, P),
        xytext=(threshold + 0.04, P + 0.05),
        fontsize=8.5,
        color=PALETTE["orange"],
        arrowprops={"arrowstyle": "-", "color": PALETTE["orange"], "lw": 1},
    )
    ax0.set_title("Precision × Recall em função do threshold", **FONT_TITLE)
    ax0.set_xlabel("Threshold", **FONT_LABEL)
    ax0.set_ylabel("Métrica", **FONT_LABEL)
    ax0.set_ylim(0, 1.05)
    ax0.set_xlim(curve["threshold"].min(), curve["threshold"].max())
    ax0.legend(
        framealpha=0.25,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=9,
    )
    add_source_note(ax0, "Fonte: model_selected_threshold_curve.csv")

    # — matriz de confusao —
    labels = [["TN", "FP"], ["FN", "TP"]]
    im = ax1.imshow(cm, cmap=CMAP_CM, aspect="auto")
    for ri, row in enumerate(cm):
        for ci, val in enumerate(row):
            text_color = "white" if max_cell and val > max_cell / 2 else PALETTE["text"]
            is_error = (ri, ci) in {(0, 1), (1, 0)}
            border_col = PALETTE["red"] if is_error else "none"
            ax1.add_patch(
                mpatches.FancyBboxPatch(
                    (ci - 0.48, ri - 0.48),
                    0.96,
                    0.96,
                    boxstyle="round,pad=0.02",
                    fill=False,
                    edgecolor=border_col,
                    linewidth=2.5,
                    zorder=3,
                )
            )
            ax1.text(
                ci,
                ri - 0.1,
                f"{labels[ri][ci]}\n{val:,}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )
            ax1.text(
                ci,
                ri + 0.3,
                f"({val/total:.1%})",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                zorder=4,
            )
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(
        ["Predito: Sem alerta", "Predito: Com alerta"], fontsize=9, color=PALETTE["muted"]
    )
    ax1.set_yticklabels(
        ["Real: Sem alerta", "Real: Com alerta"], fontsize=9, color=PALETTE["muted"]
    )
    ax1.set_title("Matriz de confusão — teste temporal", **FONT_TITLE)
    ax1.tick_params(length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.03)
    cb.set_label("Contagem", fontsize=8, color=PALETTE["muted"])
    cb.ax.yaxis.set_tick_params(color=PALETTE["muted"], labelsize=8)
    cb.outline.set_edgecolor(PALETTE["border"])
    add_source_note(ax1, "Fonte: operational_metrics_report.json")

    fig.suptitle(
        "Diagnóstico de threshold e erros de classificação",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["text"],
        y=1.01,
    )
    _apply_theme(fig, [ax0, ax1])
    fig.subplots_adjust(top=0.82, bottom=0.22)
    _save_figure(fig, output_path, apply_tight_layout=False)


def generate_project_artifact_figures(figures_dir: Path = FIGURES_DIR) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # threshold + confusion matrix
    threshold_csv = REPORTS_MODEL_SELECTION_DIR / "model_selected_threshold_curve.csv"
    operational_json = REPORTS_OPERATIONAL_DIR / "operational_metrics_report.json"
    if threshold_csv.exists() and operational_json.exists():
        curve = pd.read_csv(threshold_csv)
        report = json.loads(operational_json.read_text())
        threshold_test = next(
            (i for i in report.get("threshold_metrics", []) if i.get("split") == "test"), None
        )
        if threshold_test:
            path = figures_dir / "threshold_diagnostic_confusion_matrix.png"
            _plot_threshold_diagnostic(curve, threshold_test, path)
            generated.append(path)

    # model selection bar charts
    selection_csv = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.csv"
    if selection_csv.exists():
        sel_df = pd.read_csv(selection_csv)
        candidates = sel_df.loc[sel_df["role"].eq("official_candidate")].copy()
        if not candidates.empty:
            metric_cols = [
                "test_top15_precision_at_k",
                "test_top15_recall_at_k",
                "test_top15_lift_vs_random",
            ]
            plot_df = candidates.melt(
                id_vars="model_name",
                value_vars=[c for c in metric_cols if c in candidates.columns],
                var_name="metric",
                value_name="value",
            )
            fig, ax = plt.subplots(figsize=(11, 5))
            palette_colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]]
            sns.barplot(
                data=plot_df, x="model_name", y="value", hue="metric", ax=ax, palette=palette_colors
            )
            ax.set_title("Candidatos oficiais — métricas no teste temporal", **FONT_TITLE)
            ax.set_xlabel("Modelo", **FONT_LABEL)
            ax.set_ylabel("Valor", **FONT_LABEL)
            ax.tick_params(axis="x", rotation=20)
            ax.legend(
                title="Métrica",
                fontsize=8,
                framealpha=0.25,
                facecolor=PALETTE["surface"],
                edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text"],
            )
            _apply_theme(fig, [ax])
            path = figures_dir / "model_selection_test_top15_metrics.png"
            _save_figure(fig, path)
            generated.append(path)

            auc_cols = ["test_auc_pr", "test_auc_roc"]
            auc_df = candidates.melt(
                id_vars="model_name",
                value_vars=[c for c in auc_cols if c in candidates.columns],
                var_name="metric",
                value_name="value",
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=auc_df,
                x="model_name",
                y="value",
                hue="metric",
                ax=ax,
                palette=[PALETTE["purple"], PALETTE["cyan"]],
            )
            ax.set_title("AUC dos candidatos oficiais — teste temporal", **FONT_TITLE)
            ax.set_xlabel("Modelo", **FONT_LABEL)
            ax.set_ylabel("AUC", **FONT_LABEL)
            ax.set_ylim(0, min(1.0, max(auc_df["value"].max() * 1.15, 0.1)))
            ax.tick_params(axis="x", rotation=20)
            ax.legend(
                title="Métrica",
                fontsize=8,
                framealpha=0.25,
                facecolor=PALETTE["surface"],
                edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text"],
            )
            _apply_theme(fig, [ax])
            path = figures_dir / "model_selection_test_auc.png"
            _save_figure(fig, path)
            generated.append(path)

    # top-k operational curve
    topk_csv = REPORTS_OPERATIONAL_DIR / "operational_daily_topk_metrics.csv"
    if topk_csv.exists():
        topk_df = pd.read_csv(topk_csv)
        test_topk = topk_df.loc[topk_df["split"].eq("test")].sort_values("top_k_tags_per_day")
        if not test_topk.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.fill_between(
                test_topk["top_k_tags_per_day"],
                test_topk["precision_at_k"],
                alpha=0.12,
                color=PALETTE["blue"],
            )
            ax.fill_between(
                test_topk["top_k_tags_per_day"],
                test_topk["recall_at_k"],
                alpha=0.12,
                color=PALETTE["red"],
            )
            ax.plot(
                test_topk["top_k_tags_per_day"],
                test_topk["precision_at_k"],
                marker="o",
                markersize=5,
                label="Precision@K",
                color=PALETTE["blue"],
                linewidth=2,
            )
            ax.plot(
                test_topk["top_k_tags_per_day"],
                test_topk["recall_at_k"],
                marker="o",
                markersize=5,
                label="Recall@K",
                color=PALETTE["red"],
                linewidth=2,
            )
            ax.set_title("Curva operacional TopK Tag-dia no teste", **FONT_TITLE)
            ax.set_xlabel("Top K Tags por dia", **FONT_LABEL)
            ax.set_ylabel("Métrica", **FONT_LABEL)
            ax.set_ylim(0, 1)
            ax.legend(
                framealpha=0.25,
                facecolor=PALETTE["surface"],
                edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text"],
                fontsize=9,
            )
            _apply_theme(fig, [ax])
            path = figures_dir / "operational_topk_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    # budget trade-off
    budget_csv = REPORTS_OPERATIONAL_DIR / "operational_budget_metrics.csv"
    if budget_csv.exists():
        budget_df = pd.read_csv(budget_csv)
        test_budget = budget_df.loc[budget_df["split"].eq("test")].sort_values("budget_pct")
        if not test_budget.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.fill_between(
                test_budget["budget_pct"],
                test_budget["precision_at_budget"],
                alpha=0.12,
                color=PALETTE["teal"],
            )
            ax.fill_between(
                test_budget["budget_pct"],
                test_budget["recall_at_budget"],
                alpha=0.12,
                color=PALETTE["orange"],
            )
            ax.plot(
                test_budget["budget_pct"],
                test_budget["precision_at_budget"],
                marker="o",
                markersize=5,
                label="Precision",
                color=PALETTE["teal"],
                linewidth=2,
            )
            ax.plot(
                test_budget["budget_pct"],
                test_budget["recall_at_budget"],
                marker="o",
                markersize=5,
                label="Recall",
                color=PALETTE["orange"],
                linewidth=2,
            )
            ax.set_title("Trade-off de orçamento operacional no teste", **FONT_TITLE)
            ax.set_xlabel("% orçamento de inspeção", **FONT_LABEL)
            ax.set_ylabel("Métrica", **FONT_LABEL)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
            ax.set_ylim(0, 1)
            ax.legend(
                framealpha=0.25,
                facecolor=PALETTE["surface"],
                edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text"],
                fontsize=9,
            )
            _apply_theme(fig, [ax])
            path = figures_dir / "operational_budget_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    # segment scatter
    segment_csv = REPORTS_SEGMENT_DIR / "segment_topk_tag_day_metrics.csv"
    if segment_csv.exists():
        seg_df = pd.read_csv(segment_csv)
        seg_top15 = seg_df.loc[seg_df["top_k_tags_per_day"].eq(15)].copy()
        seg_top15 = seg_top15.loc[seg_top15["status"].ne("inconclusivo")]
        if not seg_top15.empty:
            seg_top15["segmento"] = (
                seg_top15["segment_col"].astype(str) + ": " + seg_top15["segment_value"].astype(str)
            )
            seg_top15 = seg_top15.sort_values("recall_at_k").tail(15)
            fig, ax = plt.subplots(figsize=(10, 6.5))
            seg_palette = [
                PALETTE["blue"],
                PALETTE["teal"],
                PALETTE["orange"],
                PALETTE["purple"],
                PALETTE["red"],
                PALETTE["cyan"],
            ]
            unique_cols = seg_top15["segment_col"].unique()
            col_color_map = {
                c: seg_palette[i % len(seg_palette)] for i, c in enumerate(unique_cols)
            }
            for _, row in seg_top15.iterrows():
                color = col_color_map[row["segment_col"]]
                size = 60 + (row["total_positives"] / seg_top15["total_positives"].max()) * 200
                ax.scatter(
                    row["precision_at_k"],
                    row["recall_at_k"],
                    s=size,
                    color=color,
                    alpha=0.75,
                    edgecolors=PALETTE["border"],
                    linewidth=0.8,
                    zorder=3,
                )
                ax.text(
                    row["precision_at_k"] + 0.005,
                    row["recall_at_k"],
                    row["segmento"],
                    fontsize=7,
                    color=PALETTE["muted"],
                )
            legend_handles = [mpatches.Patch(color=col_color_map[c], label=c) for c in unique_cols]
            ax.legend(
                handles=legend_handles,
                fontsize=7,
                loc="lower right",
                framealpha=0.25,
                facecolor=PALETTE["surface"],
                edgecolor=PALETTE["border"],
                labelcolor=PALETTE["text"],
            )
            ax.set_title("Desempenho Top15 por segmento operacional", **FONT_TITLE)
            ax.set_xlabel("Precision@15", **FONT_LABEL)
            ax.set_ylabel("Recall@15", **FONT_LABEL)
            ax.set_xlim(0, min(1.05, max(seg_top15["precision_at_k"].max() * 1.2, 0.1)))
            ax.set_ylim(0, min(1.05, max(seg_top15["recall_at_k"].max() * 1.2, 0.1)))
            _apply_theme(fig, [ax])
            path = figures_dir / "segments_top15_precision_recall.png"
            _save_figure(fig, path)
            generated.append(path)

    return generated


FIGURE_GUIDE: dict[str, dict[str, str]] = {
    "eda_target_distribution.png": {
        "title": "Distribuição do target_4h",
        "shows": "Compara a quantidade de ciclos sem alerta antecipável (`0`) e com alerta antecipável (`1`).",
        "read": "Use para entender o desbalanceamento do problema. Quanto menor a barra de positivos, mais difícil é treinar e avaliar modelos sem métricas específicas para classe rara.",
        "decision": "Justifica acompanhar recall, precision, lift e TopK em vez de olhar apenas acurácia.",
    },
    "eda_ciclos_por_hora.png": {
        "title": "Volume de ciclos por hora do dia",
        "shows": "Mostra em quais horários os ciclos operacionais se concentram.",
        "read": "Picos indicam janelas de maior atividade e possíveis mudanças de turno, carga operacional ou padrão de apontamento.",
        "decision": "Ajuda a calibrar análises por hora e a planejar rotinas de inspeção nos períodos de maior volume.",
    },
    "eda_top_frotas.png": {
        "title": "Top frotas por volume",
        "shows": "Lista as frotas com maior quantidade de ciclos registrados.",
        "read": "Frotas com mais registros dominam a amostra e podem influenciar estatísticas globais.",
        "decision": "Ajuda a separar efeito de volume de efeito de risco antes de priorizar uma frota.",
    },
    "eda_duracao_ciclo_hist.png": {
        "title": "Distribuição da duração de ciclo",
        "shows": "Histograma da duração dos ciclos em minutos.",
        "read": "Caudas longas ou concentrações incomuns podem indicar ciclos atípicos, paradas, atrasos ou problemas de registro.",
        "decision": "Apoia regras de tratamento de outliers e validação da coerência temporal entre início e fim do ciclo.",
    },
    "eda_top_classes.png": {
        "title": "Top classes de atividade",
        "shows": "Mostra as classes operacionais mais frequentes no dataset.",
        "read": "Classes muito frequentes tendem a ter maior influência no treinamento e na interpretação do modelo.",
        "decision": "Ajuda a revisar se as classes mais importantes para a operação estão bem representadas.",
    },
    "eda_daily_volume_target_rate.png": {
        "title": "Volume diário e taxa de target_4h",
        "shows": "Combina quantidade diária de ciclos com prevalência diária de alertas antecipáveis.",
        "read": "Dias com alta taxa de positivos e baixo volume merecem leitura diferente de dias com alto volume e taxa estável.",
        "decision": "Ajuda a identificar períodos anômalos, mudanças de regime e risco de avaliação temporal enviesada.",
    },
    "eda_target_rate_heatmap_hora_dia.png": {
        "title": "Taxa de target por hora e dia da semana",
        "shows": "Heatmap da taxa média de `target_4h` cruzando dia da semana e hora.",
        "read": "Células mais quentes indicam combinações de dia e horário com maior concentração relativa de positivos.",
        "decision": "Ajuda a investigar padrões de turno, manutenção, operação e exposição ao risco.",
    },
    "eda_top_tags_target_positives.png": {
        "title": "Tags com mais positivos",
        "shows": "Mostra os equipamentos/tags com maior volume absoluto de ciclos positivos.",
        "read": "Volume alto de positivos não significa necessariamente maior taxa de risco; pode refletir maior quantidade de ciclos.",
        "decision": "Serve como primeira lista de equipamentos para investigação operacional e validação com campo.",
    },
    "eda_target_rate_by_frota.png": {
        "title": "Taxa de target por frota",
        "shows": "Compara frotas pela taxa de ciclos positivos, filtrando grupos com volume mínimo.",
        "read": "A leitura é de risco relativo, não de volume absoluto. Uma frota menor pode aparecer no topo se a proporção de positivos for alta.",
        "decision": "Ajuda a priorizar investigação de frotas com prevalência acima do padrão geral.",
    },
    "eda_target_rate_by_tipo_classe.png": {
        "title": "Taxa de target por tipo e classe",
        "shows": "Compara o risco relativo por `Tipo` e por `Classe`.",
        "read": "Barras maiores indicam categorias com maior proporção de ciclos positivos, respeitando o volume mínimo configurado.",
        "decision": "Ajuda a conectar risco a contexto operacional, tipo de equipamento e classe de atividade.",
    },
    "eda_duracao_por_classe_boxplot.png": {
        "title": "Duração por classe de atividade",
        "shows": "Boxplots de duração de ciclo nas classes mais frequentes.",
        "read": "Medianas e dispersões diferentes sugerem que a classe operacional muda o comportamento normal de duração.",
        "decision": "Ajuda a validar features temporais e a diferenciar atrasos esperados de comportamentos atípicos.",
    },
    "eda_tte_horas_positivos_hist.png": {
        "title": "Antecedência dos ciclos positivos",
        "shows": "Distribuição de `tte_horas` para ciclos positivos, limitada à janela de 4 horas.",
        "read": "Mostra quanto tempo antes do evento crítico os ciclos positivos aparecem.",
        "decision": "Ajuda a avaliar se a janela de antecipação é operacionalmente útil para gerar ação de campo.",
    },
    "eda_missing_values.png": {
        "title": "Percentual de nulos por coluna",
        "shows": "Lista as colunas com maior percentual de valores ausentes.",
        "read": "Nulos esperados em variáveis de evento futuro devem ser interpretados diferente de nulos em dados operacionais básicos.",
        "decision": "Orienta tratamento de dados, imputação e revisão de qualidade antes da modelagem.",
    },
    "eda_correlation_heatmap.png": {
        "title": "Correlação entre variáveis numéricas",
        "shows": "Heatmap de correlação de Spearman entre variáveis numéricas.",
        "read": "Cores fortes indicam relações monotônicas. Correlação alta entre variáveis pode indicar redundância ou vazamento se envolver alvo/futuro.",
        "decision": "Ajuda a revisar multicolinearidade, redundância e sinais suspeitos antes do treinamento.",
    },
    "threshold_diagnostic_confusion_matrix.png": {
        "title": "Diagnóstico de threshold e matriz de confusão",
        "shows": "Une curva Precision/Recall por threshold e matriz de confusão no teste temporal.",
        "read": "A linha vertical marca o threshold oficial. A matriz mostra acertos e erros ciclo-a-ciclo: TN, FP, FN e TP.",
        "decision": "Explica o trade-off entre capturar positivos e gerar volume de alertas, reforçando a necessidade de priorização TopK.",
    },
    "model_selection_test_top15_metrics.png": {
        "title": "Comparação de candidatos em Top15",
        "shows": "Compara modelos candidatos em Precision@Top15, Recall@Top15 e Lift no teste temporal.",
        "read": "Modelos melhores para operação devem equilibrar captura de positivos e qualidade da lista diária priorizada.",
        "decision": "Apoia a escolha do modelo oficial sob a métrica mais próxima da rotina de inspeção.",
    },
    "model_selection_test_auc.png": {
        "title": "AUC dos modelos candidatos",
        "shows": "Compara AUC-PR e AUC-ROC dos modelos candidatos no teste temporal.",
        "read": "AUC-PR é mais informativa quando há desbalanceamento; AUC-ROC ajuda a medir separação geral.",
        "decision": "Complementa a decisão de seleção, mas não substitui métricas operacionais TopK.",
    },
    "operational_topk_precision_recall.png": {
        "title": "Curva operacional TopK",
        "shows": "Mostra Precision@K e Recall@K conforme aumenta o número de tags priorizadas por dia.",
        "read": "K maior captura mais positivos, mas tende a reduzir precisão e aumentar carga de inspeção.",
        "decision": "Ajuda a escolher um K operacionalmente viável, como Top15 Tag-dia.",
    },
    "operational_budget_precision_recall.png": {
        "title": "Trade-off por orçamento de inspeção",
        "shows": "Mostra precision e recall para diferentes percentuais de orçamento de inspeção.",
        "read": "Orçamentos maiores aumentam cobertura, mas podem reduzir qualidade média dos alertas.",
        "decision": "Ajuda a alinhar desempenho do modelo com capacidade real de inspeção da operação.",
    },
    "segments_top15_precision_recall.png": {
        "title": "Desempenho Top15 por segmento",
        "shows": "Compara precision e recall em segmentos operacionais para Top15.",
        "read": "Segmentos distantes do padrão geral podem indicar onde o modelo performa melhor ou pior.",
        "decision": "Ajuda a direcionar calibração, monitoramento e investigação por frota, tipo, classe ou outro recorte.",
    },
}


def _report_figure_path(path: Path, report_path: Path) -> str:
    """Retorna caminho relativo ao relatório quando possível."""
    try:
        return path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ─────────────────────────────────────────────
# RELATORIO MARKDOWN
# ─────────────────────────────────────────────


def write_eda_report(
    summary: dict[str, Any],
    figure_paths: list[Path],
    report_path: Path = REPORTS_EDA_DIR / "eda_report.md",
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    imbalance_note = ""
    if summary["target_4h_positives"] == 0:
        imbalance_note = (
            "- **Alerta de dados**: não há positivos em `target_4h` no dataset atual; "
            "isso bloqueia treino supervisionado útil e exige revisão da fonte/janela de eventos.\n"
        )

    lines = [
        "# Relatório de EDA — Etapa 4",
        "",
        "## Escopo",
        "",
        (
            "Análise exploratória orientada a decisão para o dataset rotulado "
            "`apontamentos_labeled.parquet`. O objetivo é explicar a estrutura dos dados, "
            "a distribuição do alvo `target_4h` e os principais recortes que sustentam "
            "as decisões de modelagem e operação."
        ),
        "",
        "`target_4h = 1` representa um ciclo associado a alerta/evento crítico dentro "
        "da janela operacional de antecipação de 4 horas, conforme a regra de rotulação "
        "do projeto.",
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
        (
            "A taxa de positivos indica a prevalência do evento que o modelo precisa "
            "antecipar. Como o problema é desbalanceado, as figuras devem ser lidas com "
            "atenção a recall, precision, lift e volume operacional de alertas."
        ),
        "",
        "## Cobertura temporal",
        "",
        f"- Início mínimo: `{summary['inicio_min']}`",
        f"- Início máximo: `{summary['inicio_max']}`",
        "",
        (
            "A leitura temporal é importante porque o projeto usa validação temporal: "
            "o comportamento mais antigo não deve contaminar a avaliação dos períodos "
            "mais recentes."
        ),
        "",
        "## Duração de ciclo",
        "",
        f"- Média: `{round(summary['duracao_ciclo_min_mean'], 4)} min`",
        f"- Mediana: `{round(summary['duracao_ciclo_min_median'], 4)} min`",
        f"- P95: `{round(summary['duracao_ciclo_min_p95'], 4)} min`",
        "",
        (
            "A diferença entre média, mediana e P95 ajuda a identificar assimetria: "
            "quando o P95 fica muito acima da mediana, existe uma cauda de ciclos longos "
            "que pode afetar features de tempo e regras de priorização."
        ),
        "",
        "## Qualidade de dados (top 5 colunas com mais nulos)",
        "",
        "| Coluna | % nulos |",
        "|---|---:|",
    ]
    for col, pct in summary["missing_pct_top5"].items():
        lines.append(f"| {col} | {pct}% |")

    lines.extend(
        [
            "",
            (
                "Os nulos devem ser interpretados pelo papel da coluna. Campos derivados "
                "de evento futuro, como `tte_horas`, podem ser nulos quando não há evento "
                "crítico associado; já nulos em colunas operacionais básicas exigem "
                "revisão de qualidade."
            ),
            "",
            "## Como interpretar as figuras",
            "",
            (
                "Cada figura abaixo responde a uma pergunta prática. A leitura recomendada "
                "é combinar volume, taxa e impacto operacional: uma categoria com muitos "
                "positivos pode ser grande apenas porque aparece muito, enquanto uma taxa "
                "alta pode representar risco relativo mesmo com menor volume."
            ),
        ]
    )
    for index, path in enumerate(figure_paths, start=1):
        rel_path = _report_figure_path(path, report_path)
        guide = FIGURE_GUIDE.get(path.name)
        if guide is None:
            lines.extend(
                [
                    "",
                    f"### {index}. {path.name}",
                    "",
                    f"![{path.name}]({rel_path})",
                    "",
                    f"- **Arquivo:** `{rel_path}`",
                    "- **Leitura:** figura gerada automaticamente pelo pipeline de EDA.",
                ]
            )
            continue

        lines.extend(
            [
                "",
                f"### {index}. {guide['title']}",
                "",
                f"![{guide['title']}]({rel_path})",
                "",
                f"- **Arquivo:** `{rel_path}`",
                f"- **O que mostra:** {guide['shows']}",
                f"- **Como ler:** {guide['read']}",
                f"- **Decisão apoiada:** {guide['decision']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Decisões e próximos passos",
            "",
            (
                "1. Monitorar a estabilidade temporal da taxa de `target_4h`, principalmente "
                "em dias ou horários com variação forte de volume."
            ),
            (
                "2. Validar com o time operacional as tags, frotas, tipos e classes que aparecem "
                "com maior volume ou maior taxa de positivos."
            ),
            (
                "3. Usar as figuras de threshold, TopK e orçamento para escolher um ponto de "
                "operação compatível com a capacidade diária de inspeção."
            ),
            (
                "4. Revisar colunas nulas e correlações altas antes de promover novas features "
                "ou retreinar modelos."
            ),
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
    return report_path


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────


def run_eda_pipeline(
    dataset_path: Path = LABELED_DATASET_PATH,
    figures_dir: Path = FIGURES_DIR,
    report_path: Path = REPORTS_EDA_DIR / "eda_report.md",
    include_project_artifacts: bool = True,
) -> dict[str, Any]:
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
        "figures": [str(f) for f in figures],
        "total_rows": summary["total_rows"],
        "target_4h_positives": summary["target_4h_positives"],
    }


def main() -> None:
    setup_logging()
    result = run_eda_pipeline()
    logger.info(f"EDA dataset: {result['dataset_path']}")
    logger.info(f"EDA report:  {result['report_path']}")
    logger.info(f"EDA figures: {len(result['figures'])}")
    logger.info(f"Registros:   {result['total_rows']}")
    logger.info(f"Positivos:   {result['target_4h_positives']}")


if __name__ == "__main__":
    main()
