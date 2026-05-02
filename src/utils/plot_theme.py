"""Tema visual padrao para notebooks executivos e tecnicos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "navy": "#1f3a5f",
    "blue": "#2c7fb8",
    "teal": "#2ca58d",
    "orange": "#f29e4c",
    "red": "#d1495b",
    "slate": "#5b6470",
    "gray": "#d9dde3",
}


def apply_manager_theme() -> None:
    """Aplica estilo unico de graficos para a trilha executiva."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["gray"],
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "grid.color": PALETTE["gray"],
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
        }
    )


def add_source_note(
    ax: plt.Axes,
    note: str = "Fonte: artefatos oficiais em reports/",
) -> None:
    """Adiciona nota de rodape padrao para leitura executiva."""
    ax.text(
        0.0,
        -0.22,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["slate"],
        ha="left",
        va="top",
    )
