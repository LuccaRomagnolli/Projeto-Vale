"""Dashboard executivo da EDA do projeto Vale — estética Apple minimalista."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "labeled" / "apontamentos_labeled.parquet"
LOGO_PATH = BASE_DIR / "pictures" / "vale-logo-removebg-preview.png"

# ── Paleta Apple-inspired ────────────────────────────────────────────────────
COLORS = {
    "bg":           "#f5f5f7",
    "surface":      "#ffffff",
    "surface_2":    "#f5f5f7",
    "surface_3":    "#e8e8ed",
    "border":       "rgba(0,0,0,0.08)",
    "border_strong":"rgba(0,0,0,0.14)",
    "text":         "#1d1d1f",
    "text_2":       "#424245",
    "text_3":       "#6e6e73",
    "blue":         "#0071e3",
    "blue_light":   "#e8f1fb",
    "green":        "#1d8348",
    "green_light":  "#e8f5e9",
    "amber":        "#9a6700",
    "amber_light":  "#fff8e1",
    "red":          "#c0392b",
    "red_light":    "#fdecea",
}

PLOTLY_TEMPLATE = "plotly_white"

DAYS_PT = {
    0: "Seg",
    1: "Ter",
    2: "Qua",
    3: "Qui",
    4: "Sex",
    5: "Sáb",
    6: "Dom",
}

st.set_page_config(
    page_title="Vale · Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Tema ─────────────────────────────────────────────────────────────────────
def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

        html, body, [class*="css"] {{
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
            -webkit-font-smoothing: antialiased;
        }}

        .stApp {{
            background: {COLORS["bg"]};
            color: {COLORS["text"]};
        }}

        /* ── Sidebar removida ────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            display: none !important;
        }}
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        /* ── Main content ───────────────────────────────────────────── */
        .block-container {{
            padding: 2rem 2.5rem 3rem;
            max-width: 1400px;
        }}

        /* ── Hero ───────────────────────────────────────────────────── */
        .hero-wrap {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 32px 36px 28px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.04);
            position: relative;
            overflow: hidden;
        }}
        .hero-wrap::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {COLORS["blue"]} 0%, #34aadc 50%, {COLORS["green"]} 100%);
        }}
        .hero-left {{
            display: flex;
            align-items: center;
            gap: 24px;
        }}
        .hero-logo-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 112px;
            min-height: 72px;
            flex-shrink: 0;
        }}
        .hero-logo {{
            display: block;
            width: 104px;
            max-height: 72px;
            height: auto;
            object-fit: contain;
        }}
        .hero-text h1 {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {COLORS["text"]};
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0 0 5px;
        }}
        .hero-text p {{
            font-size: 0.88rem;
            color: {COLORS["text_3"]};
            margin: 0;
            line-height: 1.5;
            max-width: 560px;
        }}
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: {COLORS["blue_light"]};
            color: {COLORS["blue"]};
            font-size: 0.73rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            margin-top: 10px;
        }}
        .hero-meta {{
            text-align: right;
            flex-shrink: 0;
        }}
        .hero-meta .meta-label {{
            font-size: 0.72rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_3"]};
            margin-bottom: 4px;
        }}
        .hero-meta .meta-value {{
            font-size: 0.95rem;
            font-weight: 600;
            color: {COLORS["text"]};
        }}

        /* ── KPI cards ──────────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 20px 20px 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 16px rgba(0,0,0,0.04);
            transition: box-shadow 0.2s;
        }}
        [data-testid="stMetric"]:hover {{
            box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.08);
        }}
        [data-testid="stMetricLabel"] p {{
            font-size: 0.73rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_3"]};
        }}
        [data-testid="stMetricValue"] {{
            color: {COLORS["text"]};
            font-size: 1.65rem !important;
            font-weight: 700;
            letter-spacing: -0.03em;
        }}
        [data-testid="stMetricDelta"] {{
            font-size: 0.78rem;
            font-weight: 500;
        }}

        /* ── Insights ───────────────────────────────────────────────── */
        .insight-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 92px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .insight-icon {{
            font-size: 1.1rem;
            margin-bottom: 4px;
        }}
        .insight-label {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_3"]};
        }}
        .insight-value {{
            font-size: 0.92rem;
            font-weight: 500;
            color: {COLORS["text"]};
            line-height: 1.4;
        }}

        /* ── Section label ──────────────────────────────────────────── */
        .section-title {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {COLORS["text_3"]};
            margin: 28px 0 12px;
        }}

        /* ── Tabs ───────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background: {COLORS["surface_3"]};
            border-radius: 10px;
            padding: 3px;
            width: fit-content;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px;
            padding: 7px 18px;
            font-size: 0.84rem;
            font-weight: 500;
            color: {COLORS["text_2"]};
            border: none;
        }}
        .stTabs [aria-selected="true"] {{
            background: {COLORS["surface"]} !important;
            color: {COLORS["text"]} !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}

        /* ── Chart containers ───────────────────────────────────────── */
        .chart-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}

        /* ── Dataframe ──────────────────────────────────────────────── */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ── Caption / footnotes ────────────────────────────────────── */
        .stCaption {{
            font-size: 0.77rem;
            color: {COLORS["text_3"]};
        }}

        /* ── Divider ────────────────────────────────────────────────── */
        hr {{
            border: none;
            border-top: 1px solid {COLORS["border"]};
            margin: 20px 0;
        }}

        /* hide Streamlit default footer / hamburger */
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Dados ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando dados…")
def load_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    if not dataset_path.exists():
        st.error(f"Dataset não encontrado: {dataset_path}")
        st.stop()

    df = pd.read_parquet(dataset_path)
    df["Inicio"] = pd.to_datetime(df["Inicio"], errors="coerce", utc=True)
    df["Fim"] = pd.to_datetime(df["Fim"], errors="coerce", utc=True)
    df["duracao_ciclo_min"] = (df["Fim"] - df["Inicio"]).dt.total_seconds() / 60.0
    df["hora_do_dia"] = df["Inicio"].dt.hour
    df["dia_da_semana"] = df["Inicio"].dt.dayofweek
    df["dia_nome"] = df["dia_da_semana"].map(DAYS_PT)
    df["mes"] = df["Inicio"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    df["data"] = df["Inicio"].dt.date
    return df


# ── Formatação ───────────────────────────────────────────────────────────────
def fmt_int(v: int | float) -> str:
    return f"{int(v):,}".replace(",", ".")


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%".replace(".", ",")


def fmt_float(v: float, suffix: str = "") -> str:
    return f"{v:.1f}{suffix}".replace(".", ",")


def _img_b64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode()


# ── Plotly base layout ────────────────────────────────────────────────────────
def _base_layout(title: str, height: int = 360) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=15, color=COLORS["text"], family="DM Sans, sans-serif"),
            x=0,
            xanchor="left",
            pad=dict(l=8, t=4),
        ),
        height=height,
        margin=dict(l=12, r=12, t=52, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="DM Sans, sans-serif", color=COLORS["text_2"], size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=12),
        ),
        xaxis=dict(
            gridcolor=COLORS["surface_3"],
            gridwidth=1,
            linecolor=COLORS["border_strong"],
            tickfont=dict(size=11, color=COLORS["text_3"]),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["surface_3"],
            gridwidth=1,
            linecolor=COLORS["border_strong"],
            tickfont=dict(size=11, color=COLORS["text_3"]),
            showgrid=True,
            zeroline=False,
        ),
    )


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Hero ──────────────────────────────────────────────────────────────────────
def render_header(df: pd.DataFrame) -> None:
    start = pd.Timestamp(df["Inicio"].min()).strftime("%d/%m/%Y")
    end = pd.Timestamp(df["Inicio"].max()).strftime("%d/%m/%Y")
    total = fmt_int(len(df))

    try:
        logo_tag = f'<img class="hero-logo" src="data:image/png;base64,{_img_b64(LOGO_PATH)}" alt="Vale" />'
    except Exception:
        logo_tag = "⛏"

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-left">
                <div class="hero-logo-wrap">{logo_tag}</div>
                <div class="hero-text">
                    <h1>Análise Exploratória de Dados</h1>
                    <p>Antecipação de alertas críticos em frota de mineração · Monitoramento de target_4h
                    com foco em prevalência do alvo, concentração operacional e qualidade de dados.</p>
                    <div class="hero-badge">● Dados de {start} até {end}</div>
                </div>
            </div>
            <div class="hero-meta">
                <div class="meta-label">Total de registros</div>
                <div class="meta-value">{total}</div>
                <div class="meta-label" style="margin-top:14px;">Projeto</div>
                <div class="meta-value">Vale · EDA</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── KPIs ──────────────────────────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    pos = int(df["target_4h"].sum()) if total else 0
    rate = (pos / total) * 100 if total else 0.0
    p95 = df["duracao_ciclo_min"].quantile(0.95)
    med = df["duracao_ciclo_min"].median()

    c = st.columns(6)
    c[0].metric("Registros", fmt_int(total))
    c[1].metric("Positivos", fmt_int(pos))
    c[2].metric("Taxa positivos", fmt_pct(rate))
    c[3].metric("Tags únicas", fmt_int(df["Tag"].nunique()))
    c[4].metric("Frotas", fmt_int(df["Frota"].nunique()))
    c[5].metric("P95 duração", fmt_float(p95, " min"))
    st.caption(f"Mediana de duração do ciclo: {fmt_float(med, ' min')}")


# ── Insights ──────────────────────────────────────────────────────────────────
def render_insights(df: pd.DataFrame) -> None:
    top_frota = df["Frota"].value_counts(dropna=True).head(1)
    top_classe = df["Classe"].value_counts(dropna=True).head(1)
    top_hour = df["hora_do_dia"].value_counts(dropna=True).head(1)
    null_tte = df["tte_horas"].isna().mean() * 100 if len(df) else 0

    items = [
        ("🚛", "Frota dominante", f"{top_frota.index[0]} · {fmt_int(top_frota.iloc[0])} ciclos"),
        ("🏷️", "Classe dominante", f"{top_classe.index[0]} · {fmt_int(top_classe.iloc[0])} ciclos"),
        ("🕐", "Horário de pico", f"{int(top_hour.index[0]):02d}h · {fmt_int(top_hour.iloc[0])} ciclos"),
        ("📊", "Cobertura TTE", f"{fmt_pct(100 - null_tte)} com tempo até evento"),
    ]

    cols = st.columns(4)
    for col, (icon, label, value) in zip(cols, items):
        col.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-icon">{icon}</div>
                <div class="insight-label">{label}</div>
                <div class="insight-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Overview ──────────────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame) -> None:
    # Donut
    target_counts = (
        df["target_4h"]
        .map({0: "Negativo", 1: "Positivo"})
        .value_counts()
        .rename_axis("target")
        .reset_index(name="registros")
    )
    fig_donut = px.pie(
        target_counts,
        names="target",
        values="registros",
        hole=0.68,
        color="target",
        color_discrete_map={
            "Negativo": COLORS["surface_3"],
            "Positivo": COLORS["blue"],
        },
        template=PLOTLY_TEMPLATE,
    )
    fig_donut.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(size=13, color="white", family="DM Sans"),
        marker=dict(line=dict(color="#fff", width=2)),
    )
    donut_layout = _base_layout("Distribuição do target 4h", height=380)
    donut_layout["showlegend"] = True
    donut_layout["legend"].update(orientation="h", y=-0.06, x=0.5, xanchor="center")
    fig_donut.update_layout(**donut_layout)
    # remove axes from pie
    fig_donut.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))

    # Volume mensal
    monthly = (
        df.groupby("mes", as_index=False)
        .agg(registros=("Id", "count"), positivos=("target_4h", "sum"))
        .sort_values("mes")
    )
    monthly["taxa_positivos"] = monthly["positivos"] / monthly["registros"] * 100

    fig_month = go.Figure()
    fig_month.add_bar(
        x=monthly["mes"],
        y=monthly["registros"],
        name="Registros",
        marker_color=COLORS["surface_3"],
        marker_line_color=COLORS["border_strong"],
        marker_line_width=0.5,
    )
    fig_month.add_trace(
        go.Scatter(
            x=monthly["mes"],
            y=monthly["taxa_positivos"],
            name="Taxa positivos (%)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2.5, dash="solid"),
            marker=dict(size=6, color=COLORS["blue"], line=dict(color="#fff", width=1.5)),
        )
    )
    layout = _base_layout("Volume mensal e prevalência do alvo", height=380)
    layout["yaxis2"] = dict(
        title="Taxa (%)",
        overlaying="y",
        side="right",
        showgrid=False,
        tickfont=dict(size=11, color=COLORS["text_3"]),
        zeroline=False,
    )
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12))
    fig_month.update_layout(**layout)

    c = st.columns([0.38, 0.62])
    with c[0]:
        chart(fig_donut)
    with c[1]:
        chart(fig_month)


# ── Operações ─────────────────────────────────────────────────────────────────
def render_operations(df: pd.DataFrame) -> None:
    top_frotas = df["Frota"].value_counts().head(10).reset_index()
    top_frotas.columns = ["Frota", "registros"]

    top_tags = df["Tag"].value_counts().head(15).sort_values().reset_index()
    top_tags.columns = ["Tag", "registros"]

    top_classes = df["Classe"].fillna("Sem classe").value_counts().head(12).reset_index()
    top_classes.columns = ["Classe", "registros"]

    # Frotas
    fig_frota = go.Figure(go.Bar(
        x=top_frotas["Frota"],
        y=top_frotas["registros"],
        marker_color=COLORS["blue"],
        marker_line_width=0,
        text=top_frotas["registros"].apply(fmt_int),
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_3"]),
    ))
    fig_frota.update_layout(**_base_layout("Top frotas por volume de ciclos"))

    # Tags
    fig_tag = go.Figure(go.Bar(
        x=top_tags["registros"],
        y=top_tags["Tag"],
        orientation="h",
        marker_color=COLORS["green"],
        marker_line_width=0,
    ))
    fig_tag.update_layout(**_base_layout("Top tags por volume"))

    # Classes
    fig_class = go.Figure(go.Bar(
        x=top_classes["Classe"],
        y=top_classes["registros"],
        marker_color=COLORS["text_3"],
        marker_line_width=0,
        text=top_classes["registros"].apply(fmt_int),
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_3"]),
    ))
    fig_class.update_layout(**_base_layout("Distribuição por classe de atividade"))

    c = st.columns(2)
    with c[0]:
        chart(fig_frota)
    with c[1]:
        chart(fig_tag)
    chart(fig_class)


# ── Temporal ──────────────────────────────────────────────────────────────────
def render_temporal(df: pd.DataFrame) -> None:
    hour_counts = (
        df["hora_do_dia"]
        .value_counts()
        .sort_index()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hour_counts.columns = ["hora", "registros"]

    # Heatmap
    heatmap_data = (
        df.pivot_table(
            index="dia_da_semana",
            columns="hora_do_dia",
            values="Id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=range(7), columns=range(24), fill_value=0)
        .rename(index=DAYS_PT)
    )

    # Ciclos por hora — area chart
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=hour_counts["hora"],
        y=hour_counts["registros"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["blue"], width=2),
        fillcolor="rgba(0,113,227,0.08)",
    ))
    fig_hour.update_layout(**_base_layout("Ciclos por hora do dia"))
    fig_hour.update_xaxes(tickvals=list(range(24)), ticktext=[f"{h:02d}h" for h in range(24)])

    # Heatmap
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Hora", y="Dia", color="Ciclos"),
        color_continuous_scale=["#f5f5f7", "#a1c4fd", "#0071e3"],
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    fig_heat.update_layout(
        **_base_layout("Mapa de calor operacional"),
        coloraxis_colorbar=dict(
            thickness=12,
            len=0.8,
            tickfont=dict(size=10),
        ),
    )
    fig_heat.update_xaxes(tickvals=list(range(24)), ticktext=[f"{h:02d}h" for h in range(24)])

    c = st.columns(2)
    with c[0]:
        chart(fig_hour)
    with c[1]:
        chart(fig_heat)


# ── Qualidade ──────────────────────────────────────────────────────────────────
def render_quality(df: pd.DataFrame) -> None:
    duration = df["duracao_ciclo_min"].dropna()
    p99 = duration.quantile(0.99)

    fig_hist = go.Figure(go.Histogram(
        x=duration.clip(lower=0, upper=p99),
        nbinsx=60,
        marker_color=COLORS["blue"],
        marker_line_color=COLORS["surface"],
        marker_line_width=0.5,
        opacity=0.85,
    ))
    fig_hist.update_layout(**_base_layout("Distribuição da duração de ciclo (min)"))
    fig_hist.update_xaxes(title_text="Duração (min)")
    fig_hist.update_yaxes(title_text="Frequência")

    missing = (
        (df.isna().mean() * 100)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Coluna", 0: "% nulos"})
    )
    missing.columns = ["Coluna", "% nulos"]
    missing["% nulos"] = missing["% nulos"].round(4)

    c = st.columns([0.55, 0.45])
    with c[0]:
        chart(fig_hist)
    with c[1]:
        st.markdown('<div style="margin-top:8px;font-size:0.82rem;font-weight:600;color:#6e6e73;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Nulidade por coluna (%)</div>', unsafe_allow_html=True)
        st.dataframe(missing, use_container_width=True, hide_index=True, height=320)


# ── Detalhe ────────────────────────────────────────────────────────────────────
def render_detail(df: pd.DataFrame) -> None:
    tag_summary = (
        df.groupby(["Frota", "Tipo", "Tag"], dropna=False)
        .agg(
            registros=("Id", "count"),
            positivos=("target_4h", "sum"),
            duracao_mediana_min=("duracao_ciclo_min", "median"),
            tte_mediano_horas=("tte_horas", "median"),
        )
        .reset_index()
        .sort_values(["positivos", "registros"], ascending=False)
    )
    tag_summary["taxa_positivos_%"] = (
        tag_summary["positivos"] / tag_summary["registros"] * 100
    ).round(2)
    tag_summary["duracao_mediana_min"] = tag_summary["duracao_mediana_min"].round(2)
    tag_summary["tte_mediano_horas"] = tag_summary["tte_mediano_horas"].round(2)

    st.markdown('<div style="font-size:0.82rem;font-weight:600;color:#6e6e73;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Resumo por Frota · Tipo · Tag</div>', unsafe_allow_html=True)
    st.dataframe(tag_summary, use_container_width=True, hide_index=True)

    sample = df[[
        "Id", "Inicio", "Fim", "Tag", "Frota", "Tipo",
        "Classe", "tte_horas", "target_4h",
    ]].sort_values("Inicio", ascending=False)

    st.markdown('<div style="font-size:0.82rem;font-weight:600;color:#6e6e73;text-transform:uppercase;letter-spacing:0.06em;margin:20px 0 8px;">Amostra · últimos 500 registros</div>', unsafe_allow_html=True)
    st.dataframe(sample.head(500), use_container_width=True, hide_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_theme()
    df = load_data()
    filtered = df

    render_header(df)

    if filtered.empty:
        st.warning("Nenhum registro encontrado na base analisada.")
        return

    render_kpis(filtered)

    st.markdown('<div class="section-title">Destaques</div>', unsafe_allow_html=True)
    render_insights(filtered)

    st.markdown('<div class="section-title">Análise</div>', unsafe_allow_html=True)
    tabs = st.tabs(["  Visão geral  ", "  Operação  ", "  Temporal  ", "  Qualidade  ", "  Detalhe  "])
    with tabs[0]:
        render_overview(filtered)
    with tabs[1]:
        render_operations(filtered)
    with tabs[2]:
        render_temporal(filtered)
    with tabs[3]:
        render_quality(filtered)
    with tabs[4]:
        render_detail(filtered)


if __name__ == "__main__":
    main()
