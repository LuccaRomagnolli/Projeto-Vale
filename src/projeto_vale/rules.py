from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import pandas as pd


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_critical_rules(
    rules_df: pd.DataFrame,
    critical_level: str = "muito alto",
) -> pd.DataFrame:
    """Filtra e prepara regras criticas para matching por texto normalizado."""
    rules = rules_df.copy()

    if "NIVEL" not in rules.columns:
        return pd.DataFrame(columns=["rule_evento", "rule_nivel", "rule_tipo", "rule_situacao"])

    rules["_nivel_norm"] = rules["NIVEL"].map(_normalize_text)
    rules = rules[rules["_nivel_norm"] == _normalize_text(critical_level)].copy()

    if rules.empty:
        return pd.DataFrame(columns=["rule_evento", "rule_nivel", "rule_tipo", "rule_situacao"])

    rules["rule_evento"] = rules.get("EVENTO", "").map(_normalize_text)
    rules["rule_nivel"] = rules.get("NIVEL", "").map(_normalize_text)
    rules["rule_tipo"] = rules.get("TIPO", "").map(_normalize_text)
    rules["rule_situacao"] = rules.get("SITUACAO", "").map(_normalize_text)

    keep = ["rule_evento", "rule_nivel", "rule_tipo", "rule_situacao"]
    return rules[keep].drop_duplicates().reset_index(drop=True)


def apply_critical_rules(
    telemetria_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    critical_level: str = "muito alto",
) -> pd.DataFrame:
    """Rotula eventos de telemetria como alerta critico via regras de negocio.

    Se o matching por regra nao for viavel para todas as colunas, usa fallback
    em `Is_Dont_Go` quando essa coluna existir.
    """
    df = telemetria_df.copy()

    event_col = _first_existing_column(df, ["EVENTO", "Evento", "Alarme", "Name"])
    level_col = _first_existing_column(df, ["NIVEL", "Nivel", "Criticidade"])
    type_col = _first_existing_column(df, ["TIPO", "Tipo"])
    situ_col = _first_existing_column(df, ["SITUACAO", "Situacao", "Classe"])

    df["_evento_norm"] = df[event_col].map(_normalize_text) if event_col else ""
    df["_nivel_norm"] = df[level_col].map(_normalize_text) if level_col else ""
    df["_tipo_norm"] = df[type_col].map(_normalize_text) if type_col else ""
    df["_situacao_norm"] = df[situ_col].map(_normalize_text) if situ_col else ""

    rules = prepare_critical_rules(rules_df, critical_level=critical_level)

    if not rules.empty and event_col and level_col:
        match_cols_left = ["_evento_norm", "_nivel_norm"]
        match_cols_right = ["rule_evento", "rule_nivel"]

        if type_col and rules["rule_tipo"].ne("").any():
            match_cols_left.append("_tipo_norm")
            match_cols_right.append("rule_tipo")
        if situ_col and rules["rule_situacao"].ne("").any():
            match_cols_left.append("_situacao_norm")
            match_cols_right.append("rule_situacao")

        joined = df.merge(
            rules,
            left_on=match_cols_left,
            right_on=match_cols_right,
            how="left",
            indicator=True,
        )
        by_rule = joined["_merge"].eq("both")
        label_source = pd.Series("rule_match", index=df.index)
    else:
        by_rule = pd.Series(False, index=df.index)
        label_source = pd.Series("no_rule_match", index=df.index)

    if "Is_Dont_Go" in df.columns:
        fallback = pd.to_numeric(df["Is_Dont_Go"], errors="coerce").fillna(0).astype(int).eq(1)
        used_fallback = ~by_rule & fallback
        label_source.loc[used_fallback] = "is_dont_go_fallback"
        is_critical = by_rule | fallback
    else:
        is_critical = by_rule

    out = df.copy()
    out["is_critical_alert"] = is_critical.astype(int)
    out["label_source"] = label_source
    return out


def extract_critical_alert_events(
    labeled_telemetria: pd.DataFrame,
    tag_col: str = "TAG",
    time_col: str = "Data_Evento",
) -> pd.DataFrame:
    """Extrai eventos criticos com colunas padronizadas para construcao de target."""
    if tag_col not in labeled_telemetria.columns:
        raise ValueError(f"Coluna de tag nao encontrada: {tag_col}")
    if time_col not in labeled_telemetria.columns:
        raise ValueError(f"Coluna de tempo nao encontrada: {time_col}")

    events = labeled_telemetria.loc[
        labeled_telemetria["is_critical_alert"].eq(1),
        [tag_col, time_col],
    ].copy()
    events = events.rename(columns={tag_col: "tag", time_col: "alert_time"})
    events["alert_time"] = pd.to_datetime(events["alert_time"], errors="coerce")
    events = events.dropna(subset=["tag", "alert_time"])
    events = events.sort_values(["tag", "alert_time"]).drop_duplicates()
    return events.reset_index(drop=True)
