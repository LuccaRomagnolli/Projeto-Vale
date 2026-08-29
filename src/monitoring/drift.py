"""Monitoramento de drift para o piloto operacional.

`docs/politica_promocao_modelo.md` condiciona o piloto a "monitoramento continuo
de drift, volume de alertas e segmentos raros". Nada disso existia implementado:
o texto descrevia uma exigencia sem contrapartida em codigo, e um modelo que
degradasse em producao so seria notado por inspecao manual dos relatorios.

O monitoramento compara um lote novo contra um **perfil de referencia** extraido
do treino no momento da promocao. A referencia e persistida como resumo
estatistico, e nao como copia dos dados: assim a comparacao continua possivel
mesmo onde o conjunto de treino nao esta disponivel.

Sinais acompanhados:

- **PSI por feature** -- mudanca no comportamento da frota invalida o que o
  modelo aprendeu.
- **PSI do score** -- detecta deslocamento mesmo quando nenhuma feature isolada
  se move o bastante para chamar atencao.
- **Volume de alertas** -- um pico satura a equipe de manutencao; uma queda
  brusca pode indicar modelo mudo, nao frota saudavel.
- **Cobertura por segmento** -- categoria sem representacao no treino nao tem
  previsao apoiada em nada aprendido.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import REPORTS_MONITORING_DIR

REFERENCE_PROFILE_PATH = REPORTS_MONITORING_DIR / "reference_profile.json"
DRIFT_REPORT_PATH = REPORTS_MONITORING_DIR / "drift_report.json"

# Faixas convencionais de PSI. Abaixo de 0.10 a distribuicao e considerada
# estavel; entre 0.10 e 0.25 merece acompanhamento; acima de 0.25 indica
# mudanca relevante o bastante para questionar o modelo.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

DEFAULT_BINS = 10
# Evita divisao por zero e log de zero quando um bin fica vazio de um dos lados.
EPSILON = 1e-6

SEGMENT_COLUMNS = ("Frota", "Tipo", "turno")

# Abaixo desta cardinalidade a feature e tratada como discreta: cada valor vira
# um bin. Bins por quantil sobre poucos valores distintos colapsam e inflam o
# PSI artificialmente.
DISCRETE_MAX_CARDINALITY = 25

# Features derivadas do proprio alvo ou da composicao do treino. Comparar seus
# VALORES entre treino e inferencia e apples-to-oranges por construcao: no
# treino elas vem de historico parcial (encoding out-of-fold) e na inferencia
# das estatisticas finais. O sinal util para elas e a distribuicao da categoria
# de origem, ja coberta pela cobertura por segmento.
DERIVED_ENCODING_COLUMNS = ("Classe_target_enc", "Tag_freq", "Operador_freq")


def _quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """Bordas de bin para a referencia.

    Features com poucos valores distintos recebem um bin por valor. Com bins
    por quantil, uma feature discreta faz as bordas colapsarem e o PSI dispara
    sem que exista deslocamento real -- o monitor passaria a acusar drift em
    toda execucao, e um alarme permanente e ignorado como se fosse ruido.
    """
    distinct = np.unique(values)
    if len(distinct) < 2:
        # Serie constante: um unico bin degenerado, tratado explicitamente.
        return [float(distinct[0]), float(distinct[0]) + EPSILON]

    if len(distinct) <= DISCRETE_MAX_CARDINALITY:
        midpoints = (distinct[:-1] + distinct[1:]) / 2.0
        return [-np.inf, *[float(m) for m in midpoints], np.inf]

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        return [float(edges[0]), float(edges[0]) + EPSILON]
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(edge) for edge in edges]


def _bin_proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), EPSILON)
    return np.clip(counts / total, EPSILON, None)


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    edges: list[float] | None = None,
    n_bins: int = DEFAULT_BINS,
) -> float:
    """PSI entre duas distribuicoes.

    Zero significa distribuicoes identicas; valores maiores indicam
    deslocamento. As bordas vem sempre da referencia, para que a comparacao
    entre execucoes diferentes use a mesma regua.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")

    bin_edges = edges if edges is not None else _quantile_edges(ref, n_bins)
    ref_pct = _bin_proportions(ref, bin_edges)
    cur_pct = _bin_proportions(cur, bin_edges)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float) -> str:
    """Traduz o PSI em faixa acionavel."""
    if np.isnan(psi):
        return "indisponivel"
    if psi < PSI_MODERATE:
        return "estavel"
    if psi < PSI_SIGNIFICANT:
        return "moderado"
    return "significativo"


@dataclass
class ReferenceProfile:
    """Resumo estatistico do treino, congelado no momento da promocao."""

    feature_edges: dict[str, list[float]] = field(default_factory=dict)
    feature_reference: dict[str, list[float]] = field(default_factory=dict)
    score_edges: list[float] = field(default_factory=list)
    score_reference: list[float] = field(default_factory=list)
    alert_rate: float = 0.0
    segment_values: dict[str, list[str]] = field(default_factory=dict)
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_edges": self.feature_edges,
            "feature_reference": self.feature_reference,
            "score_edges": self.score_edges,
            "score_reference": self.score_reference,
            "alert_rate": self.alert_rate,
            "segment_values": self.segment_values,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceProfile:
        return cls(
            feature_edges=payload.get("feature_edges", {}),
            feature_reference=payload.get("feature_reference", {}),
            score_edges=payload.get("score_edges", []),
            score_reference=payload.get("score_reference", []),
            alert_rate=float(payload.get("alert_rate", 0.0)),
            segment_values=payload.get("segment_values", {}),
            rows=int(payload.get("rows", 0)),
        )


def build_reference_profile(
    reference_df: pd.DataFrame,
    feature_columns: list[str],
    scores: np.ndarray | pd.Series | None = None,
    alert_rate: float = 0.0,
    n_bins: int = DEFAULT_BINS,
) -> ReferenceProfile:
    """Extrai o perfil de referencia a partir do conjunto de treino."""
    profile = ReferenceProfile(rows=int(len(reference_df)), alert_rate=float(alert_rate))

    for column in feature_columns:
        if column not in reference_df.columns or column in DERIVED_ENCODING_COLUMNS:
            continue
        values = pd.to_numeric(reference_df[column], errors="coerce").to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        edges = _quantile_edges(values, n_bins)
        profile.feature_edges[column] = edges
        profile.feature_reference[column] = _bin_proportions(values, edges).tolist()

    if scores is not None:
        score_values = np.asarray(scores, dtype=float)
        score_values = score_values[~np.isnan(score_values)]
        if score_values.size:
            profile.score_edges = _quantile_edges(score_values, n_bins)
            profile.score_reference = _bin_proportions(score_values, profile.score_edges).tolist()

    for column in SEGMENT_COLUMNS:
        if column in reference_df.columns:
            profile.segment_values[column] = sorted(
                reference_df[column].dropna().astype(str).unique().tolist()
            )

    return profile


def _psi_from_reference(
    reference_pct: list[float], current: np.ndarray, edges: list[float]
) -> float:
    ref_pct = np.clip(np.asarray(reference_pct, dtype=float), EPSILON, None)
    cur_pct = _bin_proportions(current, edges)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift(
    batch: pd.DataFrame,
    profile: ReferenceProfile,
    scores: np.ndarray | pd.Series | None = None,
    observed_alert_rate: float | None = None,
    alert_rate_tolerance: float = 0.5,
) -> dict[str, Any]:
    """Compara um lote contra a referencia e devolve os sinais de drift.

    `alert_rate_tolerance` e a variacao relativa aceita no volume de alertas:
    `0.5` significa que dobrar ou cair pela metade ja e sinalizado.
    """
    features: list[dict[str, Any]] = []
    for column, reference_pct in profile.feature_reference.items():
        if column not in batch.columns:
            features.append({"feature": column, "psi": float("nan"), "faixa": "ausente_no_lote"})
            continue
        values = pd.to_numeric(batch[column], errors="coerce").to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            features.append({"feature": column, "psi": float("nan"), "faixa": "indisponivel"})
            continue
        psi = _psi_from_reference(reference_pct, values, profile.feature_edges[column])
        features.append({"feature": column, "psi": psi, "faixa": classify_psi(psi)})

    features.sort(key=lambda item: (-1.0 if np.isnan(item["psi"]) else -item["psi"]))
    drifted = [f["feature"] for f in features if f["faixa"] == "significativo"]

    score_psi = float("nan")
    if scores is not None and profile.score_reference:
        score_values = np.asarray(scores, dtype=float)
        score_values = score_values[~np.isnan(score_values)]
        if score_values.size:
            score_psi = _psi_from_reference(
                profile.score_reference, score_values, profile.score_edges
            )

    alert_volume: dict[str, Any] = {"esperado": profile.alert_rate, "observado": None}
    if observed_alert_rate is not None:
        expected = profile.alert_rate
        relative = abs(observed_alert_rate - expected) / expected if expected > 0 else float("inf")
        alert_volume = {
            "esperado": expected,
            "observado": float(observed_alert_rate),
            "variacao_relativa": float(relative),
            "fora_da_tolerancia": bool(relative > alert_rate_tolerance),
        }

    segments: dict[str, Any] = {}
    for column, known in profile.segment_values.items():
        if column not in batch.columns:
            segments[column] = {"ausente_no_lote": True}
            continue
        present = set(batch[column].dropna().astype(str).unique())
        segments[column] = {
            # Categoria nova nao tem representacao no treino: a previsao para
            # ela nao se apoia em nada aprendido.
            "categorias_novas": sorted(present - set(known)),
            "categorias_ausentes": sorted(set(known) - present),
        }

    return {
        "rows": int(len(batch)),
        "features": features,
        "features_com_drift_significativo": drifted,
        "score_psi": score_psi,
        "score_faixa": classify_psi(score_psi),
        "volume_de_alertas": alert_volume,
        "segmentos": segments,
    }


def should_retrain(drift: dict[str, Any], max_drifted_features: int = 3) -> dict[str, Any]:
    """Aplica a regra de gatilho de retreino sobre os sinais coletados.

    A decisao e explicita e auditavel: cada motivo aparece nomeado, em vez de
    um veredito unico sem justificativa.
    """
    reasons: list[str] = []

    drifted = drift.get("features_com_drift_significativo", [])
    if len(drifted) > max_drifted_features:
        reasons.append(
            f"{len(drifted)} features com drift significativo "
            f"(limite {max_drifted_features}): {drifted[:5]}"
        )

    if drift.get("score_faixa") == "significativo":
        reasons.append(f"distribuicao do score deslocada (PSI={drift['score_psi']:.4f})")

    volume = drift.get("volume_de_alertas", {})
    if volume.get("fora_da_tolerancia"):
        reasons.append(
            f"volume de alertas fora da tolerancia: esperado {volume['esperado']:.4f}, "
            f"observado {volume['observado']:.4f}"
        )

    for column, info in drift.get("segmentos", {}).items():
        novas = info.get("categorias_novas") or []
        if novas:
            reasons.append(f"categorias novas em {column}: {novas[:5]}")

    return {"retreinar": bool(reasons), "motivos": reasons}


def save_reference_profile(profile: ReferenceProfile, path: Path = REFERENCE_PROFILE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load_reference_profile(path: Path = REFERENCE_PROFILE_PATH) -> ReferenceProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"Perfil de referencia ausente: {path}. "
            "Execute `python tasks.py monitor-baseline` apos promover um modelo."
        )
    return ReferenceProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
