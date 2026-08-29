"""Geracao de historico de manutencao SINTETICO para o piloto de demonstracao.

ATENCAO -- ESTES DADOS SAO INVENTADOS.

O conjunto real nao possui registro de ordens de servico. Essa ausencia e uma
limitacao concreta do projeto: o modelo nao sabe se um equipamento acabou de
passar por manutencao, e por isso pode continuar apontando risco alto para uma
maquina que ja foi tratada.

Este modulo simula esse historico para demonstrar como o ciclo operacional
funcionaria **quando o dado existir**. Ele serve a duas perguntas:

1. Como a interface e o fluxo diario se comportam com o dado presente.
2. Quanto uma feature de "dias desde a ultima manutencao" mudaria o modelo.

Regras que tornam o dado impossivel de confundir com o real:

- Grava em `data/simulado/`, nunca em `data/processed/`.
- Todo arquivo leva o prefixo `SIMULADO_`.
- O dataframe carrega a coluna `origem_dado = "SIMULADO"` em toda linha.
- O modelo treinado com estas features NAO e o artefato promovido; ele vive em
  `models/demo/` e nao passa pelo gate de promocao.

O gerador imita padroes plausiveis -- manutencao preventiva em intervalo
regular, corretiva concentrada apos eventos criticos -- mas NAO foi calibrado
contra a realidade da operacao. Qualquer metrica obtida com ele mede a
arquitetura, e nao o desempenho que se obteria em campo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DATA_DIR, DEFAULT_RANDOM_STATE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SIMULATED_DIR = DATA_DIR / "simulado"
MAINTENANCE_PATH = SIMULATED_DIR / "SIMULADO_ordens_de_servico.parquet"

ORIGIN_MARKER = "SIMULADO"

# Intervalo nominal entre manutencoes preventivas, em dias.
PREVENTIVE_INTERVAL_DAYS = 30
PREVENTIVE_JITTER_DAYS = 7

# Fracao dos DIAS-COM-EVENTO que gera uma corretiva.
#
# A deduplicacao para Tag-dia e essencial e foi aprendida errando: amostrar
# sobre os eventos brutos gerava 37 mil ordens corretivas em seis meses, ou
# cerca de 200 por dia numa frota de 47 caminhoes. Os eventos criticos vem em
# rajada -- ha cerca de doze por Tag-dia -- e uma rajada corresponde a UM
# reparo, nao a doze. Sem a correcao, 70% dos ciclos apareciam com "corretiva
# recente", tornando a feature inutil por saturacao.
CORRECTIVE_RATE = 0.08
CORRECTIVE_DELAY_HOURS = (2, 48)


def generate_maintenance_orders(
    cycles: pd.DataFrame,
    critical_events: pd.DataFrame | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Gera ordens de servico sinteticas para as Tags e o periodo observados.

    Duas fontes de ordem, imitando a pratica de manutencao:

    - **Preventiva**: cada Tag recebe ordens em intervalo aproximadamente
      regular, com dispersao, a partir do inicio do periodo.
    - **Corretiva**: uma fracao dos eventos criticos gera ordem nas horas
      seguintes, o que aproxima o comportamento de "quebrou, consertou".
    """
    rng = np.random.default_rng(random_state)
    times = pd.to_datetime(cycles["Fim"], errors="coerce", utc=True).dropna()
    if times.empty:
        raise ValueError("Nao ha timestamps validos para gerar manutencao simulada.")

    start, end = times.min(), times.max()
    tags = sorted(cycles["Tag"].dropna().astype(str).unique())
    orders: list[dict[str, object]] = []

    for tag in tags:
        # A primeira preventiva cai em um ponto aleatorio do primeiro intervalo,
        # para que as Tags nao fiquem sincronizadas artificialmente.
        offset = float(rng.uniform(0, PREVENTIVE_INTERVAL_DAYS))
        moment = start + pd.Timedelta(offset, unit="D")
        while moment <= end:
            orders.append(
                {
                    "Tag": tag,
                    "data_manutencao": moment,
                    "tipo": "preventiva",
                    "origem_dado": ORIGIN_MARKER,
                }
            )
            step = PREVENTIVE_INTERVAL_DAYS + float(
                rng.uniform(-PREVENTIVE_JITTER_DAYS, PREVENTIVE_JITTER_DAYS)
            )
            moment = moment + pd.Timedelta(max(step, 1.0), unit="D")

    if critical_events is not None and not critical_events.empty:
        events = critical_events.copy()
        tag_column = "TAG" if "TAG" in events.columns else "Tag"
        events[tag_column] = events[tag_column].astype(str)
        events["EVENT_TIME"] = pd.to_datetime(events["EVENT_TIME"], errors="coerce", utc=True)
        events = events.dropna(subset=["EVENT_TIME"])

        # Uma rajada de eventos no mesmo Tag-dia corresponde a um unico reparo.
        events["_dia"] = events["EVENT_TIME"].dt.date
        episodes = events.groupby([tag_column, "_dia"], as_index=False)["EVENT_TIME"].max()

        selected = episodes.sample(
            n=max(int(len(episodes) * CORRECTIVE_RATE), 1), random_state=random_state
        )
        for _, event in selected.iterrows():
            delay = float(rng.uniform(*CORRECTIVE_DELAY_HOURS))
            orders.append(
                {
                    "Tag": event[tag_column],
                    "data_manutencao": event["EVENT_TIME"] + pd.Timedelta(delay, unit="h"),
                    "tipo": "corretiva",
                    "origem_dado": ORIGIN_MARKER,
                }
            )

    frame = pd.DataFrame(orders)
    frame = frame.loc[frame["data_manutencao"] <= end].sort_values(["Tag", "data_manutencao"])
    return frame.reset_index(drop=True)


def add_maintenance_features(
    cycles: pd.DataFrame,
    orders: pd.DataFrame,
    reference_col: str = "Fim",
) -> pd.DataFrame:
    """Anexa `dias_desde_ultima_manutencao`, usando apenas ordens ANTERIORES ao ciclo.

    A causalidade e a mesma exigida pelas features de alerta: uma ordem futura
    nao pode informar o ciclo presente. Ciclo sem manutencao anterior recebe
    `NaN`, e nao zero -- zero significaria "manutencao hoje", o oposto do que
    a ausencia de historico indica.
    """
    out = cycles.copy()
    out[reference_col] = pd.to_datetime(out[reference_col], errors="coerce", utc=True)
    out["dias_desde_ultima_manutencao"] = np.nan
    out["manutencao_corretiva_recente"] = 0.0

    if orders.empty:
        return out

    orders = orders.copy()
    orders["data_manutencao"] = pd.to_datetime(orders["data_manutencao"], errors="coerce", utc=True)
    orders = orders.dropna(subset=["data_manutencao"])

    for tag, group in out.groupby("Tag", sort=False):
        tag_orders = orders.loc[orders["Tag"].astype(str) == str(tag)].sort_values(
            "data_manutencao"
        )
        if tag_orders.empty:
            continue

        order_times = tag_orders["data_manutencao"].to_numpy()
        is_corrective = (tag_orders["tipo"] == "corretiva").to_numpy()
        row_times = group[reference_col].to_numpy()

        # `side="left"` garante que uma ordem no mesmo instante do ciclo nao
        # seja contada como passada.
        positions = np.searchsorted(order_times, row_times, side="left")
        has_history = positions > 0

        deltas = np.full(len(row_times), np.nan)
        previous = np.where(has_history, positions - 1, 0)
        deltas[has_history] = (
            row_times[has_history] - order_times[previous[has_history]]
        ) / np.timedelta64(1, "D")
        out.loc[group.index, "dias_desde_ultima_manutencao"] = deltas
        out.loc[group.index, "manutencao_corretiva_recente"] = np.where(
            has_history & is_corrective[previous] & (deltas <= 7), 1.0, 0.0
        )

    return out


def save_maintenance_orders(orders: pd.DataFrame, path: Path = MAINTENANCE_PATH) -> Path:
    """Persiste em `data/simulado/`, fora do caminho dos dados reais."""
    path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(path, index=False)
    return path
