"""Simulacao de cenario economico do piloto.

ATENCAO -- ESTE MODULO PRODUZ SIMULACAO, NAO MEDICAO.

O projeto mede com rigor **quantos casos criticos a lista diaria antecipa**:
366 de 413 no conjunto de teste, contra 146 esperados por escolha aleatoria.
Isso e resultado observado.

O que ele NAO mede, porque o dado nao existe no conjunto:

1. Se inspecionar evita o alerta. Nao ha registro de intervencoes, logo nao ha
   contrafactual: nunca observamos "esta maquina foi inspecionada e o Don't Go
   nao ocorreu". A eficacia da prevencao e premissa, nao resultado.
2. Quanto custa uma hora de parada e quanto custa uma inspecao. Nao ha dado
   financeiro no conjunto.

Este modulo torna essas premissas **entradas explicitas** e devolve uma
superficie de sensibilidade, em vez de um numero unico. A intencao e que
ninguem consiga citar um retorno sem citar junto de que suposicao ele veio.

Faixas de referencia publica usadas como padrao, todas para caminhoes
fora-de-estrada em mineracao:

- Parada nao planejada: US$ 5.000 a 20.000 por hora, dirigida pela producao
  perdida e nao pelo reparo em si. Ha fontes citando ate US$ 100.000/h quando a
  interrupcao se propaga pelo ciclo de lavra.
- Reparo emergencial custa cerca de 4,8x um reparo planejado.

Fontes:
- https://www.cummins.com/news/2021/03/23/reducing-machine-downtime-mining
- https://heavyvehicleinspection.com/fleet-management/uptime/mining-uptime-executive-brief
- https://www.maptrack.com/statistics/equipment-downtime-cost

Sao ordens de grandeza de literatura publica, NAO numeros da operacao da Vale.
Substitua-os pelos valores reais antes de qualquer decisao de investimento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import REPORTS_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SIMULATION_REPORT_PATH = REPORTS_DIR / "simulacao" / "cenario_economico.json"
SIMULATION_GRID_PATH = REPORTS_DIR / "simulacao" / "cenario_economico_sensibilidade.csv"

# Marca carregada por todo artefato deste modulo, para que nenhuma saida possa
# ser confundida com metrica medida.
SIMULATION_DISCLAIMER = (
    "SIMULACAO baseada em premissas fornecidas pelo usuario e em faixas de "
    "literatura publica. NAO e medicao da operacao. A eficacia da prevencao NAO "
    "foi observada: o conjunto de dados nao possui registro de intervencoes."
)

# Faixas padrao, em USD. Amplas de proposito: o ponto do simulador e mostrar
# como o resultado varia, nao sugerir que um valor central e o correto.
DEFAULT_DOWNTIME_COST_RANGE = (5_000.0, 20_000.0)
DEFAULT_PREVENTION_RANGE = (0.10, 0.60)
DEFAULT_INSPECTION_COST = 400.0
DEFAULT_DOWNTIME_HOURS = 6.0


@dataclass(frozen=True)
class Assumptions:
    """Premissas do cenario. Nenhuma delas foi medida neste projeto."""

    downtime_cost_per_hour: float
    prevention_effectiveness: float
    inspection_cost: float
    downtime_hours_per_event: float

    def as_dict(self) -> dict[str, float]:
        return {
            "custo_hora_parada_usd": self.downtime_cost_per_hour,
            "eficacia_prevencao": self.prevention_effectiveness,
            "custo_inspecao_usd": self.inspection_cost,
            "horas_parada_por_evento": self.downtime_hours_per_event,
        }


@dataclass(frozen=True)
class MeasuredInputs:
    """Entradas efetivamente observadas no conjunto de teste.

    Estes sao os unicos numeros deste modulo que vem de medicao.
    """

    days: int
    cases_caught_by_model: int
    cases_caught_by_random: float
    inspections_per_day: int
    total_critical_cases: int
    source: str = "reports/operational/operational_metrics_report.json"

    @property
    def additional_cases(self) -> float:
        """Casos antecipados a mais que a escolha aleatoria -- resultado medido."""
        return self.cases_caught_by_model - self.cases_caught_by_random

    def as_dict(self) -> dict[str, Any]:
        return {
            "dias_avaliados": self.days,
            "casos_capturados_pelo_modelo": self.cases_caught_by_model,
            "casos_capturados_por_acaso": round(self.cases_caught_by_random, 1),
            "casos_adicionais": round(self.additional_cases, 1),
            "inspecoes_por_dia": self.inspections_per_day,
            "casos_criticos_totais": self.total_critical_cases,
            "origem": self.source,
        }


@dataclass
class ScenarioResult:
    """Resultado de um cenario, sempre acompanhado das premissas que o geraram."""

    assumptions: Assumptions
    measured: MeasuredInputs
    events_avoided: float
    downtime_hours_avoided: float
    gross_benefit: float
    inspection_cost_total: float
    net_benefit: float
    payback_ratio: float
    horizon_days: int = field(default=30)

    def as_dict(self) -> dict[str, Any]:
        return {
            "_aviso": SIMULATION_DISCLAIMER,
            "premissas_nao_medidas": self.assumptions.as_dict(),
            "entradas_medidas": self.measured.as_dict(),
            "horizonte_dias": self.horizon_days,
            "eventos_evitados_simulado": round(self.events_avoided, 1),
            "horas_de_parada_evitadas_simulado": round(self.downtime_hours_avoided, 1),
            "beneficio_bruto_usd_simulado": round(self.gross_benefit, 2),
            "custo_inspecoes_usd": round(self.inspection_cost_total, 2),
            "beneficio_liquido_usd_simulado": round(self.net_benefit, 2),
            "razao_retorno_simulado": round(self.payback_ratio, 2),
        }


def simulate_scenario(
    measured: MeasuredInputs,
    assumptions: Assumptions,
) -> ScenarioResult:
    """Projeta um cenario a partir dos casos antecipados medidos.

    A cadeia e explicita, e cada elo depois do primeiro depende de premissa:

        casos antecipados a mais   (MEDIDO)
          x eficacia da prevencao  (PREMISSA -- nunca observada)
          = eventos evitados
          x horas de parada        (PREMISSA)
          x custo por hora         (PREMISSA)
          = beneficio bruto
          - custo das inspecoes    (PREMISSA)
          = beneficio liquido
    """
    events_avoided = measured.additional_cases * assumptions.prevention_effectiveness
    downtime_hours_avoided = events_avoided * assumptions.downtime_hours_per_event
    gross_benefit = downtime_hours_avoided * assumptions.downtime_cost_per_hour

    total_inspections = measured.inspections_per_day * measured.days
    inspection_cost_total = total_inspections * assumptions.inspection_cost

    net_benefit = gross_benefit - inspection_cost_total
    payback = gross_benefit / inspection_cost_total if inspection_cost_total else float("inf")

    return ScenarioResult(
        assumptions=assumptions,
        measured=measured,
        events_avoided=events_avoided,
        downtime_hours_avoided=downtime_hours_avoided,
        gross_benefit=gross_benefit,
        inspection_cost_total=inspection_cost_total,
        net_benefit=net_benefit,
        payback_ratio=payback,
        horizon_days=measured.days,
    )


def sensitivity_grid(
    measured: MeasuredInputs,
    downtime_cost_range: tuple[float, float] = DEFAULT_DOWNTIME_COST_RANGE,
    prevention_range: tuple[float, float] = DEFAULT_PREVENTION_RANGE,
    inspection_cost: float = DEFAULT_INSPECTION_COST,
    downtime_hours: float = DEFAULT_DOWNTIME_HOURS,
    steps: int = 5,
) -> pd.DataFrame:
    """Varre as duas premissas mais incertas e devolve a superficie de resultado.

    A saida e uma superficie justamente para impedir a leitura "o sistema
    economiza X": o resultado depende de duas suposicoes que o projeto nao
    mediu, e a tabela deixa essa dependencia visivel.
    """
    rows = []
    for cost in np.linspace(*downtime_cost_range, steps):
        for effectiveness in np.linspace(*prevention_range, steps):
            result = simulate_scenario(
                measured,
                Assumptions(
                    downtime_cost_per_hour=float(cost),
                    prevention_effectiveness=float(effectiveness),
                    inspection_cost=inspection_cost,
                    downtime_hours_per_event=downtime_hours,
                ),
            )
            rows.append(
                {
                    "custo_hora_parada_usd": round(float(cost), 2),
                    "eficacia_prevencao": round(float(effectiveness), 3),
                    "eventos_evitados": round(result.events_avoided, 1),
                    "beneficio_bruto_usd": round(result.gross_benefit, 2),
                    "custo_inspecoes_usd": round(result.inspection_cost_total, 2),
                    "beneficio_liquido_usd": round(result.net_benefit, 2),
                    "razao_retorno": round(result.payback_ratio, 2),
                }
            )
    return pd.DataFrame(rows)


def break_even_effectiveness(
    measured: MeasuredInputs,
    downtime_cost_per_hour: float,
    inspection_cost: float = DEFAULT_INSPECTION_COST,
    downtime_hours: float = DEFAULT_DOWNTIME_HOURS,
) -> float:
    """Eficacia minima de prevencao para o piloto se pagar.

    E a pergunta mais util do simulador, porque inverte a incognita: em vez de
    afirmar um retorno, diz o quanto a inspecao precisaria funcionar para
    empatar. Esse patamar pode ser confrontado com a experiencia da equipe de
    manutencao, que nao precisa de dado financeiro para opinar.
    """
    total_cost = measured.inspections_per_day * measured.days * inspection_cost
    benefit_per_unit_effectiveness = (
        measured.additional_cases * downtime_hours * downtime_cost_per_hour
    )
    if benefit_per_unit_effectiveness <= 0:
        return float("inf")
    return total_cost / benefit_per_unit_effectiveness
