"""Executa a simulacao de cenario economico do piloto.

    python -m src.simulation

As entradas medidas vem do relatorio operacional; as premissas vem de faixas
de literatura publica e podem ser sobrescritas por argumento.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.evaluation.operational_scorecard import (
    PRIMARY_TOP_K,
    build_scored_frame,
    build_tag_day_panel,
)
from src.models.model_selection import SELECTED_MODEL_PATH
from src.models.train_model import load_splits, predict_scores, prepare_model_matrix
from src.simulation.economics import (
    DEFAULT_DOWNTIME_COST_RANGE,
    DEFAULT_DOWNTIME_HOURS,
    DEFAULT_INSPECTION_COST,
    DEFAULT_PREVENTION_RANGE,
    SIMULATION_DISCLAIMER,
    SIMULATION_GRID_PATH,
    SIMULATION_REPORT_PATH,
    Assumptions,
    MeasuredInputs,
    break_even_effectiveness,
    sensitivity_grid,
    simulate_scenario,
)
from src.utils.config import SPLIT_DIR
from src.utils.logging_config import get_logger, setup_logging
from src.utils.metadata import to_repo_relative_path

logger = get_logger(__name__)


def collect_measured_inputs(
    split_dir: Path = SPLIT_DIR,
    model_path: Path = SELECTED_MODEL_PATH,
    top_k: int = PRIMARY_TOP_K,
) -> MeasuredInputs:
    """Extrai do conjunto de teste os numeros efetivamente observados.

    A linha de base aleatoria nao e chutada: para cada dia, e o valor esperado
    de sortear `top_k` entre as Tags disponiveis naquele dia.
    """
    artifact = joblib.load(model_path)
    _, _, test = load_splits(split_dir)
    scores = predict_scores(
        artifact["model"], prepare_model_matrix(test, artifact["feature_columns"])
    )
    scored = build_scored_frame(test, scores, float(artifact["threshold"]), "test")
    panel = build_tag_day_panel(scored)

    tags_per_day = panel.groupby("data").size()
    positives_per_day = panel.groupby("data")["target_4h"].sum()
    selection_fraction = (top_k / tags_per_day).clip(upper=1.0)
    expected_random = float((positives_per_day * selection_fraction).sum())

    selected = panel.groupby("data", group_keys=False).head(top_k)
    caught = int(selected["target_4h"].sum())

    return MeasuredInputs(
        days=int(panel["data"].nunique()),
        cases_caught_by_model=caught,
        cases_caught_by_random=expected_random,
        inspections_per_day=top_k,
        total_critical_cases=int(panel["target_4h"].sum()),
    )


def run_simulation(
    downtime_cost: float | None = None,
    prevention: float | None = None,
    inspection_cost: float = DEFAULT_INSPECTION_COST,
    downtime_hours: float = DEFAULT_DOWNTIME_HOURS,
    report_path: Path = SIMULATION_REPORT_PATH,
    grid_path: Path = SIMULATION_GRID_PATH,
) -> dict[str, Any]:
    """Produz o cenario central, a superficie de sensibilidade e o ponto de equilibrio."""
    measured = collect_measured_inputs()

    central_cost = downtime_cost or float(sum(DEFAULT_DOWNTIME_COST_RANGE) / 2)
    central_prevention = prevention or float(sum(DEFAULT_PREVENTION_RANGE) / 2)
    central = simulate_scenario(
        measured,
        Assumptions(
            downtime_cost_per_hour=central_cost,
            prevention_effectiveness=central_prevention,
            inspection_cost=inspection_cost,
            downtime_hours_per_event=downtime_hours,
        ),
    )

    grid = sensitivity_grid(
        measured,
        inspection_cost=inspection_cost,
        downtime_hours=downtime_hours,
    )

    break_even = {
        f"custo_hora_{int(cost)}": round(
            break_even_effectiveness(measured, cost, inspection_cost, downtime_hours), 4
        )
        for cost in (
            DEFAULT_DOWNTIME_COST_RANGE[0],
            central_cost,
            DEFAULT_DOWNTIME_COST_RANGE[1],
        )
    }

    payload = central.as_dict()
    payload["ponto_de_equilibrio_eficacia"] = break_even
    payload["faixa_sensibilidade"] = {
        "custo_hora_parada_usd": list(DEFAULT_DOWNTIME_COST_RANGE),
        "eficacia_prevencao": list(DEFAULT_PREVENTION_RANGE),
        "beneficio_liquido_min_usd": float(grid["beneficio_liquido_usd"].min()),
        "beneficio_liquido_max_usd": float(grid["beneficio_liquido_usd"].max()),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    grid.insert(0, "aviso", "SIMULACAO")
    grid.to_csv(grid_path, index=False)

    payload["report_path"] = str(report_path)
    payload["grid_path"] = str(grid_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Simulacao economica do piloto operacional.")
    parser.add_argument("--custo-hora", type=float, default=None)
    parser.add_argument("--eficacia", type=float, default=None)
    parser.add_argument("--custo-inspecao", type=float, default=DEFAULT_INSPECTION_COST)
    parser.add_argument("--horas-parada", type=float, default=DEFAULT_DOWNTIME_HOURS)
    args = parser.parse_args(argv)

    result = run_simulation(
        downtime_cost=args.custo_hora,
        prevention=args.eficacia,
        inspection_cost=args.custo_inspecao,
        downtime_hours=args.horas_parada,
    )

    medido = result["entradas_medidas"]
    premissas = result["premissas_nao_medidas"]
    faixa = result["faixa_sensibilidade"]

    logger.warning(SIMULATION_DISCLAIMER)
    logger.info("--- MEDIDO no conjunto de teste ---")
    logger.info(f"Dias avaliados: {medido['dias_avaliados']}")
    logger.info(f"Casos capturados pelo modelo: {medido['casos_capturados_pelo_modelo']}")
    logger.info(f"Casos capturados por acaso: {medido['casos_capturados_por_acaso']}")
    logger.info(f"Casos adicionais antecipados: {medido['casos_adicionais']}")
    logger.info("--- PREMISSA, nao medido ---")
    for key, value in premissas.items():
        logger.info(f"{key}: {value}")
    logger.info("--- CENARIO SIMULADO ---")
    logger.info(f"Eventos evitados: {result['eventos_evitados_simulado']}")
    logger.info(f"Beneficio liquido: USD {result['beneficio_liquido_usd_simulado']:,.0f}")
    logger.info(
        f"Faixa na sensibilidade: USD {faixa['beneficio_liquido_min_usd']:,.0f} "
        f"a {faixa['beneficio_liquido_max_usd']:,.0f}"
    )
    logger.info("--- PONTO DE EQUILIBRIO ---")
    for key, value in result["ponto_de_equilibrio_eficacia"].items():
        logger.info(f"eficacia minima para empatar ({key}): {value:.1%}")
    logger.info(f"Relatorio: {to_repo_relative_path(Path(result['report_path']))}")
    logger.info(f"Sensibilidade: {to_repo_relative_path(Path(result['grid_path']))}")


if __name__ == "__main__":
    main()
