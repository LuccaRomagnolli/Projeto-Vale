"""Simulacao de cenario: premissas explicitas e dado sintetico inconfundivel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.simulation.economics import (
    SIMULATION_DISCLAIMER,
    Assumptions,
    MeasuredInputs,
    break_even_effectiveness,
    sensitivity_grid,
    simulate_scenario,
)
from src.simulation.maintenance import (
    ORIGIN_MARKER,
    add_maintenance_features,
    generate_maintenance_orders,
    save_maintenance_orders,
)


def _measured(**overrides) -> MeasuredInputs:
    defaults = {
        "days": 30,
        "cases_caught_by_model": 366,
        "cases_caught_by_random": 146.0,
        "inspections_per_day": 15,
        "total_critical_cases": 413,
    }
    defaults.update(overrides)
    return MeasuredInputs(**defaults)


def _assumptions(**overrides) -> Assumptions:
    defaults = {
        "downtime_cost_per_hour": 12_500.0,
        "prevention_effectiveness": 0.35,
        "inspection_cost": 400.0,
        "downtime_hours_per_event": 6.0,
    }
    defaults.update(overrides)
    return Assumptions(**defaults)


def _cycles(n_rows: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tag": [f"CA{i % 4}" for i in range(n_rows)],
            "Fim": pd.date_range("2026-01-01", periods=n_rows, freq="6h", tz="UTC"),
        }
    )


# --- separacao entre medido e suposto -------------------------------------


def test_result_always_carries_its_assumptions() -> None:
    """Nenhum numero simulado pode circular sem a premissa que o gerou."""
    payload = simulate_scenario(_measured(), _assumptions()).as_dict()

    assert "premissas_nao_medidas" in payload
    assert "entradas_medidas" in payload
    assert payload["_aviso"] == SIMULATION_DISCLAIMER
    assert "SIMULACAO" in payload["_aviso"]


def test_simulated_fields_are_named_as_simulated() -> None:
    payload = simulate_scenario(_measured(), _assumptions()).as_dict()
    monetarios = [k for k in payload if "usd" in k and "custo_inspecoes" not in k]
    assert monetarios, "esperava campos monetarios"
    for key in monetarios:
        assert "simulado" in key, f"{key} nao se identifica como simulado"


def test_additional_cases_is_the_only_measured_driver() -> None:
    """O unico elo medido da cadeia e a diferenca contra o acaso."""
    measured = _measured(cases_caught_by_model=366, cases_caught_by_random=146.0)
    assert measured.additional_cases == pytest.approx(220.0)


# --- comportamento economico ----------------------------------------------


def test_benefit_scales_linearly_with_effectiveness() -> None:
    dobro = simulate_scenario(_measured(), _assumptions(prevention_effectiveness=0.40))
    metade = simulate_scenario(_measured(), _assumptions(prevention_effectiveness=0.20))
    assert dobro.gross_benefit == pytest.approx(2 * metade.gross_benefit)


def test_zero_effectiveness_makes_the_pilot_a_pure_cost() -> None:
    """Se a inspecao nao evita nada, so restam as despesas."""
    result = simulate_scenario(_measured(), _assumptions(prevention_effectiveness=0.0))
    assert result.gross_benefit == 0.0
    assert result.net_benefit < 0


def test_break_even_answers_how_well_inspection_must_work() -> None:
    """Inverter a incognita e o uso mais defensavel do simulador."""
    measured = _measured()
    limiar = break_even_effectiveness(measured, downtime_cost_per_hour=12_500.0)

    no_limiar = simulate_scenario(measured, _assumptions(prevention_effectiveness=limiar))
    assert no_limiar.net_benefit == pytest.approx(0.0, abs=1.0)


def test_break_even_falls_as_downtime_gets_more_expensive() -> None:
    measured = _measured()
    barato = break_even_effectiveness(measured, 5_000.0)
    caro = break_even_effectiveness(measured, 20_000.0)
    assert caro < barato


def test_sensitivity_grid_exposes_the_full_range() -> None:
    """A saida e superficie justamente para impedir a leitura de numero unico."""
    grid = sensitivity_grid(_measured(), steps=4)
    assert len(grid) == 16
    assert grid["beneficio_liquido_usd"].nunique() > 1
    assert grid["beneficio_liquido_usd"].max() > grid["beneficio_liquido_usd"].min()


# --- dado sintetico -------------------------------------------------------


def test_every_synthetic_row_is_marked(tmp_path: Path) -> None:
    """O marcador de origem impede confusao com dado da operacao."""
    orders = generate_maintenance_orders(_cycles())
    assert (orders["origem_dado"] == ORIGIN_MARKER).all()

    path = save_maintenance_orders(orders, tmp_path / "SIMULADO_ordens.parquet")
    assert "SIMULADO" in path.name
    assert (pd.read_parquet(path)["origem_dado"] == ORIGIN_MARKER).all()


def test_generated_volume_is_plausible_for_the_fleet() -> None:
    """Ordens por Tag por mes precisa ficar em faixa operacionalmente critivel.

    A primeira versao do gerador amostrava sobre eventos brutos e produzia
    cerca de 200 ordens corretivas por dia numa frota de 47 equipamentos.
    """
    cycles = _cycles(2000)  # 4 tags, 500 dias
    events = pd.DataFrame(
        {
            "TAG": [f"CA{i % 4}" for i in range(3000)],
            "EVENT_TIME": pd.date_range("2026-01-01", periods=3000, freq="4h", tz="UTC"),
        }
    )
    orders = generate_maintenance_orders(cycles, events)

    tags = cycles["Tag"].nunique()
    meses = (orders["data_manutencao"].max() - orders["data_manutencao"].min()).days / 30
    por_tag_mes = len(orders) / tags / meses
    assert 0.5 < por_tag_mes < 8, f"{por_tag_mes:.1f} ordens/Tag/mes e implausivel"


def test_maintenance_feature_uses_only_past_orders() -> None:
    """Ordem futura nao pode informar o ciclo presente."""
    cycles = pd.DataFrame(
        {
            "Tag": ["CA0", "CA0"],
            "Fim": pd.to_datetime(["2026-01-10", "2026-01-20"], utc=True),
        }
    )
    orders = pd.DataFrame(
        {
            "Tag": ["CA0", "CA0"],
            "data_manutencao": pd.to_datetime(["2026-01-05", "2026-01-30"], utc=True),
            "tipo": ["preventiva", "preventiva"],
            "origem_dado": [ORIGIN_MARKER] * 2,
        }
    )

    out = add_maintenance_features(cycles, orders)

    # ambos os ciclos so podem enxergar a ordem de 05/01
    assert out["dias_desde_ultima_manutencao"].tolist() == pytest.approx([5.0, 15.0])


def test_cycle_without_prior_maintenance_is_null_not_zero() -> None:
    """Zero significaria manutencao hoje -- o oposto de nao ter historico."""
    cycles = pd.DataFrame({"Tag": ["CA0"], "Fim": pd.to_datetime(["2026-01-01"], utc=True)})
    orders = pd.DataFrame(
        {
            "Tag": ["CA0"],
            "data_manutencao": pd.to_datetime(["2026-02-01"], utc=True),
            "tipo": ["preventiva"],
            "origem_dado": [ORIGIN_MARKER],
        }
    )

    out = add_maintenance_features(cycles, orders)
    assert np.isnan(out["dias_desde_ultima_manutencao"].iloc[0])


def test_generator_rejects_input_without_valid_timestamps() -> None:
    empty = pd.DataFrame({"Tag": ["CA0"], "Fim": [pd.NaT]})
    with pytest.raises(ValueError, match="timestamps validos"):
        generate_maintenance_orders(empty)


# --- ponto de entrada -----------------------------------------------------


def test_measured_inputs_come_from_the_real_test_split(tmp_path: Path, split_dir, artifact_factory):
    """A linha de base aleatoria e calculada, nao arbitrada."""
    import joblib
    from src.simulation.__main__ import collect_measured_inputs

    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact_factory(feature_columns=["n_alertas_4h"]), model_path)

    measured = collect_measured_inputs(split_dir=split_dir, model_path=model_path, top_k=3)

    assert measured.days > 0
    assert measured.inspections_per_day == 3
    assert measured.total_critical_cases > 0
    # o modelo nunca pode capturar mais casos do que existem
    assert measured.cases_caught_by_model <= measured.total_critical_cases
    # a linha de base aleatoria tambem nao
    assert 0 <= measured.cases_caught_by_random <= measured.total_critical_cases


def test_random_baseline_reflects_the_daily_selection_fraction(
    tmp_path, split_dir, artifact_factory
):
    """Selecionar mais Tags por dia captura mais casos por acaso."""
    import joblib
    from src.simulation.__main__ import collect_measured_inputs

    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact_factory(feature_columns=["n_alertas_4h"]), model_path)

    poucos = collect_measured_inputs(split_dir=split_dir, model_path=model_path, top_k=1)
    muitos = collect_measured_inputs(split_dir=split_dir, model_path=model_path, top_k=4)

    assert muitos.cases_caught_by_random > poucos.cases_caught_by_random


def test_run_simulation_persists_report_and_grid(
    tmp_path, split_dir, artifact_factory, monkeypatch
):
    """Os dois artefatos precisam existir e carregar a marca de simulacao."""
    import json

    import joblib
    import src.simulation.__main__ as entry

    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact_factory(feature_columns=["n_alertas_4h"]), model_path)
    monkeypatch.setattr(entry, "SELECTED_MODEL_PATH", model_path)
    monkeypatch.setattr(entry, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(entry, "PRIMARY_TOP_K", 3)

    report = tmp_path / "cenario.json"
    grid = tmp_path / "sensibilidade.csv"
    result = entry.run_simulation(report_path=report, grid_path=grid)

    assert report.exists() and grid.exists()
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert "SIMULACAO" in saved["_aviso"]
    assert "ponto_de_equilibrio_eficacia" in result

    frame = pd.read_csv(grid)
    assert (frame["aviso"] == "SIMULACAO").all(), "cada linha do CSV precisa se declarar simulada"


def test_simulation_reports_a_range_not_a_single_number(
    tmp_path, split_dir, artifact_factory, monkeypatch
):
    import joblib
    import src.simulation.__main__ as entry

    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact_factory(feature_columns=["n_alertas_4h"]), model_path)
    monkeypatch.setattr(entry, "SELECTED_MODEL_PATH", model_path)
    monkeypatch.setattr(entry, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(entry, "PRIMARY_TOP_K", 3)

    result = entry.run_simulation(report_path=tmp_path / "c.json", grid_path=tmp_path / "g.csv")
    faixa = result["faixa_sensibilidade"]
    assert faixa["beneficio_liquido_max_usd"] > faixa["beneficio_liquido_min_usd"]
