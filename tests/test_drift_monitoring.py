"""Monitoramento de drift: os sinais precisam disparar quando devem."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift import (
    PSI_SIGNIFICANT,
    ReferenceProfile,
    build_reference_profile,
    classify_psi,
    compute_drift,
    load_reference_profile,
    population_stability_index,
    save_reference_profile,
    should_retrain,
)

FEATURES = ["n_alertas_4h", "duracao_ciclo_min"]


def _frame(n: int = 500, shift: float = 0.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "n_alertas_4h": rng.normal(5 + shift, 1.5, n),
            "duracao_ciclo_min": rng.normal(20 + shift, 4.0, n),
            "Frota": rng.choice(["F1", "F2"], n),
            "Tipo": rng.choice(["Caminhao"], n),
            "turno": rng.choice(["manha", "tarde"], n),
        }
    )


# --- PSI ------------------------------------------------------------------


def test_identical_distributions_have_near_zero_psi() -> None:
    values = np.random.default_rng(1).normal(0, 1, 2000)
    assert population_stability_index(values, values) < 1e-9


def test_shifted_distribution_produces_significant_psi() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 3000)
    shifted = rng.normal(3, 1, 3000)  # deslocamento de 3 desvios
    assert population_stability_index(reference, shifted) > PSI_SIGNIFICANT


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 3000)
    small = population_stability_index(reference, rng.normal(0.3, 1, 3000))
    large = population_stability_index(reference, rng.normal(2.0, 1, 3000))
    assert large > small


def test_psi_handles_empty_input_without_crashing() -> None:
    assert np.isnan(population_stability_index([], [1.0, 2.0]))
    assert np.isnan(population_stability_index([1.0, 2.0], []))


def test_psi_handles_a_constant_series() -> None:
    """Serie sem variacao nao pode gerar divisao por zero nem log de zero."""
    constant = np.full(100, 7.0)
    value = population_stability_index(constant, constant)
    assert np.isfinite(value)


def test_classification_covers_every_band() -> None:
    assert classify_psi(0.05) == "estavel"
    assert classify_psi(0.15) == "moderado"
    assert classify_psi(0.40) == "significativo"
    assert classify_psi(float("nan")) == "indisponivel"


# --- perfil de referencia -------------------------------------------------


def test_profile_round_trips_through_disk(tmp_path: Path) -> None:
    profile = build_reference_profile(_frame(), FEATURES, scores=np.linspace(0, 1, 500))
    path = save_reference_profile(profile, tmp_path / "perfil.json")
    restored = load_reference_profile(path)

    assert restored.rows == profile.rows
    assert set(restored.feature_reference) == set(profile.feature_reference)
    assert restored.segment_values == profile.segment_values


def test_missing_profile_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="monitor-baseline"):
        load_reference_profile(tmp_path / "ausente.json")


def test_profile_stores_summary_not_raw_data(tmp_path: Path) -> None:
    """A referencia precisa funcionar onde o conjunto de treino nao existe."""
    profile = build_reference_profile(_frame(2000), FEATURES)
    payload = json.loads(save_reference_profile(profile, tmp_path / "p.json").read_text())
    # nenhuma linha bruta; apenas bordas e proporcoes por bin
    assert set(payload["feature_reference"]) == set(FEATURES)
    assert all(len(v) <= 12 for v in payload["feature_reference"].values())


# --- deteccao de drift ----------------------------------------------------


def test_stable_batch_is_not_flagged() -> None:
    profile = build_reference_profile(_frame(seed=3), FEATURES)
    drift = compute_drift(_frame(seed=4), profile)

    assert drift["features_com_drift_significativo"] == []
    assert should_retrain(drift)["retreinar"] is False


def test_shifted_batch_is_flagged() -> None:
    profile = build_reference_profile(_frame(2000, seed=3), FEATURES)
    drift = compute_drift(_frame(2000, shift=6.0, seed=4), profile)

    assert set(drift["features_com_drift_significativo"]) == set(FEATURES)


def test_feature_absent_from_the_batch_is_reported() -> None:
    profile = build_reference_profile(_frame(), FEATURES)
    drift = compute_drift(_frame().drop(columns=["n_alertas_4h"]), profile)

    faixas = {f["feature"]: f["faixa"] for f in drift["features"]}
    assert faixas["n_alertas_4h"] == "ausente_no_lote"


def test_alert_volume_outside_tolerance_is_flagged() -> None:
    profile = build_reference_profile(_frame(), FEATURES, alert_rate=0.20)
    drift = compute_drift(_frame(), profile, observed_alert_rate=0.60)  # triplicou

    assert drift["volume_de_alertas"]["fora_da_tolerancia"] is True
    assert should_retrain(drift)["retreinar"] is True


def test_alert_volume_within_tolerance_is_not_flagged() -> None:
    profile = build_reference_profile(_frame(), FEATURES, alert_rate=0.20)
    drift = compute_drift(_frame(), profile, observed_alert_rate=0.24)
    assert drift["volume_de_alertas"]["fora_da_tolerancia"] is False


def test_new_segment_category_is_detected() -> None:
    """Categoria nova nao tem previsao apoiada em nada aprendido."""
    profile = build_reference_profile(_frame(), FEATURES)
    batch = _frame()
    batch["Frota"] = "FROTA_NUNCA_VISTA"

    drift = compute_drift(batch, profile)

    assert drift["segmentos"]["Frota"]["categorias_novas"] == ["FROTA_NUNCA_VISTA"]
    assert should_retrain(drift)["retreinar"] is True


def test_score_distribution_shift_is_detected() -> None:
    profile = build_reference_profile(
        _frame(2000), FEATURES, scores=np.random.default_rng(1).beta(2, 8, 2000)
    )
    drift = compute_drift(_frame(2000), profile, scores=np.random.default_rng(2).beta(8, 2, 2000))
    assert drift["score_faixa"] == "significativo"


# --- gatilho de retreino --------------------------------------------------


def test_trigger_names_every_reason() -> None:
    """A decisao precisa ser auditavel, nao um veredito sem justificativa."""
    drift = {
        "features_com_drift_significativo": ["a", "b", "c", "d"],
        "score_faixa": "significativo",
        "score_psi": 0.5,
        "volume_de_alertas": {"fora_da_tolerancia": True, "esperado": 0.2, "observado": 0.9},
        "segmentos": {"Frota": {"categorias_novas": ["X"]}},
    }
    verdict = should_retrain(drift)

    assert verdict["retreinar"] is True
    assert len(verdict["motivos"]) == 4


def test_trigger_tolerates_a_few_drifted_features() -> None:
    drift = {
        "features_com_drift_significativo": ["a", "b"],
        "score_faixa": "estavel",
        "volume_de_alertas": {},
        "segmentos": {},
    }
    assert should_retrain(drift)["retreinar"] is False


def test_empty_profile_does_not_crash_the_check() -> None:
    drift = compute_drift(_frame(), ReferenceProfile())
    assert drift["features"] == []
    assert should_retrain(drift)["retreinar"] is False


# --- binning ciente de cardinalidade --------------------------------------


def test_discrete_feature_does_not_produce_a_false_alarm() -> None:
    """Bins por quantil colapsam em features discretas e inflam o PSI.

    Um monitor que acusa drift em toda execucao vira ruido e para de ser lido.
    """
    rng = np.random.default_rng(5)
    reference = rng.integers(0, 5, 3000).astype(float)
    same_distribution = rng.integers(0, 5, 3000).astype(float)

    psi = population_stability_index(reference, same_distribution)

    assert psi < 0.05, f"feature discreta estavel acusou drift (PSI={psi:.4f})"


def test_genuinely_disjoint_discrete_values_are_still_flagged() -> None:
    """O tratamento discreto nao pode mascarar deslocamento real.

    E o caso de `mes`: treino cobre os meses 1 a 5 e teste, 6 e 7.
    """
    reference = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 400)
    disjoint = np.repeat([6.0, 7.0], 1000)

    assert population_stability_index(reference, disjoint) > PSI_SIGNIFICANT


def test_derived_encodings_are_left_out_of_feature_psi() -> None:
    """Comparar valores de target encoding entre treino e inferencia nao informa.

    No treino eles vem de historico parcial (out-of-fold); na inferencia, das
    estatisticas finais. O sinal util e a distribuicao da categoria de origem,
    coberta pela cobertura por segmento.
    """
    frame = _frame(500)
    frame["Classe_target_enc"] = np.linspace(0, 0.3, 500)
    frame["Tag_freq"] = 0.1

    profile = build_reference_profile(frame, [*FEATURES, "Classe_target_enc", "Tag_freq"])

    assert "Classe_target_enc" not in profile.feature_reference
    assert "Tag_freq" not in profile.feature_reference
    assert set(FEATURES).issubset(profile.feature_reference)
