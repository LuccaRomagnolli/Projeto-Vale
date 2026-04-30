from __future__ import annotations

import pandas as pd

from projeto_vale.rules import apply_critical_rules


def test_apply_critical_rules_matches_by_event_and_level():
    tele = pd.DataFrame(
        {
            "Data_Evento": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
            "TAG": ["CA1", "CA1"],
            "Alarme": ["Low Transmission Oil Level", "Other Event"],
            "Criticidade": ["Muito Alto", "Muito Alto"],
            "Tipo": ["ALARME OEM", "ALARME OEM"],
        }
    )

    rules = pd.DataFrame(
        {
            "TIPO": ["ALARME OEM"],
            "EVENTO": ["Low Transmission Oil Level"],
            "SITUACAO": ["x"],
            "NIVEL": ["Muito Alto"],
        }
    )

    out = apply_critical_rules(tele, rules)
    assert out["is_critical_alert"].tolist() == [1, 0]


def test_apply_critical_rules_fallback_is_dont_go():
    tele = pd.DataFrame(
        {
            "Data_Evento": ["2025-01-01 00:00:00"],
            "TAG": ["CA1"],
            "Is_Dont_Go": [1],
        }
    )
    rules = pd.DataFrame({"NIVEL": []})

    out = apply_critical_rules(tele, rules)
    assert int(out.loc[0, "is_critical_alert"]) == 1
    assert out.loc[0, "label_source"] == "is_dont_go_fallback"
