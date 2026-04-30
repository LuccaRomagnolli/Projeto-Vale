from __future__ import annotations

import pandas as pd

from projeto_vale.target import build_next_alert_target


def test_target_marks_alert_within_4h():
    ap = pd.DataFrame(
        {
            "Tag": ["CA1", "CA1", "CA1"],
            "Inicio": [
                "2025-01-01 00:00:00",
                "2025-01-01 05:00:00",
                "2025-01-01 10:00:00",
            ],
        }
    )
    events = pd.DataFrame(
        {
            "tag": ["CA1", "CA1"],
            "alert_time": ["2025-01-01 03:00:00", "2025-01-01 20:00:00"],
        }
    )

    y = build_next_alert_target(ap, events, horizon_hours=4)
    assert y.tolist() == [1, 0, 0]
