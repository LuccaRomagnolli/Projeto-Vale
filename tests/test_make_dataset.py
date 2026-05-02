from pathlib import Path

import pandas as pd
import pytest
from src.data.make_dataset import load_raw_data


def test_load_raw_data_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    df = load_raw_data(path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


def test_load_raw_data_invalid_extension(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("invalid")

    with pytest.raises(ValueError):
        load_raw_data(path)
