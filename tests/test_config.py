from src.utils import config


def test_project_paths_are_resolved() -> None:
    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.name == "data"
    assert config.RAW_DIR.name == "raw"
    assert config.REPORTS_DIR.name == "reports"
    assert config.REPORTS_MODEL_SELECTION_DIR.name == "model_selection"
    assert config.REPORTS_OPERATIONAL_DIR.name == "operational"
    assert config.REPORTS_SEGMENT_DIR.name == "segments"
