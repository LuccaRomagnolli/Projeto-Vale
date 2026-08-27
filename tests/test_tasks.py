import tasks


def test_documented_tasks_are_registered() -> None:
    expected_tasks = {
        "install",
        "format",
        "lint",
        "test",
        "label",
        "eda",
        "dashboard",
        "features",
        "train",
        "model-selection",
        "gate-stability",
        "evaluate",
        "evaluate-segments",
        "infer",
        "notebook",
        "smoke",
        "run-all",
        "clean",
    }

    assert expected_tasks <= set(tasks.DESCRIPTIONS)


def test_composite_tasks_reference_registered_tasks() -> None:
    runnable_tasks = set(tasks.COMMANDS) | set(tasks.COMPOSITES) | set(tasks.ALIASES) | {"clean"}

    for child_tasks in tasks.COMPOSITES.values():
        assert set(child_tasks) <= runnable_tasks


def test_aliases_reference_commands_or_composites() -> None:
    runnable_tasks = set(tasks.COMMANDS) | set(tasks.COMPOSITES) | {"clean"}

    assert set(tasks.ALIASES.values()) <= runnable_tasks
