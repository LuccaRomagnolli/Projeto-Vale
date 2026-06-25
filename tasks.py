from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


COMMANDS: dict[str, list[list[str]]] = {
    "install": [
        [PYTHON, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82"],
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "-c",
            "constraints.txt",
            "--prefer-binary",
        ],
    ],
    "format": [[PYTHON, "-m", "black", "src", "tests"]],
    "lint": [[PYTHON, "-m", "ruff", "check", "src", "tests"]],
    "test": [
        [
            PYTHON,
            "-m",
            "pytest",
            "tests",
            "-v",
            "--cov=src",
            "--cov-config=.coveragerc",
            "--cov-report=term-missing",
        ]
    ],
    "label": [[PYTHON, "-m", "src.data.make_dataset"]],
    "eda": [[PYTHON, "-m", "src.eda.run_eda"]],
    "dashboard": [[PYTHON, "-m", "streamlit", "run", "streamlit_app.py"]],
    "features": [[PYTHON, "-m", "src.features.build_features"]],
    "train-baseline": [[PYTHON, "-m", "src.models.train_baseline"]],
    "model-selection": [[PYTHON, "-m", "src.models.model_selection"]],
    "gate-stability": [[PYTHON, "-m", "src.models.stability_gate"]],
    "evaluate": [[PYTHON, "-m", "src.evaluation.evaluate_model"]],
    "evaluate-segments": [[PYTHON, "-m", "src.evaluation.segment_analysis"]],
    "infer": [[PYTHON, "-m", "src.inference"]],
    "notebook": [
        [
            PYTHON,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "notebooks/main.ipynb",
            "--ExecutePreprocessor.timeout=-1",
        ]
    ],
}

ALIASES = {
    "benchmark": "model-selection",
    "tune-hist-gbdt": "model-selection",
}

COMPOSITES = {
    "train": ["train-baseline", "model-selection"],
    "smoke": ["test", "infer", "evaluate", "evaluate-segments"],
    "run-all": [
        "label",
        "eda",
        "features",
        "train",
        "gate-stability",
        "evaluate",
        "evaluate-segments",
        "infer",
        "notebook",
    ],
}

DESCRIPTIONS = {
    "install": "instala dependencias Python com wheels binarios quando disponiveis",
    "format": "formata arquivos Python com black",
    "lint": "verifica padroes com ruff",
    "test": "executa testes com cobertura",
    "label": "gera base rotulada",
    "eda": "executa analise exploratoria",
    "dashboard": "abre o dashboard Streamlit",
    "features": "constroi variaveis modelaveis",
    "train": "treina baseline e selecao robusta",
    "train-baseline": "treina o baseline diagnostico",
    "model-selection": "seleciona o melhor candidato oficial",
    "benchmark": "alias legado de model-selection",
    "tune-hist-gbdt": "alias legado de model-selection",
    "gate-stability": "valida estabilidade temporal",
    "evaluate": "avalia metricas operacionais",
    "evaluate-segments": "avalia segmentos operacionais",
    "infer": "gera inferencia com artefato promovido",
    "notebook": "executa notebooks/main.ipynb",
    "smoke": "executa validacao rapida",
    "run-all": "executa o fluxo completo oficial",
    "clean": "remove caches locais de execucao",
}


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    if command[2:3] == ["pytest"]:
        env["COVERAGE_FILE"] = ".pytest_cache/.coverage"
    print(f"+ {display_command(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_task(task: str) -> None:
    task = ALIASES.get(task, task)

    if task == "clean":
        clean()
        return

    if task in COMPOSITES:
        for child_task in COMPOSITES[task]:
            run_task(child_task)
        return

    if task not in COMMANDS:
        raise SystemExit(f"Tarefa desconhecida: {task}")

    for command in COMMANDS[task]:
        run_command(command)


def clean() -> None:
    for path in [ROOT / ".pytest_cache", ROOT / ".ruff_cache"]:
        if path.exists():
            shutil.rmtree(path)
            print(f"removed {path.relative_to(ROOT)}")

    for path in [ROOT / ".coverage", *ROOT.glob(".coverage.*")]:
        if path.exists() and path.name != ".coveragerc":
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"removed {path.relative_to(ROOT)}")

    ignored_dirs = {".git", ".venv", "venv", "env", "node_modules"}
    for current_root, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in ignored_dirs]
        if "__pycache__" not in dirnames:
            continue
        cache_path = Path(current_root) / "__pycache__"
        shutil.rmtree(cache_path)
        dirnames.remove("__pycache__")
        print(f"removed {cache_path.relative_to(ROOT)}")


def list_tasks() -> None:
    for task in sorted(DESCRIPTIONS):
        print(f"{task:18} {DESCRIPTIONS[task]}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runner multiplataforma do projeto Vale.",
    )
    parser.add_argument("task", nargs="?", help="tarefa a executar")
    parser.add_argument("--list", action="store_true", help="lista tarefas disponiveis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list or not args.task:
        list_tasks()
        return

    run_task(args.task)


if __name__ == "__main__":
    main()
