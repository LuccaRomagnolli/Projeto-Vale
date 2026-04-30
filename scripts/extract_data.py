from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrai o arquivo datasets.7z para data/raw")
    parser.add_argument("--archive", required=True, help="Caminho do arquivo .7z")
    parser.add_argument("--output-dir", required=True, help="Diretorio de saida")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)

    if not archive.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {archive}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["bsdtar", "-xf", str(archive), "-C", str(output_dir)]
    subprocess.run(cmd, check=True)
    print(f"Extraido com sucesso em: {output_dir}")


if __name__ == "__main__":
    main()
