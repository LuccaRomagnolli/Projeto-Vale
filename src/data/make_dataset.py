"""Carga e preparação inicial dos dados brutos."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Carrega um dataset bruto com inferência simples por extensão."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato não suportado: {path}")


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    print("[TODO] Implementar pipeline de carga dos arquivos em data/raw")
    print("[TODO] Aplicar rotulação inicial de alertas usando regras de negócio")


if __name__ == "__main__":
    main()
