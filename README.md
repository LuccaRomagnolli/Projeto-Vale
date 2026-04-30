# Projeto Vale

Estrutura inicial do projeto para previsao de alerta critico nas proximas 4 horas em frotas de mineracao.

## Objetivo v1

Construir um pipeline reprodutivel com:
- ingestao de dados de apontamentos e telemetria,
- rotulagem de alerta critico baseada nas regras de negocio,
- engenharia de features temporais,
- baseline com split temporal,
- avaliacao com metricas e graficos.

## Estrutura

- `data/raw`, `data/interim`, `data/processed`
- `notebooks/`
- `src/projeto_vale/`
- `scripts/`
- `tests/`
- `reports/figures/`
- `configs/`

## Setup (uv)

```bash
uv sync
```

## Fluxo de execucao

1. Extrair dataset bruto:

```bash
uv run python scripts/extract_data.py \
  --archive "Base de Dados/datasets.7z" \
  --output-dir data/raw
```

2. Executar pipeline baseline:

```bash
uv run python scripts/run_pipeline.py \
  --apontamentos data/raw/datasets/apontamentos/desenvolver_apontamentos.parquet \
  --telemetria-dir data/raw/datasets/telemetria \
  --regras "Base de Dados/Alarmes - Regra de Negocio.xlsx" \
  --processed-dir data/processed \
  --reports-dir reports
```

3. Rodar testes:

```bash
uv run pytest
```

## Notas

- A regra critica inicial considera `NIVEL = Muito Alto`.
- Quando nao for possivel mapear regra diretamente nos eventos, o pipeline usa fallback `Is_Dont_Go` (se existir) e registra a origem do label.
- O pipeline v1 prioriza baseline e reproducibilidade; modelos avancados entram na fase 2.
