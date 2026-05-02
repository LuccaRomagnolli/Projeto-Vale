# Mining Fleet Alert Anticipation

Projeto para antecipacao de alertas criticos ("Don't Go") em equipamentos de mineracao, com foco em priorizacao operacional por `Tag`.

## Objetivo

Responder diariamente: quais equipamentos devem entrar primeiro na fila de manutencao preventiva nas proximas 4 horas.

O projeto avalia o modelo em duas perspectivas:

1. Classificacao ciclo-a-ciclo (recall, precision, AUC-PR, AUC-ROC).
2. Priorizacao operacional por `TopK Tag-dia` (precision@k, recall@k, lift@k).

## Estado Atual (dados de referencia)

- Janela de target: `target_4h`.
- Split temporal: `70/15/15`.
- Modelo campeao de benchmark: `hist_gbdt_regularized`.
- Artefato operacional atual: `models/hist_gbdt_tuned.joblib`.
- Threshold calibrado em validacao: `0.1802228521655894`.

Metricas operacionais no teste (`Top15 Tag-dia`):

- `precision@15`: `0.6689`
- `recall@15`: `0.7288`
- `lift@15`: `2.0569`

## Estrutura Real do Repositorio

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── labeled/
│   │   └── features/
│   └── external/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── alert_labeler.py
│   ├── data_loader.py
│   ├── inference.py
│   ├── data/
│   ├── eda/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Pipeline

1. Ingestao e contrato de dados: `src/data_loader.py`
2. Rotulacao robusta: `src/alert_labeler.py`
3. EDA: `src/eda/run_eda.py`
4. Feature engineering: `src/features/build_features.py`
5. Split temporal e baseline: `src/models/train_baseline.py`
6. Treino principal supervisionado: `src/models/train_model.py`
7. Benchmark de candidatos: `src/models/benchmark_models.py`
8. Tuning e backtesting HistGBDT: `src/models/tune_hist_gbdt.py`
9. Gate de estabilidade: `src/models/stability_gate.py`
10. Avaliacao operacional e segmentada:
    - `src/evaluation/evaluate_model.py`
    - `src/evaluation/segment_analysis.py`
11. Inferencia operacional: `src/inference.py`

## Setup

Requisitos:

- Python `>=3.11,<3.14`
- Dependencias de `requirements.txt`

## Metodologia de Execucao Recomendada (Mac, mais eficiente)

Fluxo validado na pratica para evitar retrabalho de ambiente:

1. Tentar execucao direta no ambiente ja funcional e validar com `make smoke`.
2. Se faltar dependencia, criar ambiente `conda` isolado com Python 3.12.
3. Evitar bootstrap em Python 3.13 para instalacao completa do `requirements.txt`, por causa de incompatibilidade de build do `pyarrow==15.0.2` com `pkg_resources`.

### Caminho rapido (quando ja existe ambiente com dependencias)

```bash
make smoke
```

Esse comando roda:

- `make test`
- `make infer`
- `make evaluate`
- `make evaluate-segments`

### Caminho isolado recomendado (do zero)

```bash
eval "$(/opt/miniconda3/bin/conda shell.zsh hook)"
conda activate base

conda create -n vale312 python=3.12 -y
conda activate vale312

python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

Observacao:

- se aparecer `CondaError: KeyboardInterrupt`, a criacao do ambiente foi interrompida; rode novamente `conda create -n vale312 python=3.12 -y` sem cancelar.
- nao use `source .venv/bin/activate` quando estiver usando `conda`.

### Opcional: venv tradicional

Instalacao com `venv` (quando voce ja possui Python compativel no sistema):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Comandos Principais

```bash
make label
make eda
make features
make train
make benchmark
make tune-hist-gbdt
make gate-stability
make evaluate
make evaluate-segments
make infer
make test
make lint
```

Pipeline completo:

```bash
make run-all
```

## Contrato de Inferencia

`src/inference.py` implementa contrato minimo para uso operacional:

- valida artefato (`model`, `feature_columns`, `threshold`);
- alinha schema de entrada com `feature_columns`;
- preenche features ausentes com `0.0`;
- gera `score`, `prediction` e persiste saida em `reports/inference_scores.parquet`.

Entrada esperada: `csv` ou `parquet` com features.

## Politica de Promocao

Ver documento oficial:

- `docs/politica_promocao_modelo.md`

Resumo:

1. Campeao por validacao temporal (`val_auc_pr`, desempates definidos).
2. Metas `Top15 Tag-dia` atendidas no teste.
3. Gate de estabilidade aprovado.
4. Segmentos raros tratados como inconclusivos, com trilha dedicada.

## Notebooks

Ordem recomendada:

1. `01_business_understanding.ipynb`
2. `02_data_understanding_eda.ipynb`
3. `03_data_preparation.ipynb`
4. `04_modeling_benchmark.ipynb`
5. `05_operational_evaluation_application.ipynb`
6. `06_segment_analysis_and_risk.ipynb`
7. `07_model_governance_and_promotion.ipynb`
8. `08_executive_readout_for_head_of_tech.ipynb`

## Observacoes

- Alguns relatorios versionados podem conter caminhos absolutos historicos de execucao local.
- O threshold de operacao deve vir do artefato promovido, nao de valor fixo em variavel de ambiente.
