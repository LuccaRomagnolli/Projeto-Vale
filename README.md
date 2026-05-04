<p align="center">
  <img src="pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Mining Fleet Alert Anticipation</h1>

<p align="center">
  Antecipação de alertas críticos <strong>"Don't Go"</strong> em equipamentos de mineração,<br>
  com foco em priorização operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20–%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Sele%C3%A7%C3%A3o-4%20Modelos-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

## Objetivo

Responder diariamente: **quais equipamentos devem entrar primeiro na fila de manutenção preventiva nas próximas 4 horas.**

O projeto avalia o modelo com a mesma metodologia operacional em todas as etapas:

1. **Priorização operacional por `TopK Tag-dia`** — precision@k, recall@k, lift@k.
2. **Classificação ciclo-a-ciclo** — recall, precision, AUC-PR, AUC-ROC como diagnóstico técnico auxiliar.

---

## Estado Atual

| Parâmetro | Valor |
|---|---|
| Janela de target | `target_4h` |
| Split temporal | `70 / 15 / 15` |
| Estratégia de modelagem | `benchmark robusto com 4 modelos` |
| Artefato de benchmark selecionado | `models/model_benchmark_selected.joblib` |
| Artefato operacional | `models/hist_gbdt_tuned.joblib` |

### Métricas Operacionais — `Top15 Tag-dia` (Teste)

| Métrica | Valor |
|---|---|
| `precision@15` | **0.6800** |
| `recall@15` | **0.7409** |
| `lift@15` | **2.0910** |

---

## Viabilidade e Metodologia (Resumo Executivo)

O projeto esta metodologicamente viavel para **piloto operacional assistido**,
porque:

1. Usa **split temporal 70/15/15** e threshold calibrado somente na validacao.
2. Prioriza metrica operacional aderente ao uso real: **TopK Tag-dia**.
3. Exige gate de estabilidade antes de promocao (`make gate-stability`).
4. Mantem avaliacao segmentada e separa segmentos de baixa prevalencia como
   inconclusivos.

Limites atuais conhecidos:

- O uso por threshold global ciclo-a-ciclo ainda gera taxa de alertas alta.
- Segmentos raros (ex.: baixa prevalencia) exigem trilha dedicada.
- A confianca operacional depende de monitoramento continuo de drift e de
  regressao de metricas em CI.

---

## Estrutura do Repositório

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

---

## Pipeline

| # | Etapa | Arquivo |
|---|---|---|
| 1 | Ingestão e contrato de dados | `src/data_loader.py` |
| 2 | Rotulação robusta | `src/alert_labeler.py` |
| 3 | EDA | `src/eda/run_eda.py` |
| 4 | Feature engineering | `src/features/build_features.py` |
| 5 | Split temporal e baseline | `src/models/train_baseline.py` |
| 6 | Treino supervisionado de referência | `src/models/train_model.py` |
| 7 | Benchmark robusto de 4 candidatos | `src/models/benchmark_models.py` |
| 8 | Tuning e backtesting HistGBDT | `src/models/tune_hist_gbdt.py` |
| 9 | Gate de estabilidade | `src/models/stability_gate.py` |
| 10 | Avaliação operacional | `src/evaluation/evaluate_model.py` |
| 11 | Análise segmentada | `src/evaluation/segment_analysis.py` |
| 12 | Inferência operacional | `src/inference.py` |

---

## Setup

**Requisitos:** Python `>=3.11, <3.14`

### Caminho rápido (ambiente já configurado)

```bash
make smoke
```

Executa em sequência: `make test` → `make infer` → `make evaluate` → `make evaluate-segments`.

### Ambiente isolado recomendado (do zero)

```bash
conda create -n vale312 python=3.12 -y
conda activate vale312

python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

> **Windows (PowerShell):**
> `conda activate` funciona apos `conda init powershell` e reinicio do terminal.
>
> **macOS/Linux:**
> use `eval "$(conda shell.bash hook)"` ou equivalente do seu shell antes do
> `conda activate`.
>
> Se aparecer `CondaError: KeyboardInterrupt`, a criacao foi interrompida;
> execute novamente sem cancelar.

### Alternativa: venv tradicional

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Comandos Principais

```bash
make label              # rotulação
make eda                # análise exploratória
make features           # feature engineering
make train              # treino do modelo
make benchmark          # benchmark de candidatos
make tune-hist-gbdt     # tuning HistGBDT
make gate-stability     # gate de estabilidade
make evaluate           # avaliação operacional
make evaluate-segments  # análise segmentada
make infer              # inferência
make test               # testes
make lint               # linting
```

**Pipeline completo:**

```bash
make run-all
```

---

## Contrato de Inferência

`src/inference.py` implementa o contrato mínimo para uso operacional:

- Valida o artefato: `model`, `feature_columns`, `threshold`
- Alinha schema de entrada com `feature_columns`
- Preenche features ausentes com `0.0`
- Gera `score`, `prediction` e persiste saída em `reports/inference_scores.parquet`

**Entrada esperada:** `csv` ou `parquet` com features.

---

## Política de Promoção

> Documento oficial: `docs/politica_promocao_modelo.md`
> Controle de alterações: `docs/controle_alteracoes.md`

1. **Seleção técnica posterior ao benchmark** — sem modelo principal a priori; escolher entre 4 candidatos por `val_top15_recall_at_k`, com desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr`.
2. **Metas `Top15 Tag-dia` atendidas** no conjunto de teste.
3. **Gate de estabilidade aprovado** — sem drift ou degradação.
4. **Segmentos raros** tratados como inconclusivos, com trilha dedicada.

---

## Notebooks

Notebook ativo do projeto:

- `notebooks/main.ipynb` - walkthrough tecnico-executivo consolidado.

> Observacao: referencias antigas a notebooks `01..09` foram removidas da
> documentacao por nao refletirem mais a estrutura atual versionada.

---

## Observações

- Alguns relatórios versionados podem conter caminhos absolutos históricos de execução local.
- O threshold de operação deve vir do artefato promovido, **não** de valor fixo em variável de ambiente.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
