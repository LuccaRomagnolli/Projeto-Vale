<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Mining Fleet Alert Anticipation</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-HistGBDT-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

# Notebook principal e contexto do projeto

Data de atualizacao: 04/05/2026

Arquivo fonte:

- `notebooks/main.ipynb`

## Objetivo do notebook

Consolidar a leitura tecnico-executiva do projeto em um unico fluxo:

1. contexto de negocio;
2. qualidade dos dados;
3. engenharia de features;
4. benchmark de modelos;
5. avaliacao operacional (`TopK Tag-dia`);
6. riscos segmentados e recomendacao final.

## Escopo atual (fonte de verdade)

Este documento foi atualizado para refletir apenas o que existe hoje no
repositorio. As referencias oficiais sao:

- `README.md`
- `docs/politica_promocao_modelo.md`
- `docs/controle_alteracoes.md`
- `reports/model_benchmark_report.json`
- `reports/hist_gbdt_tuning_report.json`
- `reports/operational_metrics_report.json`
- `reports/segment_operational_report.json`

## Mudancas aplicadas nesta atualizacao

1. Remocao de listagens antigas de notebooks `01..09` que nao fazem parte da
   estrutura atual versionada.
2. Consolidacao da trilha analitica no notebook unico `notebooks/main.ipynb`.
3. Alinhamento metodologico com a politica operacional vigente:
   - selecao por validacao temporal operacional;
   - metrica executiva centrada em `TopK Tag-dia`;
   - gate de estabilidade e avaliacao segmentada antes de promocao.

## Como reproduzir o notebook com dados atuais

1. Executar pipeline base:

```bash
make run-all
```

2. Opcional para validacao rapida:

```bash
make smoke
```

3. Abrir e executar:

- `notebooks/main.ipynb`

## Observacao de governanca

Sempre que houver alteracao de metrica principal, threshold operacional, ou
politica de promocao, este documento e o `README.md` devem ser atualizados na
mesma entrega para evitar divergencia de documentacao.
# Codigo do Notebook e Contexto do Projeto

Data de geracao: 2026-05-03

Arquivo fonte do notebook: `notebooks/main.ipynb`

## 1. Contexto do Projeto

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
| Modelo campeão | `hist_gbdt_balanced` |
| Artefato operacional | `models/hist_gbdt_tuned.joblib` |
| Threshold calibrado (validação) | `0.141388104973226` |

### Métricas Operacionais — `Top15 Tag-dia` (Teste)

| Métrica | Valor |
|---|---|
| `precision@15` | **0.6800** |
| `recall@15` | **0.7409** |
| `lift@15` | **2.0910** |

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
| 6 | Treino principal supervisionado | `src/models/train_model.py` |
| 7 | Benchmark de candidatos | `src/models/benchmark_models.py` |
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
eval "$(/opt/miniconda3/bin/conda shell.zsh hook)"
conda activate base

conda create -n vale312 python=3.12 -y
conda activate vale312

python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

> **Atenção:** Se aparecer `CondaError: KeyboardInterrupt`, a criação foi interrompida — rode novamente sem cancelar. Não use `source .venv/bin/activate` em conjunto com `conda`.

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

1. **Campeão por validação temporal operacional** — `val_top15_recall_at_k`, com desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr`.
2. **Metas `Top15 Tag-dia` atendidas** no conjunto de teste.
3. **Gate de estabilidade aprovado** — sem drift ou degradação.
4. **Segmentos raros** tratados como inconclusivos, com trilha dedicada.

---

## Notebooks

Entrada principal para avaliação de gestor:

- `09_manager_full_walkthrough.ipynb`

Trilha técnica de suporte/auditoria:

| # | Notebook |
|---|---|
| 01 | `01_business_understanding.ipynb` |
| 02 | `02_data_understanding_eda.ipynb` |
| 03 | `03_data_preparation.ipynb` |
| 04 | `04_modeling_benchmark.ipynb` |
| 05 | `05_operational_evaluation_application.ipynb` |
| 06 | `06_segment_analysis_and_risk.ipynb` |
| 07 | `07_model_governance_and_promotion.ipynb` |
| 08 | `08_executive_readout_for_head_of_tech.ipynb` |
| 09 | `09_manager_full_walkthrough.ipynb` |

---

## Observações

- Alguns relatórios versionados podem conter caminhos absolutos históricos de execução local.
- O threshold de operação deve vir do artefato promovido, **não** de valor fixo em variável de ambiente.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>

## 2. Resumo dos Artefatos e Resultados

| Item | Valor |
|---|---|
| Campeao do benchmark | `hist_gbdt_balanced` |
| Regra de selecao | `maior val_top15_recall_at_k; desempate por val_top15_precision_at_k, val_top15_lift_vs_random e val_auc_pr` |
| Val Precision@15 | `0.6619` |
| Val Recall@15 | `0.7493` |
| Val Lift@15 | `2.1231` |

| Item | Valor |
|---|---|
| Melhor tuning | `hist_gbdt_tuned_04` |
| Threshold operacional | `0.141388` |
| Test Precision@15 | `0.6800` |
| Test Recall@15 | `0.7409` |
| Test Lift@15 | `2.0910` |

| Item | Valor |
|---|---|
| Alertas/dia no Top15 | `15.0` |
| Positivos capturados no Top15 | `306` |
| Total de positivos Tag-dia | `413` |

## 3. Arquivos Principais do Projeto

- `src/data/make_dataset.py`
- `src/alert_labeler.py`
- `src/features/build_features.py`
- `src/models/train_model.py`
- `src/models/benchmark_models.py`
- `src/models/tune_hist_gbdt.py`
- `src/models/stability_gate.py`
- `src/evaluation/evaluate_model.py`
- `src/evaluation/segment_analysis.py`
- `src/evaluation/operational_scorecard.py`
- `src/inference.py`
- `Makefile`

## 4. Estrutura do Notebook

- Celula 1: Markdown - 09 - Manager Full Walkthrough
- Celula 2: Codigo - `from pathlib import Path`
- Celula 3: Markdown - 1. Problema e objetivos
- Celula 4: Codigo - `summary = pd.DataFrame(`
- Celula 5: Markdown - 2. Qualidade e cobertura de dados
- Celula 6: Codigo - `def decode_one_hot_prefix(df: pd.DataFrame, prefix: str) -> pd.Series:`
- Celula 7: Markdown - 3. EDA visual completa
- Celula 8: Codigo - `import matplotlib.pyplot as plt`
- Celula 9: Markdown - 4. Preparação e feature engineering
- Celula 10: Codigo - `all_cols = features_df.columns.tolist()`
- Celula 11: Markdown - 5. Benchmark de candidatos
- Celula 12: Codigo - `# ── Benchmark table ──────────────────────────────────────────────────────────`
- Celula 13: Markdown - 6. Modelo campeão e threshold operacional
- Celula 14: Codigo - `threshold = threshold_test['threshold']`
- Celula 15: Markdown - 7. Explicação do modelo
- Celula 16: Codigo - `imp = feature_importance_df.copy().head(12)`
- Celula 17: Markdown - 8. Avaliação operacional
- Celula 18: Codigo - `topk = oper_topk_df.query("split == 'test'").copy().sort_values('top_k_tags_per_`
- Celula 19: Markdown - 9. Segmentos e riscos
- Celula 20: Codigo - `strong = pd.DataFrame(segment_json.get('strongest_top15_segments', []))`
- Celula 21: Markdown - 10. Recomendação executiva

## 5. Codigo Completo do Notebook

### Celula de Codigo 1 (notebook cell 2)

```python
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if not (ROOT / 'reports').exists() and (ROOT.parent / 'reports').exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plot_theme import apply_manager_theme, add_source_note, PALETTE

apply_manager_theme()

REPORTS = ROOT / 'reports'
DATA = ROOT / 'data' / 'processed'

features_df = pd.read_parquet(DATA / 'features' / 'features_dataset.parquet')
split_test_df = pd.read_parquet(DATA / 'features' / 'splits' / 'features_test.parquet')
benchmark_df = pd.read_csv(REPORTS / 'model_benchmark_report.csv')
oper_budget_df = pd.read_csv(REPORTS / 'operational_budget_metrics.csv')
oper_topk_df = pd.read_csv(REPORTS / 'operational_daily_topk_metrics.csv')
segment_threshold_df = pd.read_csv(REPORTS / 'segment_threshold_metrics.csv')
segment_topk_df = pd.read_csv(REPORTS / 'segment_topk_tag_day_metrics.csv')
segment_hotspots_df = pd.read_csv(REPORTS / 'segment_tag_hotspots.csv')
threshold_curve_df = pd.read_csv(REPORTS / 'hist_gbdt_threshold_curve.csv')
feature_importance_df = pd.read_csv(REPORTS / 'model_feature_importance.csv')

with open(REPORTS / 'operational_metrics_report.json', 'r', encoding='utf-8') as f:
    oper_json = json.load(f)

with open(REPORTS / 'model_benchmark_report.json', 'r', encoding='utf-8') as f:
    benchmark_json = json.load(f)

with open(REPORTS / 'hist_gbdt_tuning_report.json', 'r', encoding='utf-8') as f:
    tuning_json = json.load(f)

with open(REPORTS / 'segment_operational_report.json', 'r', encoding='utf-8') as f:
    segment_json = json.load(f)

with open(ROOT / 'docs' / 'politica_promocao_modelo.md', 'r', encoding='utf-8') as f:
    policy_text = f.read()

threshold_test = [x for x in oper_json['threshold_metrics'] if x['split'] == 'test'][0]
topk15 = oper_topk_df.loc[oper_topk_df['top_k_tags_per_day'] == 15].iloc[0]
winner_name = benchmark_json['winner']['model_name']
winner_val_recall_top15 = benchmark_json['winner']['val_top15_recall_at_k']
winner_val_precision_top15 = benchmark_json['winner']['val_top15_precision_at_k']
winner_val_lift_top15 = benchmark_json['winner']['val_top15_lift_vs_random']
winner_val_auc_pr = benchmark_json['winner']['val_auc_pr']

features_df['Inicio'] = pd.to_datetime(features_df['Inicio'], errors='coerce', utc=True)
features_df['Fim'] = pd.to_datetime(features_df['Fim'], errors='coerce', utc=True)
split_test_df['Fim'] = pd.to_datetime(split_test_df['Fim'], errors='coerce', utc=True)

print('Dados carregados com sucesso.')
print('Rows features:', len(features_df), '| Rows test:', len(split_test_df), '| Benchmark models:', len(benchmark_df))
```

### Celula de Codigo 2 (notebook cell 4)

```python
summary = pd.DataFrame(
    {
        'Indicador': [
            'Total de registros de features',
            'Período coberto',
            'Modelo campeão (benchmark)',
            'Threshold operacional',
            'Precision@Top15',
            'Recall@Top15',
            'Lift@Top15',
        ],
        'Valor': [
            f"{len(features_df):,}",
            f"{features_df['Fim'].min().date()} a {features_df['Fim'].max().date()}",
            winner_name,
            f"{threshold_test['threshold']:.4f}",
            f"{topk15['precision_at_k']:.4f}",
            f"{topk15['recall_at_k']:.4f}",
            f"{topk15['lift_vs_random']:.4f}",
        ],
    }
)

display(summary)

display(Markdown(
    '**Insight + implicação operacional:** o modelo já sustenta priorização com bom equilíbrio em Top15; isso suporta piloto controlado com governança ativa.'
))
```

### Celula de Codigo 3 (notebook cell 6)

```python
def decode_one_hot_prefix(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Decodifica colunas one-hot com prefixo dado, retornando a categoria ativa.
    Retorna 'desconhecido' quando nenhuma coluna está ativa ou não existem colunas.
    """
    FALLBACK = "desconhecido"
    prefix_pattern = f"{prefix}_"

    cols = [c for c in df.columns if c.startswith(prefix_pattern)]
    if not cols:
        return pd.Series([FALLBACK] * len(df), index=df.index, dtype="string")

    matrix = df[cols].fillna(0).astype(int)
    row_sums = matrix.sum(axis=1)

    decoded = (
        matrix
        .idxmax(axis=1)
        .str.removeprefix(prefix_pattern)   # mais legível que str.replace com regex=False
        .astype("string")
    )
    decoded[row_sums == 0] = FALLBACK       # .loc desnecessário para máscara booleana simples

    return decoded


# ── Preparação ────────────────────────────────────────────────────────────────
qc = features_df.copy()
qc["frota_decoded"] = decode_one_hot_prefix(qc, "Frota")
qc["tipo_decoded"]  = decode_one_hot_prefix(qc, "Tipo")
qc["turno"]         = qc["turno"].astype(str)

# ── Métricas auxiliares ────────────────────────────────────────────────────────
missing  = (qc.isna().mean() * 100).sort_values(ascending=False).head(12)
vol_dia  = qc.groupby(qc["Fim"].dt.date)["Id"].count()          # bug corrigido: .['dt.date'] → .dt.date

# ── Figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# — Nulos por coluna
ax = axes[0, 0]
missing.sort_values().plot(kind="barh", ax=ax, color=PALETTE["orange"])
ax.set_title("Top 12 colunas com nulos (%)")
ax.set_xlabel("% nulos")
add_source_note(ax)

# — Cobertura temporal
ax = axes[0, 1]
ax.plot(vol_dia.index, vol_dia.values, color=PALETTE["blue"], linewidth=2)
ax.set_title("Volume de ciclos por dia")
ax.set_ylabel("Quantidade de ciclos")
ax.tick_params(axis="x", rotation=45)
add_source_note(ax)

# — Volume por frota
ax = axes[1, 0]
qc["frota_decoded"].value_counts().head(8).plot(kind="bar", ax=ax, color=PALETTE["teal"])
ax.set_title("Top frotas por volume")
ax.set_ylabel("Quantidade de ciclos")
ax.tick_params(axis="x", rotation=35)
add_source_note(ax)

# — Volume por turno
ax = axes[1, 1]
qc["turno"].value_counts().plot(kind="bar", ax=ax, color=PALETTE["navy"])
ax.set_title("Volume por turno")
ax.set_ylabel("Quantidade de ciclos")
ax.tick_params(axis="x", rotation=0)
add_source_note(ax)

plt.tight_layout()
plt.show()                                                        # bug corrigido: plt.show literal de URL

display(Markdown(
    "**Insight + implicação operacional:** a cobertura temporal é contínua e os volumes "
    "por frota/turno permitem operar um ranking diário com representatividade suficiente."
))
```

### Celula de Codigo 4 (notebook cell 8)

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display, Markdown

# ── Tema executivo claro ──────────────────────────────────────────────────────
EXEC = {
    'bg':        '#F7F8FC',
    'panel':     '#FFFFFF',
    'grid':      '#EAECF4',
    'border':    '#D5D8E8',
    'text':      '#1A1D2E',
    'muted':     '#6B7094',
    'teal':      '#0E9E84',
    'red':       '#D94F4F',
    'blue':      '#3A7FD5',
    'blue_fill': '#3A7FD520',
    'divider':   '#D5D8E8',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.facecolor':    EXEC['panel'],
    'figure.facecolor':  EXEC['bg'],
    'axes.edgecolor':    EXEC['border'],
    'axes.labelcolor':   EXEC['muted'],
    'xtick.color':       EXEC['muted'],
    'ytick.color':       EXEC['muted'],
    'axes.grid':         True,
    'grid.color':        EXEC['grid'],
    'grid.linewidth':    0.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'text.color':        EXEC['text'],
})

# ── Dados ─────────────────────────────────────────────────────────────────────
eda = features_df.copy()
eda['dia'] = eda['Fim'].dt.date

# ── Layout 3 linhas × 1 coluna ────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 18), facecolor=EXEC['bg'])
fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.05, hspace=0.55)

gs   = gridspec.GridSpec(3, 1, figure=fig)
axes = [fig.add_subplot(gs[i]) for i in range(3)]

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
fig.text(0.10, 0.975,
         'Análise Exploratória · Monitoramento de Risco',
         fontsize=14, fontweight='bold', color=EXEC['text'], va='top')

fig.text(0.10, 0.958,
         f"Base: {len(eda):,} registros  |  Target: target_4h  |  "
         f"Período: {eda['dia'].min()} → {eda['dia'].max()}",
         fontsize=9, color=EXEC['muted'], va='top')

divider = Line2D([0.10, 0.95], [0.948, 0.948],
                 transform=fig.transFigure,
                 color=EXEC['border'], linewidth=0.8, clip_on=False)
fig.add_artist(divider)

# ─────────────────────────────────────────────────────────────────────────────
# Distribuição do target
# ─────────────────────────────────────────────────────────────────────────────
ax     = axes[0]
counts = eda['target_4h'].value_counts().sort_index()
colors = [EXEC['teal'], EXEC['red']]
bars   = ax.bar(counts.index, counts.values,
                color=colors, width=0.4, zorder=3, linewidth=0)

total = counts.sum()
for bar, (label, val) in zip(bars, counts.items()):
    pct = val / total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f'{val:,}  ({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=9, color=EXEC['text'], fontweight='bold')

ax.set_title('Distribuição do Target',
             fontsize=11, fontweight='bold',
             color=EXEC['text'], pad=12, loc='left')
ax.set_xlabel('target_4h', labelpad=8)
ax.set_ylabel('Registros', labelpad=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Negativo (0)', 'Positivo (1)'])
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

legend_els = [
    Line2D([0], [0], color=EXEC['teal'], lw=0,
           marker='s', markersize=8, label='Sem alerta'),
    Line2D([0], [0], color=EXEC['red'],  lw=0,
           marker='s', markersize=8, label='Com alerta'),
]
ax.legend(handles=legend_els, fontsize=8.5, frameon=False,
          labelcolor=EXEC['muted'], loc='upper right')

for spine in ax.spines.values():
    spine.set_edgecolor(EXEC['border'])
    spine.set_linewidth(0.8)

# ─────────────────────────────────────────────────────────────────────────────
# revalência temporal
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
daily_prev = (eda.groupby('dia')['target_4h']
                 .mean()
                 .rolling(7, min_periods=1)
                 .mean())

x = daily_prev.index
y = daily_prev.values

ax.plot(x, y, color=EXEC['blue'], linewidth=1.8, zorder=3)
ax.fill_between(x, y, alpha=0.10, color=EXEC['blue'], zorder=2)

mean_val = y.mean()
ax.axhline(mean_val, color=EXEC['muted'], linewidth=0.8,
           linestyle='--', zorder=1)
ax.text(x[-1], mean_val + (y.max() - y.min()) * 0.03,
        f'Média: {mean_val:.1%}',
        ha='right', va='bottom', fontsize=8, color=EXEC['muted'])

idx_max = daily_prev.idxmax()
idx_min = daily_prev.idxmin()
for idx, label, color in [
    (idx_max, f'Máx {daily_prev[idx_max]:.1%}', EXEC['red']),
    (idx_min, f'Mín {daily_prev[idx_min]:.1%}', EXEC['teal']),
]:
    ax.annotate(label,
                xy=(idx, daily_prev[idx]),
                xytext=(0, 16), textcoords='offset points',
                ha='center', fontsize=8,
                color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

ax.set_title('Prevalência Temporal · Média Móvel 7 dias',
             fontsize=11, fontweight='bold',
             color=EXEC['text'], pad=12, loc='left')
ax.set_ylabel('Taxa de positivos', labelpad=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax.tick_params(axis='x', rotation=30)

for spine in ax.spines.values():
    spine.set_edgecolor(EXEC['border'])
    spine.set_linewidth(0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Heatmap de correlação
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
corr_cols = [
    'target_4h', 'duracao_ciclo_min', 'n_alertas_4h', 'n_alertas_24h',
    'n_ciclos_4h', 'n_ciclos_24h', 'dias_desde_ultimo_alerta',
    'delta_alertas_por_hora_4h_24h', 'Tag_freq', 'Classe_target_enc'
]
sub  = eda[corr_cols].sample(min(30_000, len(eda)), random_state=42)
corr = sub.corr(numeric_only=True)

labels_curtos = [
    'target_4h', 'dur_ciclo', 'alert_4h', 'alert_24h',
    'ciclos_4h', 'ciclos_24h', 'dias_ult_alert',
    'Δalert/h', 'tag_freq', 'classe_enc'
]

mask = np.eye(len(corr), dtype=bool)

sns.heatmap(
    corr, mask=mask,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    annot=True, fmt='.2f',
    annot_kws={'size': 7, 'color': EXEC['text']},
    linewidths=0.4, linecolor=EXEC['bg'],
    xticklabels=labels_curtos,
    yticklabels=labels_curtos,
    cbar=True, cbar_kws={'shrink': 0.6, 'pad': 0.02},
    ax=ax
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors=EXEC['muted'], labelsize=7.5)
cbar.outline.set_edgecolor(EXEC['border'])

ax.set_title('Correlação entre Variáveis-Chave',
             fontsize=11, fontweight='bold',
             color=EXEC['text'], pad=12, loc='left')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)

# ── Rodapé ────────────────────────────────────────────────────────────────────
fig.text(
    0.10, 0.012,
    '● Insight: o target não é aleatório no tempo — sinais consistentes em '
    'histórico de alertas e intensidade operacional justificam modelagem '
    'supervisionada de risco.',
    fontsize=8.5, color=EXEC['muted'], va='bottom', style='italic'
)

plt.savefig('eda_executivo.png', dpi=150, bbox_inches='tight',
            facecolor=EXEC['bg'])
plt.show()
```

### Celula de Codigo 5 (notebook cell 10)

```python
all_cols = features_df.columns.tolist()

families = {
    'Temporais': [c for c in all_cols if c in {'hora_do_dia','dia_da_semana','mes','turno','is_fim_de_semana','duracao_ciclo_min'}],
    'Alertas rolling': [c for c in all_cols if c.startswith('n_alertas_') or c.startswith('alertas_por_hora_')],
    'Ciclos rolling': [c for c in all_cols if c.startswith('n_ciclos_') or c.startswith('duracao_media_ciclo_') or c.startswith('duracao_std_ciclo_')],
    'Mudança/degradação': [c for c in all_cols if c.startswith('delta_') or c.startswith('ratio_')],
    'Codificação categórica': [c for c in all_cols if c.endswith('_freq') or c.endswith('_target_enc') or c.startswith('Frota_') or c.startswith('Tipo_')],
}

family_df = pd.DataFrame(
    {'Família': list(families.keys()), 'Quantidade de features': [len(v) for v in families.values()]}
).sort_values('Quantidade de features', ascending=False)

display(family_df)

example_cols = [
    'n_alertas_4h', 'n_alertas_24h', 'n_ciclos_4h', 'duracao_media_ciclo_24h',
    'delta_alertas_por_hora_4h_24h', 'Tag_freq', 'Classe_target_enc'
]
display(features_df[example_cols + ['target_4h']].head(8))

display(Markdown(
    '**Insight + implicação operacional:** as features capturam contexto temporal, histórico e mudança de comportamento da operação, o que permite priorização mais robusta por equipamento.'
))
```

### Celula de Codigo 6 (notebook cell 12)

```python
# ── Benchmark table ──────────────────────────────────────────────────────────
bench_cols = [
    'model_name',
    'val_top15_precision_at_k',
    'val_top15_recall_at_k',
    'val_top15_lift_vs_random',
    'test_top15_precision_at_k',
    'test_top15_recall_at_k',
    'test_top15_lift_vs_random',
    'val_auc_pr',
]

bench = benchmark_df[bench_cols].copy()
bench = bench.sort_values(
    ['val_top15_recall_at_k', 'val_top15_precision_at_k', 'val_top15_lift_vs_random', 'val_auc_pr'],
    ascending=False,
)

rename_cols = {
    'val_top15_precision_at_k':  'Val Precision@15',
    'val_top15_recall_at_k':     'Val Recall@15',
    'val_top15_lift_vs_random':  'Val Lift@15',
    'test_top15_precision_at_k': 'Test Precision@15',
    'test_top15_recall_at_k':    'Test Recall@15',
    'test_top15_lift_vs_random': 'Test Lift@15',
    'val_auc_pr':                'Val AUC-PR',
}

display(bench.rename(columns=rename_cols).style.format(precision=4))

# ── Plot ──────────────────────────────────────────────────────────────────────
metric_map = {
    'val_top15_recall_at_k':     ('Recall@15',    PALETTE['teal']),
    'val_top15_precision_at_k':  ('Precision@15', PALETTE['blue']),
    'val_top15_lift_vs_random':  ('Lift@15',      PALETTE['orange']),
}

plot_df = bench.melt(
    id_vars='model_name',
    value_vars=list(metric_map.keys()),
    var_name='metric_key',
    value_name='Valor',
)
plot_df['Métrica'] = plot_df['metric_key'].map({k: v[0] for k, v in metric_map.items()})
palette  = {v[0]: v[1] for v in metric_map.values()}
order    = bench['model_name'].tolist()           # mantém ordem do ranking

fig, ax = plt.subplots(figsize=(13, 5))

sns.barplot(
    data=plot_df,
    x='model_name',
    y='Valor',
    hue='Métrica',
    palette=palette,
    order=order,
    ax=ax,
    width=0.7,
    edgecolor='white',
    linewidth=0.6,
)

# ── Rótulos de valor sobre cada barra ────────────────────────────────────────
for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.3f',
        label_type='edge',
        fontsize=7.5,
        padding=2,
        color='#444',
    )

# ── Linha de referência: recall do modelo aleatório (lift = 1 → recall = base rate)
# Comente a linha abaixo se não quiser ou ajuste o valor
# ax.axhline(y=base_rate, color='grey', lw=1, ls='--', label='Random baseline')

# ── Estética ──────────────────────────────────────────────────────────────────
ax.set_title(
    'Benchmark por scorecard operacional · Top-15 Tag-dia (validação)',
    fontsize=13, fontweight='bold', pad=14,
)
ax.set_xlabel('Modelo', fontsize=11)
ax.set_ylabel('Valor da métrica', fontsize=11)
ax.tick_params(axis='x', rotation=35, labelsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.set_ylim(0, plot_df['Valor'].max() * 1.15)   # espaço para rótulos
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.35, linestyle='--', lw=0.7)
ax.legend(title='Métrica (validação)', title_fontsize=9, fontsize=9, framealpha=0.6)

# Destaque visual no modelo campeão (primeira barra de cada grupo)
ax.axvspan(-0.5, 0.5, color='gold', alpha=0.07, zorder=0, label='_nolegend_')

add_source_note(ax)
plt.tight_layout()
plt.show()

# ── Insight ───────────────────────────────────────────────────────────────────
display(Markdown(
    f"**Insight + implicação operacional:** o modelo campeão foi **`{winner_name}`**, "
    f"com **Recall@Top15 = `{winner_val_recall_top15:.4f}`** na validação — "
    f"captura {winner_val_recall_top15:.1%} dos positivos verdadeiros dentro dos 15 primeiros. "
    f"A **Precision@15 = `{winner_val_precision_top15:.4f}`** e o "
    f"**Lift@15 = `{winner_val_lift_top15:.2f}×`** confirmam ganho expressivo sobre amostragem aleatória. "
    f"O AUC-PR (`{winner_val_auc_pr:.4f}`) serve como diagnóstico técnico auxiliar de discriminação global."
))
```

### Celula de Codigo 7 (notebook cell 14)

```python
threshold = threshold_test['threshold']
curve = threshold_curve_df.copy()
closest_idx = (curve['threshold'] - threshold).abs().idxmin()
selected = curve.loc[closest_idx]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Curva recall/precision
ax = axes[0]
ax.plot(curve['threshold'], curve['recall'], label='Recall', color=PALETTE['teal'], linewidth=2)
ax.plot(curve['threshold'], curve['precision'], label='Precision', color=PALETTE['orange'], linewidth=2)
ax.axvline(threshold, color=PALETTE['red'], linestyle='--', label=f'Threshold = {threshold:.4f}')
ax.set_title('Trade-off Recall vs Precision por threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Métrica')
ax.legend()
add_source_note(ax)

# Matriz de confusão
ax = axes[1]
rows = int(threshold_test['rows'])
tp = int(threshold_test['true_positive'])
fp = int(threshold_test['false_positive'])
fn = int(threshold_test['false_negative'])
tn = rows - tp - fp - fn
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_title('Matriz de confusão (teste, threshold operacional)')
ax.set_xlabel('Predito')
ax.set_ylabel('Real')
ax.set_xticklabels(['0', '1'])
ax.set_yticklabels(['0', '1'], rotation=0)

plt.tight_layout()
plt.show()

display(Markdown(
    f"**Insight + implicação operacional:** com threshold `{threshold:.4f}`, o modelo mantém recall alto (`{threshold_test['recall']:.3f}`) com custo de falso positivo controlado via estratégia de priorização TopK."
))
```

### Celula de Codigo 8 (notebook cell 16)

```python
imp = feature_importance_df.copy().head(12)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=imp, y='feature', x='importance', color=PALETTE['navy'], ax=ax)
ax.set_title('Top variáveis explicativas (proxy de importância)')
ax.set_xlabel('Importância relativa')
ax.set_ylabel('Feature')
add_source_note(ax)
plt.tight_layout()
plt.show()

business_translation = {
    'n_alertas_24h': 'Histórico recente de alertas críticos do equipamento.',
    'n_alertas_4h': 'Sinal imediato de deterioração no curto prazo.',
    'alertas_por_hora_24h': 'Intensidade normalizada de alertas ao longo do dia.',
    'duracao_ciclo_min': 'Mudança no padrão operacional do ciclo.',
    'dias_desde_ultimo_alerta': 'Recência de evento crítico anterior.',
    'Tag_freq': 'Comportamento histórico da tag no conjunto.',
    'Classe_target_enc': 'Risco histórico associado à classe de atividade.',
}

translation_df = imp[['feature']].copy()
translation_df['Leitura de negócio'] = translation_df['feature'].map(business_translation).fillna(
    'Variável técnica de apoio ao ranking de risco.'
)
display(translation_df)

display(Markdown(
    '**Insight + implicação operacional:** os principais sinais estão ligados a histórico e aceleração de alertas; isso permite justificar ações preventivas com base em evidência de comportamento operacional.'
))
```

### Celula de Codigo 9 (notebook cell 18)

```python
topk = oper_topk_df.query("split == 'test'").copy().sort_values('top_k_tags_per_day')
budget = oper_budget_df.query("split == 'test'").copy().sort_values('budget_pct')
top15_row = topk.loc[topk['top_k_tags_per_day'] == 15].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# TopK: precision/recall ficam em proporcao; lift usa eixo secundario.
ax = axes[0]
ax.plot(
    topk['top_k_tags_per_day'],
    topk['precision_at_k'],
    marker='o',
    color=PALETTE['blue'],
    label='Precision@K',
)
ax.plot(
    topk['top_k_tags_per_day'],
    topk['recall_at_k'],
    marker='o',
    color=PALETTE['teal'],
    label='Recall@K',
)
ax.axvline(15, color=PALETTE['red'], linestyle='--', linewidth=1.5, label='Top15 escolhido')
ax.scatter(
    [15, 15],
    [top15_row['precision_at_k'], top15_row['recall_at_k']],
    color=PALETTE['red'],
    zorder=4,
)
ax.set_title('Precision e recall por TopK Tag-dia')
ax.set_xlabel('TopK Tags/dia')
ax.set_ylabel('Precision / Recall')
ax.set_ylim(0, 1)

ax_lift = ax.twinx()
ax_lift.plot(
    topk['top_k_tags_per_day'],
    topk['lift_vs_random'],
    marker='s',
    color=PALETTE['orange'],
    label='Lift@K',
)
ax_lift.set_ylabel('Lift vs aleatório')
ax_lift.set_ylim(0, max(topk['lift_vs_random']) * 1.2)

lines, labels = ax.get_legend_handles_labels()
lift_lines, lift_labels = ax_lift.get_legend_handles_labels()
ax.legend(lines + lift_lines, labels + lift_labels, loc='lower right')
add_source_note(ax)

# Budget: precision/recall e lift tambem em escalas separadas.
ax = axes[1]
ax.plot(
    budget['budget_pct'] * 100,
    budget['precision_at_budget'],
    marker='o',
    color=PALETTE['blue'],
    label='Precision',
)
ax.plot(
    budget['budget_pct'] * 100,
    budget['recall_at_budget'],
    marker='o',
    color=PALETTE['teal'],
    label='Recall',
)
ax.set_title('Precision e recall por orçamento de alerta (teste)')
ax.set_xlabel('% de ciclos priorizados')
ax.set_ylabel('Precision / Recall')
ax.set_ylim(0, 1)

ax_lift = ax.twinx()
ax_lift.plot(
    budget['budget_pct'] * 100,
    budget['lift_vs_random'],
    marker='s',
    color=PALETTE['orange'],
    label='Lift',
)
ax_lift.set_ylabel('Lift vs aleatório')
ax_lift.set_ylim(0, max(budget['lift_vs_random']) * 1.2)

lines, labels = ax.get_legend_handles_labels()
lift_lines, lift_labels = ax_lift.get_legend_handles_labels()
ax.legend(lines + lift_lines, labels + lift_labels, loc='upper right')
add_source_note(ax)

plt.tight_layout()
plt.show()

topk_table = topk[
    ['top_k_tags_per_day', 'alerts_per_day', 'precision_at_k', 'recall_at_k', 'lift_vs_random']
].rename(
    columns={
        'top_k_tags_per_day': 'TopK Tags/dia',
        'alerts_per_day': 'Alertas/dia',
        'precision_at_k': 'Precision@K',
        'recall_at_k': 'Recall@K',
        'lift_vs_random': 'Lift@K',
    }
)
display(topk_table.round(4))

display(Markdown(
    f"**Insight + implicação operacional:** Top15 é o melhor ponto de equilíbrio: "
    f"mantém Precision@15 em `{top15_row['precision_at_k']:.3f}`, "
    f"eleva Recall@15 para `{top15_row['recall_at_k']:.3f}` e preserva "
    f"Lift@15 de `{top15_row['lift_vs_random']:.2f}×`, sem expandir a rotina para Top20."
))
```

### Celula de Codigo 10 (notebook cell 20)

```python
strong = pd.DataFrame(segment_json.get('strongest_top15_segments', []))
weak = pd.DataFrame(segment_json.get('weakest_top15_segments', []))
inconc = pd.DataFrame(segment_json.get('inconclusive_segments', []))

segment_precision = (
    segment_threshold_df.groupby('segment_col', as_index=False)
    .agg(precision_media=('precision', 'mean'), recall_medio=('recall', 'mean'))
)

fig, ax = plt.subplots(figsize=(8, 5))
plot_df = segment_precision.melt(id_vars='segment_col', value_vars=['precision_media', 'recall_medio'], var_name='Métrica', value_name='Valor')
sns.barplot(data=plot_df, x='segment_col', y='Valor', hue='Métrica', palette=[PALETTE['blue'], PALETTE['teal']], ax=ax)
ax.set_title('Desempenho médio no threshold por segmento')
ax.set_xlabel('Segmento')
ax.set_ylabel('Valor médio')
add_source_note(ax)
plt.tight_layout()
plt.show()

print('Segmentos fortes (Top15):')
display(strong.head(8))

print('Segmentos frágeis (Top15):')
display(weak.head(8))

print('Segmentos inconclusivos:')
display(inconc.head(8))

print('Hotspots por Tag:')
display(segment_hotspots_df.head(10))

display(Markdown(
    '**Insight + implicação operacional:** há segmentos com desempenho desigual; a governança deve manter exceções para baixa prevalência e criar plano dedicado de mitigação por segmento crítico.'
))
```

## 6. Observacoes

- Este arquivo foi gerado a partir do notebook executado em `notebooks/main.ipynb`.
- Os outputs graficos e tabelas renderizadas nao foram embutidos aqui; o foco e preservar o codigo executavel e o contexto do projeto.
- Para revisar os resultados renderizados, use o notebook original ou os relatorios em `reports/` e `docs/`.
