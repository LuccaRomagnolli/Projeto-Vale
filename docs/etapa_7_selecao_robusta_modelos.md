<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-Selecao-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

# Etapa 7 - Selecao robusta de modelos

Data: 04/05/2026

## Objetivo

Selecionar tecnicamente o artefato operacional sem definir modelo principal a
priori. Quatro familias fortes competem sob o mesmo split temporal, tuning
Optuna, threshold calibrado somente na validacao e scorecard `Top15 Tag-dia`.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Selecao robusta | `src/models/model_selection.py` |
| Artefato selecionado | `models/model_selected.joblib` |
| Relatorio | `reports/model_selection/model_selection_report.json` |
| Trials | `reports/model_selection/model_selection_trials.csv` |
| Scores | `reports/model_selection/model_selection_scores.parquet` |
| Backtesting | `reports/model_selection/model_selection_backtest_report.csv` |
| Importancia de variaveis | `reports/model_selection/model_selected_feature_importance.csv` |
| Testes | `tests/test_model_selection.py` |

## Modelos avaliados

| Campo | Valor |
|---|---|
| Candidatos oficiais | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna`, `extra_trees_optuna` |
| Baseline diagnostico | `logistic_regression_baseline` |
| Trials por candidato | `30` |
| Folds de backtesting | `3` |
| TopK operacional primario | `15 Tags por dia` |

## Colunas removidas por vazamento ou identidade bruta

| Coluna | Motivo |
|---|---|
| `next_critical_event_time` | Informacao futura direta |
| `tte_horas` | Tempo ate evento futuro |
| `target_4h` | Variavel alvo |
| `Id`, `Inicio`, `Fim` | Identificadores e timestamps brutos |
| `Tag`, `Classe` | Substituidas por features historicas/encodings |

## Criterio de selecao

1. Maior `val_top15_recall_at_k`.
2. Desempate por `val_top15_precision_at_k`.
3. Desempate por `val_top15_lift_vs_random`.
4. Desempate final por `val_auc_pr`.

O conjunto de teste temporal permanece reservado para reporte final e nao
decide hiperparametros.

## Decisao

Status: `VIGENTE`. A etapa oficial nao possui modelo principal a priori; o
artefato operacional passa a ser `models/model_selected.joblib`, produzido por
`make model-selection`.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
