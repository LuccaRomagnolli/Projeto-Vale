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

# Controle de alteracoes metodologicas

Data de atualizacao: 04/05/2026

## Objetivo

Registrar decisoes metodologicas relevantes com estado anterior, estado
posterior, justificativa e evidencias.

## Alteracao 01 - Metrica primaria de selecao e promocao

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| Criterio principal | Maior `val_auc_pr`, com desempate por `val_precision` e `val_f1` | Maior `val_top15_recall_at_k`, com desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr` | A pergunta de negocio e priorizar equipamentos para manutencao dentro de capacidade diaria. |
| Unidade executiva | Ciclo individual | `Tag-dia`, usando maior score diario por equipamento | Reduz duplicidade e aproxima a metrica do fluxo operacional. |
| Segmentos | Analise concentrada apenas no fim | Scores de treino, benchmark e tuning carregam `Frota`, `Tipo`, `turno` e `Classe` | Garante consistencia entre benchmark, tuning, avaliacao e segmentacao. |
| Estabilidade | `std(test_recall)` e `std(test_precision)` ciclo-a-ciclo | `std(test_top15_recall_at_k)` e `std(test_top15_precision_at_k)` | Estabilidade deve ser medida na mesma metrica usada para promocao. |

## Impacto

| Item | Resultado |
|---|---|
| Campeao do benchmark | `hist_gbdt_balanced` |
| Melhor tuning HistGBDT | `hist_gbdt_tuned_04` |
| Artefato operacional | `models/hist_gbdt_tuned.joblib` |
| Threshold operacional | `0.141388104973226` |
| Test Precision@Top15 Tag-dia | `0.6800` |
| Test Recall@Top15 Tag-dia | `0.7409` |
| Test Lift@Top15 Tag-dia | `2.0910` |
| Gate de estabilidade | Aprovado (`recall_std=0.0107`, `precision_std=0.0254`) |

## Evidencias

| Evidencia | Caminho |
|---|---|
| Benchmark de modelos | `reports/model_benchmark_report.json` |
| Scores do benchmark com segmentos | `reports/model_benchmark_scores.parquet` |
| Tuning e backtesting | `reports/hist_gbdt_tuning_report.json` |
| Backtesting temporal | `reports/hist_gbdt_backtest_report.csv` |
| Avaliacao operacional | `reports/operational_metrics_report.json` |
| Analise segmentada | `reports/segment_operational_report.json` |
| Notebook executivo | `notebooks/main.ipynb` |

## Alteracao 02 - Padronizacao documental

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| Estrutura de `docs/` | Documentos deletados no working tree e formatacao desigual | Documentos recriados e padronizados com logo, cabecalho e secoes comuns | Facilitar leitura executiva, manutencao e auditoria. |
| Inferencia | Citada no README, sem etapa propria em `docs/` | Nova etapa `docs/etapa_11_inferencia_operacional.md` | Fechar a trilha ponta a ponta ate uso operacional. |
| Notebooks | Referencias historicas a notebooks `01..09` | Notebook oficial unico `notebooks/main.ipynb` | Alinhar documentacao a estrutura real do repositorio. |

## Decisao

Status: `VIGENTE`. A metrica executiva oficial permanece `TopK Tag-dia`, com
AUC-PR, precision e recall ciclo-a-ciclo como diagnosticos auxiliares.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
