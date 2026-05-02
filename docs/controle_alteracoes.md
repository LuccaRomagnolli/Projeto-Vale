# Controle de Alteracoes Metodologicas

Data: 02/05/2026

## Objetivo

Registrar decisoes metodologicas relevantes com estado anterior, estado posterior
e justificativa, conforme orientacao do estudo guiado.

## Alteracao 01 - Metrica primaria de selecao e promocao

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| Criterio principal do benchmark | Maior `val_auc_pr`, com desempate por `val_precision` e `val_f1` | Maior `val_top15_recall_at_k`, com desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr` | A pergunta de negocio e priorizar equipamentos para manutencao/inspecao. `Top15 Tag-dia` mede a utilidade operacional do ranking dentro de uma capacidade diaria, enquanto AUC-PR ciclo-a-ciclo fica como diagnostico tecnico. |
| Unidade de avaliacao executiva | Ciclo individual | `Tag-dia`, usando o maior score diario por equipamento | Reduz duplicidade de ciclos do mesmo equipamento e aproxima a avaliacao do fluxo real do dispatcher/manutencao. |
| Tratamento de segmentos | Analise segmentada concentrada na avaliacao operacional | Scores de treino, benchmark e tuning passam a carregar `Frota`, `Tipo`, `turno` e `Classe` | Garante que benchmark, tuning, avaliacao operacional e analise segmentada usem a mesma metodologia. |
| Gate de estabilidade | `std(test_recall)` e `std(test_precision)` ciclo-a-ciclo | `std(test_top15_recall_at_k)` e `std(test_top15_precision_at_k)` | Estabilidade deve ser medida na mesma metrica usada para promocao operacional. |

## Impacto da alteracao

| Item | Resultado apos alteracao |
|---|---|
| Campeao do benchmark | `hist_gbdt_balanced` |
| Melhor tuning HistGBDT | `hist_gbdt_tuned_04` |
| Threshold do artefato operacional | `0.141388` |
| Test Precision@Top15 Tag-dia | `0.6800` |
| Test Recall@Top15 Tag-dia | `0.7409` |
| Test Lift@Top15 Tag-dia | `2.0910` |
| Gate de estabilidade operacional | Aprovado (`recall_std=0.0107`, `precision_std=0.0254`) |

## Evidencias atualizadas

| Evidencia | Caminho |
|---|---|
| Benchmark de modelos | `reports/model_benchmark_report.json` |
| Scores do benchmark com segmentos | `reports/model_benchmark_scores.parquet` |
| Tuning e backtesting | `reports/hist_gbdt_tuning_report.json` |
| Backtesting temporal | `reports/hist_gbdt_backtest_report.csv` |
| Avaliacao operacional | `reports/operational_metrics_report.json` |
| Analise segmentada | `reports/segment_operational_report.json` |
| Notebook executivo | `notebooks/main.ipynb` |

## Decisao

A alteracao foi mantida porque melhora a coerencia entre pergunta de negocio,
avaliacao, selecao de modelo, estabilidade temporal e governanca de promocao.
AUC-PR, precision e recall ciclo-a-ciclo permanecem no projeto como metricas
tecnicas auxiliares, mas nao como criterio primario de promocao.
