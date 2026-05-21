# Relatório de EDA - Etapa 4

## Escopo

Análise exploratória orientada a decisão para o dataset rotulado `apontamentos_labeled.parquet`.

## Sumário Executivo

- Registros analisados: `377907`
- Tags únicas: `47`
- Frotas únicas: `5`
- Tipos únicos: `2`
- Positivos target_4h: `70811`
- Taxa de positivos: `18.737679%`

## Cobertura temporal

- Início mínimo: `2025-01-01 03:00:00+00:00`
- Início máximo: `2025-07-01 02:57:55+00:00`

## Duração de ciclo

- Média: `29.6854 min`
- Mediana: `22.9 min`
- P95: `60.0 min`

## Qualidade de dados (top 5 colunas com mais nulos)

| Coluna | % nulos |
|---|---:|
| next_critical_event_time | 19.3161% |
| tte_horas | 19.3161% |
| Id | 0.0% |
| Inicio | 0.0% |
| Fim | 0.0% |

## Figuras geradas

- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_target_distribution.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_ciclos_por_hora.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_top_frotas.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_duracao_ciclo_hist.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_top_classes.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_daily_volume_target_rate.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_target_rate_heatmap_hora_dia.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_top_tags_target_positives.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_target_rate_by_frota.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_target_rate_by_tipo_classe.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_duracao_por_classe_boxplot.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_tte_horas_positivos_hist.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_missing_values.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/eda_correlation_heatmap.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/model_selection_test_top15_metrics.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/model_selection_test_auc.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/operational_topk_precision_recall.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/operational_budget_precision_recall.png`
- `/Users/luccaromagnolli/Desktop/Projeto-Vale/reports/eda/figures/segments_top15_precision_recall.png`

## Decisões e próximos passos

1. Revisar cobertura de eventos críticos para destravar positivos no `target_4h`.
2. Validar sincronização temporal entre apontamentos e telemetria.
3. Seguir para Etapa 5 apenas após estabilizar a rotulação com taxa de positivos não nula.
