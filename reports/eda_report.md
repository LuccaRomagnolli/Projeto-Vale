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

- `/Users/luccaromagnolli/Desktop/Projeto Vale/reports/figures/eda_target_distribution.png`
- `/Users/luccaromagnolli/Desktop/Projeto Vale/reports/figures/eda_ciclos_por_hora.png`
- `/Users/luccaromagnolli/Desktop/Projeto Vale/reports/figures/eda_top_frotas.png`
- `/Users/luccaromagnolli/Desktop/Projeto Vale/reports/figures/eda_duracao_ciclo_hist.png`
- `/Users/luccaromagnolli/Desktop/Projeto Vale/reports/figures/eda_top_classes.png`

## Decisões e próximos passos

1. Revisar cobertura de eventos críticos para destravar positivos no `target_4h`.
2. Validar sincronização temporal entre apontamentos e telemetria.
3. Seguir para Etapa 5 apenas após estabilizar a rotulação com taxa de positivos não nula.
