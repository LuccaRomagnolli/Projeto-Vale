# Politica unica de promocao de modelo

Data: 02/05/2026

## Objetivo

Padronizar a decisao de promocao para evitar ambiguidade entre "modelo principal"
e "campeao de benchmark".

Controle formal da alteracao metodologica:

- `docs/controle_alteracoes.md`

## Regras obrigatorias de promocao

Um modelo so pode ser promovido quando TODAS as condicoes abaixo forem verdadeiras:

1. Campeao por validacao temporal operacional no benchmark (`val_top15_recall_at_k`,
   desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e
   `val_auc_pr` apenas como diagnostico tecnico).
2. Metas operacionais TopK no teste temporal atendidas:
   - `Precision@Top15 Tag-dia >= 0.60`
   - `Recall@Top15 Tag-dia >= 0.70`
   - `Lift@Top15 Tag-dia >= 1.90`
3. Gate de estabilidade temporal aprovado (`make gate-stability`):
   - `std(test_top15_recall_at_k) <= 0.03`
   - `std(test_top15_precision_at_k) <= 0.05`
4. Segmentos de baixa prevalencia identificados como inconclusivos e fora da
   metrica principal de aceite.

## Fluxo operacional

1. `make benchmark`
2. `make tune-hist-gbdt`
3. `make gate-stability`
4. `make evaluate`
5. `make evaluate-segments`

Se qualquer etapa falhar, o modelo nao pode ser promovido.

## Evidencias minimas para auditoria

- `reports/model_benchmark_report.json`
- `reports/hist_gbdt_tuning_report.json`
- `reports/hist_gbdt_backtest_report.csv`
- `reports/operational_metrics_report.json`
- `reports/segment_operational_report.json`

## Regra para segmento raro

Segmentos com baixa amostra ou baixa prevalencia nao devem reprovar o modelo
global automaticamente. Eles entram como:

- `inconclusivo_baixa_amostra`
- `inconclusivo_baixa_prevalencia`

Esses casos exigem trilha de acao propria: coleta adicional, heuristica local ou
calibracao dedicada por segmento.
