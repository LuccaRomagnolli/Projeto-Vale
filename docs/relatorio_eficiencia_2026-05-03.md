# Relatorio de Eficiencia e Consistencia

Data: 03/05/2026

## 1) Objetivo

Avaliar a eficiencia ponta a ponta do projeto, testar todas as etapas da
pipeline e verificar se os resultados estao satisfatorios para operacao diaria
na Vale, usando como referencia o documento `Estudo Guiado - Analise Avancada de Dados.pdf`.

## 2) Escopo de teste executado

Comandos executados nesta avaliacao:

1. `make run-all`
2. `make test`
3. `make infer`

Status geral: `APROVADO`.

### Etapas do pipeline (`make run-all`)

| Etapa | Status | Evidencia principal |
|---|---|---|
| Ingestao + rotulacao (`label`) | OK | `data/processed/labeled/quality_report_ingestao.json`, `data/processed/labeled/labeling_report.json` |
| EDA (`eda`) | OK | `reports/eda_report.md` + figuras |
| Features (`features`) | OK | `data/processed/features/feature_report.json` |
| Baseline + modelo principal (`train`) | OK | `reports/baseline_report.json`, `reports/model_principal_report.json` |
| Benchmark (`benchmark`) | OK | `reports/model_benchmark_report.json` |
| Tuning + backtest (`tune-hist-gbdt`) | OK | `reports/hist_gbdt_tuning_report.json`, `reports/hist_gbdt_backtest_report.csv` |
| Gate de estabilidade (`gate-stability`) | OK | `src.models.stability_gate` retornou aprovado |
| Avaliacao operacional (`evaluate`) | OK | `reports/operational_metrics_report.json` |
| Avaliacao segmentada (`evaluate-segments`) | OK | `reports/segment_operational_report.json` |

### Testes automatizados (`make test`)

| Indicador | Resultado |
|---|---|
| Total de testes | `55` |
| Passou/Falhou | `55 passed` |
| Cobertura global (`src`) | `80%` |

Pontos com cobertura mais baixa:

- `src/models/tune_hist_gbdt.py`: `55%`
- `src/evaluation/evaluate_model.py`: `56%`
- `src/alert_labeler.py`: `67%`

### Inferencia operacional (`make infer`)

| Indicador | Resultado |
|---|---|
| Entrada | `data/processed/features/splits/features_test.parquet` |
| Artefato | `models/hist_gbdt_tuned.joblib` |
| Linhas processadas | `56687` |
| Threshold aplicado | `0.141388` |
| Features ausentes preenchidas | `0` |

## 3) Eficiencia dos resultados

## 3.1 Dados e preparo

| Indicador | Resultado |
|---|---|
| Linhas com features | `377907` |
| Colunas de features | `57` |
| Positivos `target_4h` | `70811` |
| Eventos criticos finais | `107002` |

Leitura: volume e cobertura temporal sao suficientes para operacao de ranking
diario e para backtesting com multiplas janelas.

## 3.2 Baseline vs modelos supervisionados

| Modelo | Metrica de referencia | Resultado |
|---|---|---|
| Baseline heuristico | `AUC-PR teste` | `0.1498` |
| Modelo principal (LightGBM) | `AUC-PR teste` | `0.2810` |
| Campeao benchmark (`hist_gbdt_balanced`) | `Recall@Top15 val` | `0.7493` |
| Melhor tuning (`hist_gbdt_tuned_04`) | `Recall@Top15 val` | `0.7520` |

Leitura: houve ganho claro sobre baseline e consistencia entre benchmark/tuning
na metrica operacional primaria.

## 3.3 Eficiencia operacional (criterio executivo)

### TopK Tag-dia (teste)

| K | Precision@K | Recall@K | Lift@K | Alertas/dia |
|---:|---:|---:|---:|---:|
| 10 | 0.6767 | 0.4915 | 2.0808 | 10 |
| 15 | 0.6800 | 0.7409 | 2.0910 | 15 |
| 20 | 0.6150 | 0.8935 | 1.8912 | 20 |

Leitura: `Top15` entrega o melhor equilibrio operacional para rotina diaria
(boa cobertura sem inflar demais o volume).

### Threshold ciclo-a-ciclo (teste)

| Metrica | Valor |
|---|---:|
| Precision | 0.2727 |
| Recall | 0.8006 |
| Lift | 1.8208 |
| Predicted positive rate | 0.4397 |

Leitura: no nivel de ciclo, a taxa de alerta ainda e alta para uso direto sem
priorizacao. O uso recomendado continua sendo painel `TopK Tag-dia`.

## 3.4 Consistencia temporal

Gate de estabilidade (operacional `Top15 Tag-dia`):

- `std(test_top15_recall_at_k) = 0.0107` (limite `<= 0.03`)
- `std(test_top15_precision_at_k) = 0.0254` (limite `<= 0.05`)

Status: `APROVADO`.

Observacao tecnica: no backtest ciclo-a-ciclo ainda existe variacao maior em
recall (`std(test_recall)=0.0416`), reforcando que a governanca deve continuar
centrada na metrica operacional.

## 3.5 Consistencia por segmento

Achados principais:

1. Turnos apresentam boa robustez em Top15 (lift acima de 2x nos segmentos mais fortes).
2. Segmento `Classe=Hibernando` aparece como `inconclusivo_baixa_prevalencia`.
3. Total de segmentos inconclusivos no relatorio atual: `3`.

Leitura: resultado satisfatorio para operacao com governanca, desde que os
segmentos inconclusivos fiquem em trilha dedicada.

## 4) Aderencia ao Estudo Guiado (resumo)

Checagem contra os Conteudos Minimos:

1. Validacao temporal e anti-leakage: `OK` (split temporal e backtesting).
2. Baseline comparativo: `OK`.
3. Modelagem com multiplos candidatos e tuning: `OK`.
4. Avaliacao com metricas tecnicas e impacto operacional: `OK`.
5. Analise de erros/segmentos e estabilidade temporal: `OK`.
6. Controle formal de alteracoes metodologicas: `OK` (`docs/controle_alteracoes.md`).

## 5) Veredito para operacao Vale

Status para uso no dia a dia: `SATISFATORIO PARA PILOTO OPERACIONAL ASSISTIDO`.

Conclusao executiva:

1. O projeto esta tecnicamente estavel e operacionalmente util para priorizacao diaria.
2. `Top15 Tag-dia` atende os criterios de equilibrio entre cobertura e capacidade operacional.
3. Ainda nao e recomendado uso totalmente autonomo sem governanca de segmentos e monitoramento de drift.

## 6) Recomendacoes para melhorar consistencia

Prioridade alta (curto prazo):

1. Adicionar teste automatizado de regressao de metricas operacionais (`Top15`) com limites minimos em CI.
2. Formalizar monitoramento diario de drift (prevalencia, precision@15, recall@15, lift@15) com alerta de degradacao.
3. Tratar segmentos `inconclusivo_*` com politica clara: fallback heuristico ou exclusao de decisao automatica.

Prioridade media:

1. Aumentar folds de backtest temporal (ex: 5+) para reduzir sensibilidade a uma janela especifica.
2. Rodar calibracao segmentada controlada (por `turno` e `Classe`) sem quebrar governanca global.
3. Expandir cobertura de testes em `tune_hist_gbdt.py` e `evaluate_model.py` para cenarios de borda operacional.

Prioridade estrutural:

1. Introduzir banda de confianca (bootstrap) para `precision@15` e `recall@15` no relatorio executivo.
2. Versionar snapshot de dados e hash de artefatos no relatorio final para rastreabilidade completa.
3. Integrar custo operacional explicito (FN vs FP) na selecao de threshold complementar ao TopK.

## 7) Evidencias consultadas

- `reports/model_benchmark_report.json`
- `reports/hist_gbdt_tuning_report.json`
- `reports/hist_gbdt_backtest_report.csv`
- `reports/operational_metrics_report.json`
- `reports/segment_operational_report.json`
- `reports/operational_budget_metrics.csv`
- `reports/operational_daily_topk_metrics.csv`
- `reports/baseline_report.json`
- `reports/model_principal_report.json`
- `docs/controle_alteracoes.md`
- `Estudo Guiado - Analise Avancada de Dados.pdf`
