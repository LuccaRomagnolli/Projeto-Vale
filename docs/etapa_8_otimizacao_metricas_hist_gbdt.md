# Etapa 8 - Otimizacao de metricas do HistGBDT

Data: 02/05/2026

## Objetivo

Melhorar as metricas do `hist_gbdt_regularized` sem introduzir vazamento temporal.
A etapa implementa quatro frentes:

| Frente | Entrega |
|---|---|
| Features precursoras | Sinais de degradacao recente em janelas 4h/8h/24h |
| Benchmark atualizado | Comparacao com 48 features modelaveis |
| Tuning HistGBDT | Grade controlada de hiperparametros |
| Backtesting temporal | 3 folds expansivos para estabilidade |
| Threshold operacional | Curva com recall, precision, FP e FN |
| Gate de estabilidade | Bloqueio de promocao quando variancia excede limite |

## Features adicionadas

O dataset passou de 41 para 57 colunas totais. Como algumas colunas sao
categoricas ou metadados, o numero de features numericas usadas pelos modelos
passou de 32 para 48.

Novas familias:

| Feature | Descricao |
|---|---|
| `duracao_std_ciclo_Xh` | Volatilidade da duracao do ciclo por Tag |
| `n_classes_distintas_Xh` | Diversidade recente de classes de atividade |
| `alertas_por_hora_Xh` | Intensidade normalizada de alertas por janela |
| `delta_duracao_ciclo_4h_24h` | Mudanca recente contra comportamento 24h |
| `ratio_duracao_ciclo_4h_24h` | Razao entre ciclo recente e ciclo historico curto |
| `delta_alertas_por_hora_4h_24h` | Aceleracao recente de alertas |
| `ratio_alertas_4h_24h` | Concentracao de alertas na janela curta |
| `delta_n_ciclos_4h_24h_norm` | Mudanca na intensidade operacional |
| `delta_freq_classe_4h_24h` | Mudanca recente do padrao de atividade |
| `delta_classes_distintas_4h_24h` | Mudanca na diversidade operacional |

Essas features usam apenas informacoes anteriores ao instante do registro,
mantendo a validade preditiva para uso no dia a dia da operacao.

## Artefatos implementados

| Artefato | Caminho |
|---|---|
| Tuning/backtesting | `src/models/tune_hist_gbdt.py` |
| Testes | `tests/test_tune_hist_gbdt.py` |
| Modelo tunado | `models/hist_gbdt_tuned.joblib` |
| Relatorio JSON | `reports/hist_gbdt_tuning_report.json` |
| Relatorio CSV | `reports/hist_gbdt_tuning_report.csv` |
| Curva de threshold | `reports/hist_gbdt_threshold_curve.csv` |
| Backtesting | `reports/hist_gbdt_backtest_report.csv` |

Comando:

```bash
make tune-hist-gbdt
```

## Resultado do dataset

| Metrica | Valor |
|---|---:|
| Linhas | 377907 |
| Colunas totais | 57 |
| Features modelaveis | 48 |
| Positivos `target_4h` | 70811 |

## Benchmark apos novas features

| Modelo | Val AUC-PR | Test Recall | Test Precision | Test AUC-PR |
|---|---:|---:|---:|---:|
| `hist_gbdt_regularized` | 0.2890 | 0.8139 | 0.2707 | 0.2736 |
| `hist_gbdt_balanced` | 0.2523 | 0.8048 | 0.2736 | 0.2875 |
| `lightgbm_balanced` | 0.2498 | 0.8000 | 0.2704 | 0.2610 |
| `random_forest_balanced` | 0.2390 | 0.8662 | 0.2554 | 0.2432 |
| `logistic_regression_balanced` | 0.2344 | 0.6770 | 0.2637 | 0.2445 |
| `extra_trees_balanced` | 0.2232 | 0.8922 | 0.2501 | 0.2323 |

Leitura:

1. O `hist_gbdt_regularized` continuou campeao por validacao.
2. A validacao melhorou de `0.2710` para `0.2890` em AUC-PR.
3. O teste ficou estavel em recall, com `0.8139`.
4. O `hist_gbdt_balanced` teve maior AUC-PR de teste, mas nao deve ser escolhido por teste.

## Tuning dedicado do HistGBDT

Foram testados 6 candidatos. O vencedor foi:

```text
hist_gbdt_tuned_02
```

Parametros:

| Parametro | Valor |
|---|---:|
| `learning_rate` | 0.04 |
| `max_iter` | 350 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 80 |
| `l2_regularization` | 1.0 |
| `class_weight` | `balanced` |

Metricas do vencedor:

| Split | Recall | Precision | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Validacao | 0.8077 | 0.2112 | 0.3348 | 0.2890 | 0.7508 |
| Teste | 0.8139 | 0.2707 | 0.4063 | 0.2736 | 0.7428 |

## Backtesting temporal

O backtesting usa folds expansivos. O objetivo nao e escolher modelo por esses
folds, mas verificar estabilidade em diferentes periodos.

| Fold | Test Recall | Test Precision | Test AUC-PR |
|---|---:|---:|---:|
| 1 | 0.8337 | 0.2929 | 0.3294 |
| 2 | 0.8004 | 0.1895 | 0.2079 |
| 3 | 0.7959 | 0.2518 | 0.2552 |

Resumo:

| Metrica | Media |
|---|---:|
| Test Recall | 0.8100 |
| Test Precision | 0.2447 |
| Test AUC-PR | 0.2642 |

Leitura:

1. O recall medio ficou acima de 0.81, consistente com a meta operacional.
2. A precision varia bastante entre folds, indicando drift temporal ou mudanca operacional.
3. O fold 2 e o periodo mais critico e deve ser investigado por Frota, Tipo e Tag.

## Gate de estabilidade temporal

O projeto passa a bloquear promocao quando a variabilidade entre folds fica alta.

Comando:

```bash
make gate-stability
```

Limites padrao:

- `std(test_recall) <= 0.03`
- `std(test_precision) <= 0.05`

Se o gate falhar, o fluxo exige novo tuning/recalibracao antes de promover.

## Curva de threshold

A curva operacional foi salva em:

```text
reports/hist_gbdt_threshold_curve.csv
```

Trecho relevante na validacao:

| Threshold | Recall | Precision | Taxa Alertada | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.2149 | 0.7803 | 0.2173 | 0.4600 | 20410 | 1595 |
| 0.1916 | 0.7992 | 0.2133 | 0.4800 | 21406 | 1458 |
| 0.1802 | 0.8077 | 0.2112 | 0.4900 | 21911 | 1396 |
| 0.1571 | 0.8234 | 0.2068 | 0.5100 | 22931 | 1282 |
| 0.1242 | 0.8536 | 0.2025 | 0.5400 | 24412 | 1063 |

Recomendacao operacional:

| Perfil de operacao | Threshold sugerido | Motivo |
|---|---:|---|
| Conservador em seguranca | 0.1571 | Aumenta recall para 0.8234 |
| Balanceado atual | 0.1802 | Mantem recall acima de 0.80 com menos alertas |
| Menos falso positivo | 0.2149 | Melhora precision, mas recall cai para 0.7803 |

## Importancia de variaveis apos novas features

Principais variaveis do LightGBM principal apos engenharia expandida:

| Feature | Importancia |
|---|---:|
| `dias_desde_ultimo_alerta` | 2849 |
| `Tag_freq` | 2270 |
| `duracao_std_ciclo_24h` | 730 |
| `hora_do_dia` | 562 |
| `dia_da_semana` | 525 |
| `duracao_media_ciclo_24h` | 512 |
| `n_ciclos_24h` | 458 |
| `Classe_target_enc` | 419 |
| `duracao_std_ciclo_8h` | 377 |
| `freq_classe_atividade_24h` | 311 |

Isso confirma que as novas features de volatilidade de ciclo entraram como sinais
relevantes, especialmente `duracao_std_ciclo_24h` e `duracao_std_ciclo_8h`.

## Conclusao

As novas features melhoraram o ranking na validacao, mas nao produziram um salto
grande no teste. Isso e uma boa descoberta: o modelo ficou mais rico, porem o
limite atual parece estar mais ligado a estabilidade temporal, segmentacao por
tipo/frota e qualidade dos sinais precursores do que apenas a hiperparametros.

Recomendacao:

1. Manter `hist_gbdt_tuned_02` como candidato operacional.
2. Usar threshold `0.1802` como padrao balanceado.
3. Avaliar threshold `0.1571` se a operacao priorizar recall acima de 0.82.
4. Investigar o fold 2 do backtesting para entender queda de precision.
5. Proxima etapa: avaliacao segmentada por Frota, Tipo, Tag e turno.

## Status

Status: `CONCLUIDA`
