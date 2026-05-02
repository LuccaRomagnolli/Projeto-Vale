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

| Modelo | Val P@15 | Val R@15 | Val Lift@15 | Test P@15 | Test R@15 | Val AUC-PR tecnica |
|---|---:|---:|---:|---:|---:|---:|
| `hist_gbdt_balanced` | 0.6619 | 0.7493 | 2.1231 | 0.6822 | 0.7433 | 0.2523 |
| `hist_gbdt_regularized` | 0.6548 | 0.7412 | 2.1002 | 0.6689 | 0.7288 | 0.2890 |
| `lightgbm_balanced` | 0.6524 | 0.7385 | 2.0925 | 0.6756 | 0.7361 | 0.2498 |
| `logistic_regression_balanced` | 0.6238 | 0.7062 | 2.0009 | 0.6733 | 0.7337 | 0.2344 |
| `extra_trees_balanced` | 0.6238 | 0.7062 | 2.0009 | 0.6578 | 0.7167 | 0.2232 |
| `random_forest_balanced` | 0.6214 | 0.7035 | 1.9933 | 0.6689 | 0.7288 | 0.2390 |

Leitura:

1. O `hist_gbdt_balanced` venceu a validacao operacional `Top15 Tag-dia`.
2. O `hist_gbdt_regularized` manteve a melhor AUC-PR tecnica, mas essa metrica agora e auxiliar.
3. Todos os principais candidatos ficaram acima da meta de `Recall@Top15 Tag-dia >= 0.70` no teste.
4. O teste continua fora da selecao do campeao para evitar escolha oportunista.

## Tuning dedicado do HistGBDT

Foram testados 6 candidatos. O vencedor foi:

```text
hist_gbdt_tuned_04
```

Parametros:

| Parametro | Valor |
|---|---:|
| `learning_rate` | 0.04 |
| `max_iter` | 420 |
| `max_leaf_nodes` | 21 |
| `min_samples_leaf` | 120 |
| `l2_regularization` | 2.0 |
| `class_weight` | `balanced` |

Metricas do vencedor:

| Split | P@15 Tag-dia | R@15 Tag-dia | Lift@15 | Recall ciclo | Precision ciclo | AUC-PR tecnica |
|---|---:|---:|---:|---:|---:|---:|
| Validacao | 0.6643 | 0.7520 | 2.1307 | 0.8015 | 0.2139 | 0.2450 |
| Teste | 0.6800 | 0.7409 | 2.0910 | 0.8006 | 0.2727 | 0.2589 |

## Backtesting temporal

O backtesting usa folds expansivos. O objetivo nao e escolher modelo por esses
folds, mas verificar estabilidade em diferentes periodos.

| Fold | Test R@15 | Test P@15 | Test Lift@15 | Test Recall ciclo | Test Precision ciclo | Test AUC-PR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.7500 | 0.6842 | 2.1658 | 0.8612 | 0.2885 | 0.3217 |
| 2 | 0.7265 | 0.6246 | 2.0470 | 0.7627 | 0.1914 | 0.2031 |
| 3 | 0.7283 | 0.6700 | 2.0634 | 0.7889 | 0.2465 | 0.2515 |

Resumo:

| Metrica | Media |
|---|---:|
| Test Recall ciclo | 0.8043 |
| Test Precision ciclo | 0.2421 |
| Test AUC-PR | 0.2588 |

Leitura:

1. O `Recall@Top15 Tag-dia` ficou estavel entre folds e consistente com a meta operacional.
2. A precision operacional varia menos que a precision ciclo-a-ciclo, reforcando o uso do painel TopK.
3. O fold 2 e o periodo mais critico e deve ser investigado por Frota, Tipo e Tag.

## Gate de estabilidade temporal

O projeto passa a bloquear promocao quando a variabilidade entre folds fica alta.

Comando:

```bash
make gate-stability
```

Limites padrao:

- `std(test_top15_recall_at_k) <= 0.03`
- `std(test_top15_precision_at_k) <= 0.05`

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

1. Manter `hist_gbdt_tuned_04` como candidato operacional.
2. Usar threshold `0.1414` como padrao balanceado.
3. Priorizar o painel `Top15 Tag-dia` em vez de acionar alertas por threshold global.
4. Investigar o fold 2 do backtesting para entender queda de precision.
5. Proxima etapa: avaliacao segmentada por Frota, Tipo, Tag e turno.

## Status

Status: `CONCLUIDA`
