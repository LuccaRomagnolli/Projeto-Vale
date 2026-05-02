# Etapa 7 - Modelo supervisionado principal

Data: 02/05/2026

## Objetivo

Treinar um candidato supervisionado para superar o baseline heuristico da Etapa 6
e alimentar a politica de promocao definida por benchmark + estabilidade + TopK.

Esta etapa transforma o dataset com features temporais e historicas em um modelo
supervisionado de score de risco para a janela de predicao de 4 horas. A entrega
foi desenhada para ser utilizavel no dia a dia: separacao temporal ja congelada,
colunas de vazamento removidas, threshold calibrado em validacao e artefatos
persistidos para inferencia e auditoria.

## Entregaveis implementados

1. Pipeline de treino em `src/models/train_model.py`.
2. Selecao automatica de features numericas/booleanas.
3. Remocao explicita de colunas de vazamento:
   - `next_critical_event_time`
   - `tte_horas`
   - `target_4h`
   - identificadores e timestamps brutos.
4. Modelo principal:
   - LightGBM quando `lightgbm` esta instalado.
   - Fallback operacional para `sklearn.HistGradientBoostingClassifier` quando LightGBM nao esta disponivel no ambiente.
5. Threshold calibrado na validacao temporal.
6. Persistencia de artefatos:
   - `models/model_principal.joblib`
   - `reports/model_principal_report.json`
   - `reports/model_principal_scores.parquet`
   - `reports/model_feature_importance.csv`
7. Testes unitarios:
   - `tests/test_train_model.py`

## Modelo executado nesta maquina

O pipeline usa LightGBM como modelo principal quando a biblioteca esta instalada.
O ambiente atual foi corrigido com `lightgbm==4.6.0`, portanto o treino principal
foi reexecutado com `lightgbm.LGBMClassifier`.

| Campo | Valor |
|---|---|
| Modelo executado | `lightgbm.LGBMClassifier` |
| Versao instalada | `lightgbm==4.6.0` |
| Artefato | `models/model_principal.joblib` |
| Dataset de entrada | `data/processed/features/features_dataset.parquet` |
| Splits temporais | `data/processed/features/splits/` |
| Features usadas | 48 |
| Threshold calibrado | `0.10810492961676205` |
| Relatorio | `reports/model_principal_report.json` |
| Scores | `reports/model_principal_scores.parquet` |
| Importancia de variaveis | `reports/model_feature_importance.csv` |

Observacao importante: o fallback para `sklearn.HistGradientBoostingClassifier`
continua implementado. Ele existe para manter o pipeline executavel em ambientes
onde LightGBM ainda nao esteja instalado, mas o artefato atual do projeto ja foi
gerado com LightGBM.

## Features utilizadas

O treino selecionou apenas colunas numericas e booleanas, depois de remover
identificadores, timestamps brutos e qualquer coluna que represente informacao
futura.

Colunas removidas para evitar vazamento:

| Coluna | Motivo |
|---|---|
| `Id` | Identificador operacional, nao generalizavel |
| `Inicio` | Timestamp bruto, substituido por features temporais |
| `Fim` | Timestamp bruto, substituido por features temporais |
| `Tag` | Identificador bruto, substituido por `Tag_freq` |
| `Classe` | Categoria bruta, substituida por agregacoes/encoding |
| `next_critical_event_time` | Informacao futura direta |
| `tte_horas` | Tempo ate evento futuro, alvo alternativo de regressao |
| `target_4h` | Variavel alvo |

As 48 features efetivamente usadas foram:

```text
hora_do_dia
dia_da_semana
mes
is_fim_de_semana
duracao_ciclo_min
hora
dia_semana
n_ciclos_4h
duracao_media_ciclo_4h
duracao_std_ciclo_4h
freq_classe_atividade_4h
n_classes_distintas_4h
n_ciclos_8h
duracao_media_ciclo_8h
duracao_std_ciclo_8h
freq_classe_atividade_8h
n_classes_distintas_8h
n_ciclos_24h
duracao_media_ciclo_24h
duracao_std_ciclo_24h
freq_classe_atividade_24h
n_classes_distintas_24h
n_alertas_4h
alertas_por_hora_4h
n_alertas_8h
alertas_por_hora_8h
n_alertas_24h
alertas_por_hora_24h
dias_desde_ultimo_alerta
n_precondicoes_satisfeitas_4h
nivel_maximo_evento_recente
delta_duracao_ciclo_4h_24h
ratio_duracao_ciclo_4h_24h
delta_alertas_por_hora_4h_24h
ratio_alertas_4h_24h
delta_n_ciclos_4h_24h_norm
delta_freq_classe_4h_24h
delta_classes_distintas_4h_24h
Tag_freq
Operador_freq
Classe_target_enc
Frota_793-D 2S
Frota_793-D 3S
Frota_793-D 4S
Frota_793-D 5S
Frota_LeTourneau L 1850
Tipo_Caminhao
Tipo_Escavadeira
```

## Calibracao de threshold

O threshold nao foi escolhido no teste. Ele foi calibrado exclusivamente no
conjunto de validacao temporal, priorizando recall minimo operacional.

| Parametro | Valor |
|---|---|
| Recall minimo desejado na validacao | `0.80` |
| Threshold escolhido | `0.10810492961676205` |
| Recall obtido na validacao | `0.8018179314144057` |
| Precision obtida na validacao | `0.21397331765224742` |

Essa escolha reflete o custo assimetrico do problema: perder um alerta Don't Go
critico e mais grave do que gerar uma parada preventiva desnecessaria. Mesmo
assim, a etapa ja melhora a precisao em relacao ao baseline que praticamente
marca todos os casos como risco.

## Metricas finais

| Split | Recall | Precision | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Treino | 0.9996 | 0.2935 | 0.4538 | 0.7648 | 0.9283 |
| Validacao | 0.8018 | 0.2140 | 0.3378 | 0.2392 | 0.7360 |
| Teste | 0.7880 | 0.2660 | 0.3977 | 0.2810 | 0.7455 |

Leitura tecnica:

1. O modelo generaliza melhor que o baseline em ranking de risco, medido por AUC-PR.
2. O recall de teste ficou proximo da meta operacional inicial de 80%, chegando a `0.7880`.
3. A precision de teste subiu para `0.2660`, contra `0.1498` do baseline heuristico.
4. A diferenca entre AUC-PR de treino e validacao indica que a proxima etapa deve focar regularizacao, Optuna e validacao por janelas temporais adicionais.

## Comparacao com baseline

| Modelo | Split | Threshold | Recall | Precision | F1 | AUC-PR | AUC-ROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline heuristico | Teste | 0.0000 | 1.0000 | 0.1498 | 0.2605 | 0.1498 | 0.5000 |
| LightGBM supervisionado | Teste | 0.1081 | 0.7880 | 0.2660 | 0.3977 | 0.2810 | 0.7455 |

Conclusao: o baseline ainda captura todos os positivos porque o threshold ficou
em zero, mas isso tem custo operacional alto, pois equivale a alertar uma parcela
muito ampla da operacao. O modelo supervisionado reduz falsos positivos, melhora
o ranking de risco e entrega um score mais acionavel para o dispatcher.

## Importancia de variaveis

A importancia foi extraida de `feature_importances_` do LightGBM. Principais
sinais:

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

Interpretacao de negocio:

1. A identidade operacional agregada da Tag ainda carrega sinal forte, mesmo via frequency encoding.
2. A frota/modelo do equipamento parece separar perfis de risco de forma relevante.
3. Sinais temporais e padroes recentes de atividade contribuem para ordenar risco.
4. Features diretamente derivadas de precondicoes aparecem com baixa importancia neste treino, possivelmente por cobertura limitada das colunas de evento/situacao nos apontamentos.

## Comandos executados

1. `make lint`
2. `make test`
3. `make train`
4. `make benchmark`

Resultado:

```text
36 tests passed
coverage total: 87%
modelo principal: lightgbm.LGBMClassifier
recall teste: 0.787986
AUC-PR teste: 0.280971
campeao benchmark: hist_gbdt_regularized
recall teste campeao: 0.813899
AUC-PR validacao campeao: 0.288982
```

## Criterios de aceite

| Criterio | Status | Evidencia |
|---|---|---|
| Treino usa split temporal congelado | OK | `data/processed/features/splits/split_metadata.json` |
| Modelo nao usa alvo ou futuro como feature | OK | Lista de colunas excluidas no pipeline |
| Threshold calibrado fora do teste | OK | Calibracao feita no split de validacao |
| Artefato serializado | OK | `models/model_principal.joblib` |
| Scores auditaveis por split | OK | `reports/model_principal_scores.parquet` |
| Relatorio de metricas | OK | `reports/model_principal_report.json` |
| Importancia de variaveis | OK | `reports/model_feature_importance.csv` |
| Benchmark de modelos | OK | `reports/model_benchmark_report.json` |
| Testes automatizados passando | OK | `make test` com 36 testes |

## Riscos e proximas melhorias

| Risco | Impacto | Mitigacao recomendada |
|---|---|---|
| Ambiente atual ainda sem Optuna | Busca de hiperparametros ainda manual | Instalar `optuna` e rodar tuning temporal |
| LightGBM nao foi campeao do benchmark | Modelo principal e forte, mas nao foi o melhor por validacao | Considerar promover `hist_gbdt_regularized` ou otimizar LightGBM |
| Gap treino-validacao em AUC-PR | Indicio de overfitting parcial | Regularizacao, tuning e validacao temporal por multiplas janelas |
| Forte dependencia de `Tag_freq` | Risco de piorar em equipamentos novos | Adicionar features fisicas/telemetricas mais explicativas por equipamento |

## Regra de promocao

Esta etapa nao promove modelo sozinha. A promocao depende da politica em:

- `docs/politica_promocao_modelo.md`

Em especial:

1. vencedor por validacao no benchmark;
2. gate temporal aprovado em `make gate-stability`;
3. metas TopK operacionais atingidas;
4. avaliacao segmentada sem riscos ocultos em segmentos criticos.

## Status final

Status: `CONCLUIDA (candidato)`
