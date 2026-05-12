# Dicionário de Dados — Antecipação de Alertas Críticos

Referência completa das variáveis, fontes e artefatos do projeto. Descreve o significado operacional de cada campo, sua origem e seu papel na modelagem.

---

## Sumário

- [Convenções](#convenções)
- [Fontes de dados](#fontes-de-dados)
- [Apontamentos brutos](#apontamentos-brutos)
- [Telemetria bruta](#telemetria-bruta)
- [Regras de negócio](#regras-de-negócio)
- [Base rotulada](#base-rotulada)
- [Eventos críticos](#eventos-críticos)
- [Features temporais e de calendário](#features-temporais-e-de-calendário)
- [Features rolling de ciclos](#features-rolling-de-ciclos)
- [Features rolling de alertas](#features-rolling-de-alertas)
- [Histórico, precondições e degradação](#histórico-precondições-e-degradação)
- [Features categóricas codificadas](#features-categóricas-codificadas)
- [Colunas removidas da matriz modelável](#colunas-removidas-da-matriz-modelável)
- [Splits temporais](#splits-temporais)
- [Scores de modelo](#scores-de-modelo)
- [Relatório de seleção de modelo](#relatório-de-seleção-de-modelo)
- [Métricas operacionais TopK](#métricas-operacionais-topk)
- [Métricas por orçamento de alertas](#métricas-por-orçamento-de-alertas)
- [Métricas segmentadas](#métricas-segmentadas)
- [Hotspots por tag](#hotspots-por-tag)
- [Alertas deduplicados para operação](#alertas-deduplicados-para-operação)
- [Como ler as métricas principais](#como-ler-as-métricas-principais)
- [Observações de uso](#observações-de-uso)

---

## Convenções

| Termo | Significado |
|---|---|
| **Ciclo** | Registro de uma atividade operacional de um equipamento, vindo da base de apontamentos. |
| **Evento de telemetria** | Registro de alarme ou evento operacional emitido pelo equipamento. |
| **Don't Go** | Alarme crítico que indica que o equipamento não deve operar sem avaliação ou intervenção. |
| **Tag** | Código de identificação do equipamento. |
| **Tag-dia** | Concatenação entre equipamento e dia operacional; usada para montar a lista diária TopK. |
| **Janela 4h** | Horizonte de antecipação — o modelo busca prever eventos críticos nas próximas 4 horas. |
| **Split temporal** | Divisão treino/validação/teste respeitando a ordem cronológica dos dados. |
| **Feature modelável** | Variável que pode entrar no modelo sem enxergar o futuro. |
| **Coluna de auditoria** | Variável mantida para análise e validação, mas removida da matriz de treino quando causa vazamento. |

---

## Fontes de dados

| Fonte | Caminho | Uso no projeto |
|---|---|---|
| Apontamentos brutos | `data/raw/datasets/apontamentos/desenvolver_apontamentos.parquet` | Base de ciclos operacionais. |
| Telemetria bruta | `data/raw/datasets/telemetria/*.parquet` | Eventos e alarmes dos equipamentos. |
| Dicionário oficial | `data/external/dicionario/Dicionario_Dados.xlsx` | Descrição original das colunas de apontamentos e telemetria. |
| Regras de negócio | `data/external/regras_negocio/Alarmes - Regra de Negocio.xlsx` | Fonte de verdade para alarmes críticos e Don't Go. |
| Base rotulada | `data/processed/labeled/apontamentos_labeled.parquet` | Ciclos com `target_4h`. |
| Eventos críticos | `data/processed/labeled/critical_events.parquet` | Eventos finais considerados críticos. |
| Base de features | `data/processed/features/features_dataset.parquet` | Dataset consolidado para EDA e modelagem. |
| Splits oficiais | `data/processed/features/splits/*.parquet` | Treino, validação e teste temporais. |
| Relatórios | `reports/` | Métricas, seleção de modelo, scorecards e análises segmentadas. |

---

## Apontamentos brutos

| Variável | Tipo | Descrição | Uso |
|---|---|---|---|
| `Id` | int64 | Identificador único do ciclo de apontamento. | Chave técnica e rastreabilidade. Não é feature modelável. |
| `Inicio` | datetime64 | Data e hora de início do apontamento. | Base para features temporais e ordenação. |
| `Fim` | datetime64 | Data e hora de término do apontamento. | Referência temporal principal para rotulação e splits. |
| `Tag` | object | Código de identificação do equipamento. | Unidade operacional priorizada no TopK. |
| `Frota` | object | Modelo ou frota do equipamento. | Segmentação e one-hot encoding. |
| `Tipo` | object | Tipo do equipamento, por exemplo caminhão ou escavadeira. | Segmentação e one-hot encoding. |
| `Classe` | object | Classificação da atividade ou ciclo. | EDA, segmentação e target encoding. |
| `Nome_Operador_Anon` | object | Código anonimizado do operador, no formato `OP_XXX`. | Pode originar encoding de frequência; dado sensível anonimizado. |
| `Matricula_Operador_Hash` | object | Hash da matrícula do operador. | Auditoria anonimizada; não deve ser usado sem tratamento. |

---

## Telemetria bruta

| Variável | Tipo | Descrição | Uso |
|---|---|---|---|
| `Id_Eventos_Telemetria` | int64 | Identificador único do evento de telemetria. | Rastreabilidade do evento. |
| `Data_Evento` | datetime64 | Data e hora do registro do evento. | Alinhamento temporal com ciclos. |
| `Inicio_Turno` | object | Data/hora de início do turno do evento. | Contexto operacional. |
| `Fim_Turno` | object | Data/hora de término do turno do evento. | Contexto operacional. |
| `Dia` | int64 | Dia do mês do evento. | Variável de calendário bruta. |
| `Localidade` | object | Mina ou localidade onde o equipamento opera. | Contexto e possíveis recortes futuros. |
| `TAG` | object | Código do equipamento na telemetria. | Ligação com `Tag` dos apontamentos. |
| `Tag_Frota` | object | Modelo ou frota do equipamento na telemetria. | Conferência e segmentação. |
| `Tipo` | object | Tipo do equipamento. | Segmentação. |
| `Nome_Operador_Anon` | object | Código anonimizado do operador. | Contexto anonimizado. |
| `Matricula_Operador_Hash` | object | Hash da matrícula do operador. | Contexto anonimizado. |
| `Id_Alarme` | int64 | Identificador do tipo de alarme disparado. | Ligação com catálogo e regras. |
| `Alarme` | object | Nome ou descrição do alarme. | Classificação de criticidade. |
| `Id_Criticidade` | int64 | Código numérico do nível de criticidade. | Criticidade do evento. |
| `Criticidade` | object | Descrição textual da criticidade. | Interpretação operacional. |
| `Valor` | object | Valor associado ao alarme no momento do evento. | Contexto do evento. |
| `Classe` | object | Estado do alarme, por exemplo `Activate` ou `Inactive`. | Filtro e contexto do evento. |
| `Is_Dont_Go` | int8 | Flag binária indicando se o alarme acionou (`1`) ou não (`0`). | Base para eventos críticos e rotulação. |

---

## Regras de negócio

Arquivo: `data/external/regras_negocio/Alarmes - Regra de Negocio.xlsx`

| Variável | Descrição |
|---|---|
| `TIPO` | Tipo de regra ou família do alarme. |
| `EVENTO` | Nome do evento ou alarme definido pela regra. |
| `SITUACAO` | Condição que caracteriza criticidade ou acionamento. |
| `QTD` | Quantidade de ocorrências necessária para a regra. |
| `TEMPO` | Janela temporal da regra, geralmente em minutos. |
| `NIVEL` | Nível de criticidade definido pelo negócio, como `Muito Alto`. |

---

## Base rotulada

Arquivos principais: `apontamentos_labeled.parquet` e `features_dataset.parquet`

| Variável | Tipo | Descrição | Papel na modelagem |
|---|---|---|---|
| `Id` | int64 | Identificador do ciclo. | Auditoria; removido da matriz modelável. |
| `Inicio` | datetime64[UTC] | Início do ciclo. | Auditoria e criação de features temporais; removido como timestamp bruto. |
| `Fim` | datetime64[UTC] | Fim do ciclo. | Referência para split temporal, target e features; removido como timestamp bruto. |
| `Tag` | object | Equipamento do ciclo. | Unidade de priorização; removido cru da matriz modelável. |
| `Frota` | object | Frota ou modelo do equipamento. | Usado antes do one-hot; disponível na base rotulada. |
| `Tipo` | object | Tipo do equipamento. | Usado antes do one-hot; disponível na base rotulada. |
| `Classe` | object | Classe da atividade ou ciclo. | Segmentação e encoding; removida crua do modelo final. |
| `next_critical_event_time` | datetime64[UTC] | Próximo evento crítico futuro associado ao equipamento. | Auditoria e rotulação — vazamento se usada como feature. |
| `tte_horas` | float64 | Time-to-event em horas até o próximo evento crítico. | Auditoria e rotulação — vazamento se usada como feature. |
| `target_4h` | int64 | Alvo binário: `1` se há evento crítico em até 4h, `0` caso contrário. | Variável resposta — nunca entra como feature. |

---

## Eventos críticos

Arquivo: `data/processed/labeled/critical_events.parquet`

| Variável | Tipo | Descrição |
|---|---|---|
| `TAG` | object | Equipamento associado ao evento crítico. |
| `EVENT_TIME` | datetime64[UTC] | Data e hora final do evento crítico usado para rotulação. |

---

## Features temporais e de calendário

| Variável | Tipo | Descrição | Uso |
|---|---|---|---|
| `hora_do_dia` | int32 | Hora do dia extraída do ciclo. | Captura padrões intradiários. |
| `hora` | int32 | Representação alternativa da hora do ciclo. | Compatibilidade com pipeline. |
| `dia_da_semana` | int32 | Dia da semana do ciclo. | Captura sazonalidade semanal. |
| `dia_semana` | int32 | Representação alternativa do dia da semana. | Compatibilidade com pipeline. |
| `mes` | int32 | Mês do ciclo. | Captura variação mensal. |
| `turno` | object | Turno operacional derivado do horário. | Segmentação e EDA. |
| `is_fim_de_semana` | int64 | Flag `1` para sábado/domingo, `0` caso contrário. | Sinal de calendário. |
| `duracao_ciclo_min` | float64 | Duração do ciclo em minutos: `Fim - Inicio`. | Sinal de ciclo atípico, gargalo ou condição operacional. |

---

## Features rolling de ciclos

Calculadas por equipamento em janelas retrospectivas. Usam apenas histórico anterior ao ciclo avaliado.

| Padrão de variável | Tipo | Descrição |
|---|---|---|
| `n_ciclos_{4h,8h,24h}` | float64 | Quantidade de ciclos recentes na janela. Mede intensidade operacional. |
| `duracao_media_ciclo_{4h,8h,24h}` | float64 | Duração média dos ciclos recentes. Mede ritmo e possíveis gargalos. |
| `duracao_std_ciclo_{4h,8h,24h}` | float64 | Desvio padrão da duração dos ciclos recentes. Mede variabilidade operacional. |
| `freq_classe_atividade_{4h,8h,24h}` | float64 | Frequência da classe de atividade recente. Mede repetição e concentração de uma classe. |
| `n_classes_distintas_{4h,8h,24h}` | float64 | Número de classes de atividade distintas na janela. Mede diversidade operacional recente. |

---

## Features rolling de alertas

| Variável | Tipo | Descrição |
|---|---|---|
| `n_alertas_4h` | int64 | Número de alertas anteriores nas últimas 4 horas. |
| `n_alertas_8h` | int64 | Número de alertas anteriores nas últimas 8 horas. |
| `n_alertas_24h` | int64 | Número de alertas anteriores nas últimas 24 horas. |
| `alertas_por_hora_4h` | float64 | Taxa de alertas por hora na janela de 4h. |
| `alertas_por_hora_8h` | float64 | Taxa de alertas por hora na janela de 8h. |
| `alertas_por_hora_24h` | float64 | Taxa de alertas por hora na janela de 24h. |

---

## Histórico, precondições e degradação

| Variável | Tipo | Descrição |
|---|---|---|
| `dias_desde_ultimo_alerta` | float64 | Tempo em dias desde o último alerta crítico conhecido antes do ciclo. |
| `n_precondicoes_satisfeitas_4h` | int64 | Quantidade de condições históricas recentes compatíveis com risco em 4h. |
| `nivel_maximo_evento_recente` | int64 | Maior nível de criticidade observado recentemente antes do ciclo. |
| `delta_duracao_ciclo_4h_24h` | float64 | Diferença entre duração média recente de 4h e referência de 24h. |
| `ratio_duracao_ciclo_4h_24h` | float64 | Razão entre duração média recente de 4h e referência de 24h. |
| `delta_alertas_por_hora_4h_24h` | float64 | Diferença entre taxa de alertas em 4h e 24h. |
| `ratio_alertas_4h_24h` | float64 | Razão entre alertas recentes de 4h e referência de 24h. |
| `delta_n_ciclos_4h_24h_norm` | float64 | Mudança normalizada do volume de ciclos em 4h contra 24h. |
| `delta_freq_classe_4h_24h` | float64 | Mudança da frequência de classe de atividade entre 4h e 24h. |
| `delta_classes_distintas_4h_24h` | float64 | Mudança na diversidade de classes entre 4h e 24h. |

---

## Features categóricas codificadas

| Variável | Tipo | Descrição | Observação |
|---|---|---|---|
| `Tag_freq` | float64 | Frequência histórica da `Tag` no conjunto de treino. | Substitui uso cru do identificador. |
| `Operador_freq` | float64 | Frequência histórica do operador anonimizado. | Codificação agregada, sem identificação direta. |
| `Classe_target_enc` | float64 | Encoding supervisionado da classe de atividade. | Deve ser calculado sem vazamento temporal. |
| `Frota_793-D 2S` | bool | Indicador one-hot para frota `793-D 2S`. | `True` quando o ciclo pertence à frota. |
| `Frota_793-D 3S` | bool | Indicador one-hot para frota `793-D 3S`. | `True` quando o ciclo pertence à frota. |
| `Frota_793-D 4S` | bool | Indicador one-hot para frota `793-D 4S`. | `True` quando o ciclo pertence à frota. |
| `Frota_793-D 5S` | bool | Indicador one-hot para frota `793-D 5S`. | `True` quando o ciclo pertence à frota. |
| `Frota_LeTourneau L 1850` | bool | Indicador one-hot para frota `LeTourneau L 1850`. | `True` quando o ciclo pertence à frota. |
| `Tipo_Caminhao` | bool | Indicador one-hot para equipamento do tipo caminhão. | `True` quando aplicável. |
| `Tipo_Escavadeira` | bool | Indicador one-hot para equipamento do tipo escavadeira. | `True` quando aplicável. |

---

## Colunas removidas da matriz modelável

Aparecem nos datasets para auditoria e rotulação, mas não devem ser usadas como entrada direta do modelo final.

| Variável | Motivo de remoção |
|---|---|
| `Id` | Identificador técnico; não generaliza. |
| `Inicio` | Timestamp bruto; substituído por features temporais. |
| `Fim` | Timestamp bruto e referência de rotulação e split. |
| `Tag` | Identificador de equipamento; substituído por encoding de frequência e usado para ranking operacional. |
| `Classe` | Categoria bruta; substituída por encoding e agregações. |
| `next_critical_event_time` | Informação futura, usada apenas para construir o alvo. |
| `tte_horas` | Informação futura, diretamente relacionada ao target. |
| `target_4h` | Variável resposta. |

---

## Splits temporais

Arquivo: `data/processed/features/splits/split_metadata.json`

| Variável | Descrição |
|---|---|
| `rows_total` | Total de linhas no dataset de features. |
| `rows_train` | Quantidade de linhas no treino. |
| `rows_val` | Quantidade de linhas na validação. |
| `rows_test` | Quantidade de linhas no teste. |
| `train_start`, `train_end` | Início e fim do período de treino. |
| `val_start`, `val_end` | Início e fim do período de validação. |
| `test_start`, `test_end` | Início e fim do período de teste. |

---

## Scores de modelo

Arquivo: `reports/model_selection_scores.parquet`

| Variável | Tipo | Descrição |
|---|---|---|
| `Id` | int64 | Ciclo avaliado. |
| `Tag` | object | Equipamento avaliado. |
| `Fim` | datetime64[UTC] | Data e hora do ciclo usado para avaliação temporal. |
| `target_4h` | int64 | Valor real do alvo. |
| `model_name` | object | Nome do modelo avaliado. |
| `split` | object | Partição: `train`, `val` ou `test`. |
| `score` | float64 | Probabilidade de risco produzida pelo modelo. |
| `threshold` | float64 | Corte usado para transformar score em predição binária. |
| `prediction` | int64 | Predição binária após aplicar o threshold. |
| `Classe` | object | Classe do ciclo para análise de erro e segmento. |
| `turno` | object | Turno para análise segmentada. |
| `Frota` | object | Frota para análise segmentada. |
| `Tipo` | object | Tipo de equipamento para análise segmentada. |

---

## Relatório de seleção de modelo

Arquivo: `reports/model_selection_report.csv`

| Variável | Descrição |
|---|---|
| `model_name` | Nome do candidato ou modelo. |
| `role` | Papel do modelo, por exemplo candidato oficial ou baseline diagnóstico. |
| `estimator` | Algoritmo ou estimador usado. |
| `fit_seconds` | Tempo de treinamento em segundos. |
| `threshold` | Threshold calibrado do modelo. |
| `eligible_for_selection` | Indica se o modelo podia vencer a seleção oficial. |
| `{train,val,test}_threshold` | Threshold aplicado em cada split. |
| `{train,val,test}_precision` | Precisão ciclo a ciclo: entre os positivos previstos, proporção correta. |
| `{train,val,test}_recall` | Recall ciclo a ciclo: entre positivos reais, proporção capturada. |
| `{train,val,test}_f1` | Média harmônica entre precision e recall. |
| `{train,val,test}_auc_pr` | Área sob a curva Precision-Recall. |
| `{train,val,test}_auc_roc` | Área sob a curva ROC. |
| `*_top15_precision_at_k` | Precisão operacional usando Top15 Tag-dia. |
| `*_top15_recall_at_k` | Recall operacional usando Top15 Tag-dia. |
| `*_top15_lift_vs_random` | Ganho do Top15 contra escolha aleatória. |
| `*_top15_alerts_per_day` | Quantidade média de alertas por dia no Top15. |
| `best_trial_number` | Número do melhor trial de tuning, quando aplicável. |
| `best_params` | Hiperparâmetros selecionados no tuning. |

---

## Métricas operacionais TopK

Arquivo: `reports/operational_daily_topk_metrics.csv`

| Variável | Descrição |
|---|---|
| `split` | Partição avaliada. |
| `top_k_tags_per_day` | Quantidade de equipamentos priorizados por dia. |
| `days` | Número de dias avaliados. |
| `selected_alerts` | Total de seleções feitas pelo TopK. |
| `alerts_per_day` | Média de equipamentos selecionados por dia. |
| `tag_day_prevalence` | Prevalência de Tag-dias positivos no universo avaliado. |
| `precision_at_k` | Proporção de selecionados que eram positivos. |
| `recall_at_k` | Proporção de positivos capturados pelo TopK. |
| `lift_vs_random` | Ganho em relação a uma seleção aleatória. |
| `positives_captured` | Total de positivos capturados pelo TopK. |
| `total_positives` | Total de positivos disponíveis no período. |

---

## Métricas por orçamento de alertas

Arquivo: `reports/operational_budget_metrics.csv`

| Variável | Descrição |
|---|---|
| `split` | Partição avaliada. |
| `budget_pct` | Percentual de ciclos priorizados. |
| `selected_rows` | Número de ciclos selecionados pelo orçamento. |
| `alerts_per_day` | Média diária de alertas gerados. |
| `precision_at_budget` | Precisão dentro do orçamento de alertas. |
| `recall_at_budget` | Recall dentro do orçamento de alertas. |
| `lift_vs_random` | Ganho contra seleção aleatória. |
| `positives_captured` | Positivos capturados no orçamento. |
| `total_positives` | Positivos reais disponíveis no split. |

---

## Métricas segmentadas

Arquivos: `segment_threshold_metrics.csv`, `segment_topk_tag_day_metrics.csv` e `segment_tag_hotspots.csv`

| Variável | Descrição |
|---|---|
| `segment_col` | Coluna usada para segmentar, como `Frota`, `Tipo`, `Classe` ou `turno`. |
| `segment_value` | Valor específico do segmento. |
| `rows` | Número de ciclos no segmento. |
| `prevalence` | Taxa de positivos reais no segmento. |
| `predicted_positive` | Quantidade de ciclos previstos como positivos. |
| `predicted_positive_rate` | Taxa de predições positivas. |
| `true_positive` | Verdadeiros positivos. |
| `false_positive` | Falsos positivos. |
| `false_negative` | Falsos negativos. |
| `precision` | Precisão no segmento. |
| `recall` | Recall no segmento. |
| `lift_vs_random` | Ganho contra seleção aleatória no segmento. |
| `top_k_tags_per_day` | Tamanho da lista diária usada no segmento. |
| `tag_days` | Número de combinações Tag-dia no segmento. |
| `selected_alerts` | Seleções realizadas no segmento. |
| `positives_captured` | Positivos capturados no segmento. |
| `total_positives` | Positivos totais do segmento. |
| `status` | Classificação de saúde e confiabilidade do segmento. |
| `recommendation` | Recomendação operacional para o segmento. |

---

## Hotspots por tag

Arquivo: `reports/segment_tag_hotspots.csv`

| Variável | Descrição |
|---|---|
| `Tag` | Equipamento avaliado. |
| `tag_days` | Quantidade de dias com presença da tag. |
| `avg_score` | Score médio da tag no período avaliado. |
| `max_score` | Maior score observado para a tag. |
| `positive_days` | Dias em que a tag teve evento positivo real. |
| `selected_days` | Dias em que a tag foi selecionada pelo TopK. |
| `selected_true_positive_days` | Dias selecionados corretamente como positivos. |
| `selected_false_positive_days` | Dias selecionados sem positivo real. |
| `missed_positive_days` | Dias positivos não capturados pelo TopK. |
| `selected_precision` | Precisão das seleções daquela tag. |
| `tag_positive_rate` | Taxa de dias positivos da tag. |

---

## Alertas deduplicados para operação

Arquivo: `reports/operational_deduplicated_alerts.csv`

| Variável | Descrição |
|---|---|
| `Id` | Ciclo selecionado. |
| `Tag` | Equipamento priorizado. |
| `Fim` | Data e hora do ciclo usado para priorização. |
| `target_4h` | Indica se houve evento crítico real em até 4h. |
| `split` | Partição avaliada. |
| `score` | Score de risco do modelo. |
| `threshold` | Corte técnico usado para predição binária. |
| `prediction` | Predição binária do ciclo. |

---

## Como ler as métricas principais

| Métrica | Interpretação |
|---|---|
| `precision` | Quando o modelo alerta, com que frequência ele acerta. |
| `recall` | Dos eventos positivos reais, quantos o modelo conseguiu capturar. |
| `f1` | Equilíbrio entre precision e recall. |
| `auc_pr` | Qualidade do ranking em problema desbalanceado; mais informativa que ROC quando positivos são raros. |
| `auc_roc` | Capacidade geral de separar positivos e negativos. |
| `precision_at_k` | Precisão dentro da lista curta de equipamentos priorizados por dia. |
| `recall_at_k` | Cobertura dos positivos reais pela lista TopK. |
| `lift_vs_random` | Quantas vezes o modelo supera uma escolha aleatória. |
| `threshold` | Corte técnico para transformar score em alerta binário. |
| `score` | Risco estimado pelo modelo; usado principalmente para ordenar tags. |

---

## Observações de uso

> O uso operacional recomendado é o ranking **Top15 Tag-dia**, não o acionamento automático para todo ciclo acima do threshold.

> `target_4h`, `tte_horas` e `next_critical_event_time` existem para rotulação e auditoria e **não devem ser usadas como entrada do modelo**. Identificadores brutos como `Id`, `Tag`, timestamps crus e dados de operador devem ser tratados com cuidado para evitar memorização ou exposição desnecessária.

> As features rolling sempre devem usar apenas histórico anterior ao ciclo avaliado. Toda avaliação final deve respeitar o split temporal para evitar contaminação entre passado e futuro.