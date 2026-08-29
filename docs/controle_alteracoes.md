<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-Selecao-EF9F27?style=flat-square"/>
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
| Segmentos | Analise concentrada apenas no fim | Scores da selecao, avaliacao e segmentacao carregam `Frota`, `Tipo`, `turno` e `Classe` | Garante consistencia entre selecao, avaliacao e segmentacao. |
| Estabilidade | `std(test_recall)` e `std(test_precision)` ciclo-a-ciclo | `std(test_top15_recall_at_k)` e `std(test_top15_precision_at_k)` | Estabilidade deve ser medida na mesma metrica usada para promocao. |

## Impacto

| Item | Resultado |
|---|---|
| Candidatos oficiais | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna` |
| Baseline diagnostico | `logistic_regression_baseline` |
| Artefato operacional | `models/model_selected.joblib` |
| Threshold operacional | Calibrado na validacao e persistido no artefato |
| Test Precision@Top15 Tag-dia | `0.6756` |
| Test Recall@Top15 Tag-dia | `0.7361` |
| Test Lift@Top15 Tag-dia | `2.0774` |
| Gate de estabilidade | Aprovado (`recall_std=0.0107`, `precision_std=0.0254`) |

## Evidencias

| Evidencia | Caminho |
|---|---|
| Selecao robusta de modelos | `reports/model_selection/model_selection_report.json` |
| Trials por familia | `reports/model_selection/model_selection_trials.csv` |
| Scores da selecao com segmentos | `reports/model_selection/model_selection_scores.parquet` |
| Backtesting temporal | `reports/model_selection/model_selection_backtest_report.csv` |
| Avaliacao operacional | `reports/operational/operational_metrics_report.json` |
| Analise segmentada | `reports/segments/segment_operational_report.json` |
| Notebook executivo | `notebooks/main.ipynb` |

## Alteracao 02 - Padronizacao documental

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| Estrutura de `docs/` | Documentos deletados no working tree e formatacao desigual | Documentos recriados e padronizados com logo, cabecalho e secoes comuns | Facilitar leitura executiva, manutencao e auditoria. |
| Inferencia | Citada no README, sem etapa propria em `docs/` | Nova etapa `docs/etapa_11_inferencia_operacional.md` | Fechar a trilha ponta a ponta ate uso operacional. |
| Notebooks | Referencias historicas a notebooks `01..09` | Notebook oficial unico `notebooks/main.ipynb` | Alinhar documentacao a estrutura real do repositorio. |

## Alteracao 03 - Conformidade com Estudo Guiado e Boas Praticas

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| EDA | Sem analise de multicolinearidade numerica | Heatmap de correlacao Spearman inserido | Identificar features redundantes conforme exigencia do guia. |
| Modelagem | Apenas uma categoria (Classificacao Supervisionada) | Adicao de Deteccao de Anomalias (Isolation Forest) como trilha exploratoria | Cumprir exigencia de no minimo duas abordagens distintas de modelagem. |
| Interpretabilidade | Nao implementado programaticamente | Scripts para SHAP (summary_plot e waterfall_plot) adicionados | Explicar impacto local e global das features e cumprir requisito do guia. |
| Analise de Erros | Foco total em metricas TopK globais | Exportacao de matriz de confusao e `extreme_false_negatives.csv` agrupado | Analise sistematica de falhas para entender pontos cegos do modelo. |

## Alteracao 04 - Correcao de vazamento temporal e reparo das features de alerta

Data: 29/08/2026

### 4.1 Defeito de resolucao temporal nas features de alerta

As duas fontes de dados foram gravadas com resolucoes de datetime diferentes:

| Fonte | Coluna | Resolucao |
|---|---|---|
| `data/processed/labeled/apontamentos_labeled.parquet` | `Fim` | `datetime64[ns]` |
| `data/processed/labeled/critical_events.parquet` | `EVENT_TIME` | `datetime64[us]` |

O codigo derivava inteiros com `astype("int64")`, que devolve a resolucao interna
do pandas, e comparava as duas escalas com `searchsorted`. Como os valores diferem
por um fator de `1000`, todo ciclo parecia posterior a todos os eventos criticos.

| Consequencia | Efeito no dataset versionado |
|---|---|
| `n_alertas_4h`, `n_alertas_8h`, `n_alertas_24h` | Identicamente `0` em todas as linhas |
| `dias_desde_ultimo_alerta` | Media de `20159` dias, isto e, dias desde a epoca Unix |

Todo o bloco de historico de alertas estava inerte no treino do modelo promovido
anterior. A conversao passou a ser explicita em `src/utils/timeutils.py`, com
teste de regressao que verifica a fronteira real da janela em
`tests/test_feature_engineering.py`.

### 4.2 Vazamento nos encodings categoricos e no split

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| `Tag_freq`, `Operador_freq` | `value_counts` sobre o dataset inteiro antes do split | Frequencias aprendidas apenas no treino | Validacao e teste eram codificados com a propria massa de frequencia |
| `Classe_target_enc` | Media expansiva sobre a uniao dos splits | Ajustada no treino; treino mantem a versao causal que exclui o proprio rotulo | Impede que estatistica futura alcance o treino |
| Categorias de `Frota` e `Tipo` | `get_dummies` sobre o dataset inteiro | Categorias fixadas pelo treino | O conjunto de colunas era definido por categorias de validacao e teste |
| Fronteiras do split | Corte posicional, sem intervalo de guarda | Corte por calendario com embargo de `4h` | `target_4h` olha para frente: a cauda de cada bloco tinha rotulo definido pelo bloco seguinte |
| Folds do backtest | Sem embargo, encoder unico | Embargo por fold e encoder ajustado no treino de cada fold | Mesma contaminacao se repetia a cada fold |

### 4.3 Decomposicao medida do impacto

Ablacao controlada em `src/evaluation/leakage_ablation.py`, reproduzivel por
`make leakage-ablation`. Os dois bracos usam o mesmo dataset de features, o mesmo
estimador e os mesmos hiperparametros; o conjunto de teste e verificado como
identico, ja que o embargo corta apenas treino e validacao.

| Metrica | Com vazamento | Sem vazamento | Custo da correcao |
|---|---:|---:|---:|
| `test_top15_precision_at_k` | `0.8267` | `0.8178` | `-0.0089` |
| `test_top15_recall_at_k` | `0.9007` | `0.8910` | `-0.0097` |
| `test_top15_lift_vs_random` | `2.5421` | `2.5147` | `-0.0273` |
| `test_auc_pr` | `0.5718` | `0.5755` | `+0.0037` |

Leitura honesta do resultado: a variacao de `0.6756` para `0.8178` em
`precision@15` decorre majoritariamente do reparo das features de alerta, e nao
da correcao de vazamento. O otimismo atribuivel ao vazamento e de aproximadamente
`1` ponto percentual.

**Limitacao registrada:** apenas o termo de `-0.0089` e medicao controlada. A
comparacao com `0.6756` envolve outra familia de modelo, outros hiperparametros e
outras versoes de biblioteca. Alem disso, os hiperparametros da ablacao foram
selecionados sobre os dados sem vazamento, o que desfavorece levemente o braco
com vazamento: o otimismo real pode ser um pouco maior que o medido.

### 4.4 Impacto no artefato promovido

| Item | Antes | Depois |
|---|---|---|
| Modelo selecionado | `lightgbm_optuna` | `hist_gbdt_optuna` |
| `test_top15_precision_at_k` | `0.6756` | `0.8178` |
| `test_top15_recall_at_k` | `0.7361` | `0.8910` |
| `test_top15_lift_vs_random` | `2.0774` | `2.5147` |
| `test_auc_pr` | `0.2737` | `0.5755` |
| `recall_std` do gate | `0.0151` | `0.0066` |
| `precision_std` do gate | `0.0244` | `0.0189` |

Todos os pisos da politica de promocao permanecem atendidos.

## Alteracao 05 - Objetivo da busca, calibracao do threshold e rastreamento

Data: 29/08/2026

### 5.1 Objetivo do Optuna

| Campo | Antes | Depois | Justificativa |
|---|---|---|---|
| Objetivo da busca | Escalar `recall*1e6 + precision*1e3 + lift + auc*1e-3` | Metrica primaria da politica (`val_top15_recall_at_k`) | Diferencas de recall abaixo de `1e-3` eram engolidas pelo termo de precisao; `lift` nao e limitado e acima de `1000` inverteria a prioridade |
| Escolha do melhor trial | `study.best_trial`, que so enxerga o escalar | Ordenacao lexicografica completa da politica | Os criterios de desempate passam a valer tambem dentro da familia |
| Reinjecao de parametros no refit | Manual, para `scale_pos_weight`, `subsample_freq` e `class_weight` | `params_json` do trial, que ja guarda a configuracao completa | Elimina divergencia entre a configuracao avaliada e a treinada |

### 5.2 Atribuicao medida do impacto

Tres execucoes completas, isolando uma variavel de cada vez:

| Execucao | Stack | Objetivo | `precision@15` | `recall@15` | `lift@15` |
|---|---|---|---:|---:|---:|
| A | `pandas 3.0.5` | escalar | `0.817778` | `0.891041` | `2.514716` |
| B | `pandas 2.3.3` | escalar | `0.817778` | `0.891041` | `2.514716` |
| C | `pandas 2.3.3` | metrica primaria | `0.831111` | `0.905569` | `2.555717` |

`A == B` ate a sexta casa decimal: a mudanca de versao do pandas nao teve
**nenhum** efeito sobre as metricas, como esperado apos `src/utils/timeutils.py`
tornar o codigo independente da resolucao de datetime.

`B -> C` concentra toda a variacao (`+0.0133` em precisao). A causa nao e a regra
de selecao: comparando os dois criterios sobre os mesmos trials, ambos escolhem
o **mesmo** trial nas tres familias. A diferenca vem da trajetoria de busca --
o TPE usa o objetivo para guiar a exploracao, entao um objetivo diferente
explora trials diferentes.

### 5.3 Threshold degenerado

A calibracao devolvia silenciosamente o menor threshold quando o recall minimo
era inalcancavel. O caso ocorreu com o baseline heuristico, publicado com
threshold `0.0` e recall `1.0`.

O flag decisivo nao e "alvo inalcancavel": alertar sobre tudo **atinge** qualquer
alvo de recall. O que caracteriza a falha e a taxa de alerta. `ThresholdChoice`
passa a expor `alert_rate` e `degenerate`, e o gate de promocao bloqueia a
promocao nesse caso. O modelo vigente registra `alert_rate = 0.400`.

### 5.4 Demais itens

| Item | Estado |
|---|---|
| `subsample` do LightGBM | Era inerte sem `subsample_freq`; a dimensao era buscada em 30 trials sem efeito |
| Teste durante a busca | Deixou de ser pontuado nos trials: economiza cerca de um terco da avaliacao e remove o risco de vazamento por ordenacao |
| Calibracao de probabilidade | `src/models/calibration.py`, isotonica ajustada na validacao. Desligada por padrao: nao altera metricas TopK, mas altera a leitura do score e o threshold publicado |
| Rastreamento | MLflow com store sqlite em `mlruns/`. O backend de arquivos foi descontinuado no MLflow 3.x |
| Pruner do Optuna | **Nao implementado.** Exige valores intermediarios, e cada trial e um `fit` unico. Reportar por iteracao seria possivel em LightGBM e XGBoost, mas nao em HistGradientBoosting: as familias ficariam com orcamentos de busca desiguais, enviesando a comparacao |
| `pandas` | Fixado em `2.3.3`: o MLflow 3.x exige `pandas<3` |

### 5.5 Impacto no artefato promovido

| Item | Antes | Depois |
|---|---|---|
| `test_top15_precision_at_k` | `0.8178` | `0.8311` |
| `test_top15_recall_at_k` | `0.8910` | `0.9056` |
| `test_top15_lift_vs_random` | `2.5147` | `2.5557` |

Modelo selecionado permanece `hist_gbdt_optuna`. Todos os criterios da politica
seguem atendidos.

## Decisao

Status: `VIGENTE`. A metrica executiva oficial permanece `TopK Tag-dia`, com
AUC-PR, precision e recall ciclo-a-ciclo como diagnosticos auxiliares. O pipeline
oficial agora inclui interpretabilidade e modelos nao supervisionados como ferramentas de diagnostico de suporte.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
