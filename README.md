<p align="center">
  <img src="pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração</h1>

<p align="center">
  Priorização diária de equipamentos com maior risco de alerta crítico <strong>"Don't Go"</strong><br>
  nas próximas 4 horas, com foco operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20–%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-Selecao-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Divisao-70/15/15-888780?style=flat-square"/>
</p>

---

## Visão Geral

Este projeto transforma registros históricos de operação de equipamentos de
mineração em uma lista diária de prioridade para manutenção preventiva. A
pergunta operacional é simples:

> **Quais equipamentos devem ser verificados primeiro para reduzir o risco de
> um alerta crítico nas próximas 4 horas?**

Em vez de tratar o problema apenas como uma classificação individual de ciclos,
o projeto avalia se o modelo ajuda na decisão real da operação: ordenar as
`Tags` por risco e destacar, a cada dia, os equipamentos que merecem atenção
primeiro.

O resultado atual é um artefato operacional promovido em
`models/model_selected.joblib`, selecionado por validação temporal e avaliado
com métricas de priorização `Top15 Tag-dia`.

---

## O Que o Projeto Entrega

| Entrega | Finalidade |
|---|---|
| Base rotulada | Indica se cada ciclo antecede um alerta crítico em até 4 horas |
| Variáveis de comportamento | Resume histórico recente de ciclos, alertas, duração, frequência e contexto |
| Modelos concorrentes | Compara quatro famílias supervisionadas sob a mesma regra de seleção |
| Artefato promovido | Modelo final, colunas esperadas e `threshold` calibrado |
| Relatórios operacionais | Métricas `TopK`, avaliação segmentada e evidências de estabilidade |
| Inferência | Geração de `score` e `prediction` para novos dados |

---

## Dados e Escopo

| Item | Valor atual |
|---|---:|
| Registros do conjunto de dados | `377907` |
| `Tags` únicas | `47` |
| Frotas únicas | `5` |
| Tipos de equipamento | `2` |
| Eventos críticos finais | `107002` |
| Positivos em `target_4h` | `70811` (`18.737679%`) |
| Variáveis totais | `57` colunas |
| Variáveis modeláveis | `48` |

O alvo principal é `target_4h`: ele vale `1` quando há um evento crítico em até
4 horas após o ciclo observado, e `0` caso contrário. Essa janela foi adotada
porque é curta o suficiente para apoiar uma ação operacional e longa o bastante
para permitir antecipação.

---

## Metodologia

### 1. Ingestão e contrato dos dados

Os dados são carregados, padronizados e validados antes de qualquer modelagem. O
objetivo desta etapa é garantir que nomes de colunas, tipos, datas e campos
obrigatórios estejam consistentes.

Arquivo principal: `src/data_loader.py`

### 2. Rotulação dos alertas

A rotulação identifica quais ciclos antecedem um alerta crítico dentro da janela
de 4 horas. Colunas que olham diretamente para o futuro são usadas apenas para
criar o alvo e depois são removidas da modelagem.

Arquivo principal: `src/alert_labeler.py`

### 3. Análise exploratória orientada à decisão

A análise exploratória verifica distribuição temporal, prevalência do alvo,
concentração por `Tag`, frota, tipo de equipamento e padrões que podem afetar a
operação. A intenção não é apenas descrever a base, mas descobrir riscos de
modelagem, desbalanceamento e segmentos que exigem cuidado.

Arquivo principal: `src/eda/run_eda.py`

### 4. Engenharia de variáveis

As variáveis são construídas com informação disponível até o momento do ciclo.
Entram, por exemplo, contagens de ciclos recentes, duração média, variação de
duração, alertas recentes, sazonalidade, frequência histórica por `Tag` e
codificações controladas de categorias.

Arquivo principal: `src/features/build_features.py`

### 5. Divisão temporal

O projeto usa divisão temporal `70 / 15 / 15`:

| Parte | Uso |
|---|---|
| Treino | Ajustar os modelos |
| Validação | Calibrar `threshold`, escolher hiperparâmetros e selecionar o modelo |
| Teste | Reportar desempenho final sem influenciar a escolha |

Essa estratégia evita que informações do futuro influenciem decisões do passado
e aproxima a avaliação do uso real.

### 6. Seleção robusta de modelos

Quatro candidatos oficiais competem com a mesma base, mesma divisão temporal,
mesma métrica primária e otimização com Optuna:

| Papel | Modelos |
|---|---|
| Candidatos oficiais | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna`, `extra_trees_optuna` |
| Referência diagnóstica | `logistic_regression_baseline` |

O modelo não é escolhido por preferência prévia. Ele é selecionado por
desempenho operacional na validação.

Arquivo principal: `src/models/model_selection.py`

### 7. Avaliação operacional

A avaliação principal usa `TopK Tag-dia`. Para cada dia, o modelo ordena as
`Tags` por risco e mede a qualidade dos `K` primeiros equipamentos indicados.

| Métrica | Interpretação |
|---|---|
| `precision@k` | Entre os equipamentos priorizados, quantos realmente tiveram alerta |
| `recall@k` | Dos equipamentos com alerta, quantos apareceram na lista priorizada |
| `lift@k` | Quanto a priorização melhora em relação a uma escolha aleatória |

As métricas ciclo-a-ciclo, como `recall`, `precision`, `AUC-PR` e `AUC-ROC`,
continuam sendo calculadas, mas são usadas como diagnóstico técnico auxiliar.

Arquivos principais:

- `src/evaluation/evaluate_model.py`
- `src/evaluation/segment_analysis.py`

### 8. Promoção e estabilidade

Antes de uso operacional assistido, o modelo precisa passar por regras de
promoção e estabilidade. A política oficial exige metas mínimas em teste,
aprovação da barreira de estabilidade (`gate`) e tratamento explícito de
segmentos raros.

Documentos oficiais:

- `docs/politica_promocao_modelo.md`
- `docs/controle_alteracoes.md`

---

## Estratégias Técnicas

### Prevenção de vazamento de informação

Colunas que revelam o futuro ou identificam diretamente eventos posteriores são
removidas antes da modelagem:

| Coluna | Motivo |
|---|---|
| `next_critical_event_time` | Informação futura direta |
| `tte_horas` | Tempo até evento futuro |
| `target_4h` | Variável alvo |
| `Id`, `Inicio`, `Fim` | Identificadores e datas brutas |
| `Tag`, `Classe` | Substituídas por variáveis históricas ou codificadas |

### Priorização em vez de alarme indiscriminado

O projeto evita depender apenas de um `threshold` global ciclo-a-ciclo, porque
isso pode gerar muitos alertas para a operação. A estratégia principal é
produzir uma lista ordenada por risco e limitar a decisão ao orçamento diário de
atenção, como `Top15 Tags por dia`.

### Validação temporal

Todas as decisões relevantes são tomadas na validação temporal. O teste fica
reservado para estimar como o modelo se comportaria em dados posteriores.

### Avaliação por segmentos

O desempenho é analisado por recortes operacionais, como frota, tipo e `Tag`.
Segmentos com baixa prevalência são marcados como inconclusivos, não como falha
global automática. Isso evita conclusões fortes em amostras pequenas.

### Governança de promoção

A promoção depende de evidências versionadas em `reports/`, política explícita
em `docs/` e comandos reproduzíveis via `Makefile`.

---

## Estado Atual

| Parâmetro | Valor |
|---|---|
| Janela do alvo | `target_4h` |
| Divisão temporal | `70 / 15 / 15` |
| Modelo selecionado | `lightgbm_optuna` |
| Artefato operacional | `models/model_selected.joblib` |
| Relatório de seleção | `reports/model_selection_report.json` |
| `threshold` promovido | Persistido no artefato selecionado |

### Métricas Operacionais no Teste

Resultado em `Top15 Tag-dia`:

| Métrica | Valor |
|---|---:|
| `precision@15` | **0.6756** |
| `recall@15` | **0.7361** |
| `lift@15` | **2.0774** |

Leitura prática: no teste temporal, a lista diária de 15 equipamentos teve
precisão de aproximadamente 67,6%, recuperou aproximadamente 73,6% dos casos
críticos cobertos pelo critério operacional e foi cerca de 2,08 vezes melhor do
que uma priorização aleatória.

---

## Viabilidade Atual

O projeto está metodologicamente apto para **piloto operacional assistido**,
desde que usado com monitoramento e acompanhamento humano.

Motivos:

1. A validação respeita a ordem temporal dos dados.
2. A seleção usa métrica aderente à decisão real da operação.
3. O teste não decide hiperparâmetros nem escolha de modelo.
4. Há regra oficial de promoção e barreira de estabilidade (`gate`).
5. A avaliação segmentada separa riscos globais de recortes inconclusivos.

Limites conhecidos:

- O uso ciclo-a-ciclo por `threshold` global ainda pode gerar volume alto de
  alertas.
- Segmentos raros precisam de trilha dedicada antes de decisões automáticas.
- O desempenho deve ser monitorado continuamente contra mudança de padrão dos
  dados, degradação de métricas e aumento de falsos alertas.

---

## Estrutura do Repositório

```text
.
├── data/
│   ├── raw/                  # dados brutos
│   ├── processed/            # dados rotulados e variáveis geradas
│   └── external/             # dados auxiliares
├── docs/                     # documentação por etapa e governança
├── models/                   # artefatos treinados
├── notebooks/                # notebook principal do projeto
├── pictures/                 # imagens usadas na documentação
├── reports/                  # relatórios, métricas e evidências
├── src/                      # código-fonte do fluxo
│   ├── data/                 # preparação de dados
│   ├── eda/                  # análise exploratória
│   ├── evaluation/           # avaliação operacional e segmentada
│   ├── features/             # construção de variáveis
│   ├── models/               # treino, seleção e estabilidade
│   └── utils/                # utilitários
├── tests/                    # testes automatizados
├── Makefile                  # comandos padronizados
├── pyproject.toml            # configuração do projeto Python
└── requirements.txt          # dependências
```

---

## Fluxo do Projeto

| Etapa | Comando | Arquivo principal | Saída esperada |
|---:|---|---|---|
| 1 | `make label` | `src/data/make_dataset.py` | base rotulada |
| 2 | `make eda` | `src/eda/run_eda.py` | relatório exploratório |
| 3 | `make features` | `src/features/build_features.py` | variáveis modeláveis |
| 4 | `make train` | `src/models/train_baseline.py` e `src/models/model_selection.py` | referência inicial e seleção |
| 5 | `make gate-stability` | `src/models/stability_gate.py` | validação de estabilidade |
| 6 | `make evaluate` | `src/evaluation/evaluate_model.py` | métricas operacionais |
| 7 | `make evaluate-segments` | `src/evaluation/segment_analysis.py` | análise por segmento |
| 8 | `make infer` | `src/inference.py` | pontuações de inferência |

Fluxo completo:

```bash
make run-all
```

Validação rápida:

```bash
make smoke
```

O comando `make smoke` executa testes, inferência, avaliação operacional e
avaliação segmentada.

---

## Configuração do Ambiente

Requisito: Python `>=3.11, <3.14`

### Ambiente isolado com Conda

```bash
conda create -n vale312 python=3.12 -y
conda activate vale312

python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

No Windows com PowerShell, execute `conda init powershell` e reinicie o
terminal antes de `conda activate`.

No macOS ou Linux, use o comando de ativação apropriado ao seu terminal, como:

```bash
eval "$(conda shell.bash hook)"
conda activate vale312
```

### Ambiente com venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, a ativação costuma ser:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Comandos Principais

```bash
make install            # instala dependências
make format             # formata arquivos Python com black
make lint               # verifica padrões com ruff
make test               # executa testes com cobertura
make label              # gera base rotulada
make eda                # executa análise exploratória
make features           # constrói variáveis
make train              # treina referência inicial e seleção robusta
make model-selection    # seleciona o melhor candidato oficial
make benchmark          # apelido legado de model-selection
make tune-hist-gbdt     # apelido legado de model-selection
make gate-stability     # valida estabilidade temporal
make evaluate           # avalia métricas operacionais
make evaluate-segments  # avalia segmentos operacionais
make infer              # gera inferência com artefato promovido
make smoke              # validação rápida
make run-all            # fluxo completo
make clean              # remove artefatos locais de execução
```

---

## Contrato de Inferência

`src/inference.py` implementa o contrato mínimo para uso operacional:

1. Carrega o artefato promovido em `models/model_selected.joblib`.
2. Valida a presença de `model`, `feature_columns` e `threshold`.
3. Alinha as colunas de entrada com `feature_columns`.
4. Preenche variáveis ausentes com `0.0`.
5. Gera `score` e `prediction`.
6. Salva a saída em `reports/inference_scores.parquet`.

Entrada esperada: arquivo `csv` ou `parquet` com as variáveis modeláveis.

Saída esperada:

| Coluna | Significado |
|---|---|
| `score` | probabilidade estimada de alerta crítico na janela |
| `prediction` | decisão binária após aplicação do `threshold` |

---

## Relatórios e Evidências

| Arquivo | Conteúdo |
|---|---|
| `reports/model_selection_report.json` | modelo selecionado, regra de seleção e métricas |
| `reports/model_selection_trials.csv` | tentativas de otimização com Optuna |
| `reports/model_selection_backtest_report.csv` | avaliação temporal retrospectiva |
| `reports/model_selected_threshold_curve.csv` | curva de calibração de `threshold` |
| `reports/operational_metrics_report.json` | métricas operacionais consolidadas |
| `reports/segment_operational_report.json` | desempenho por segmentos |
| `reports/inference_scores.parquet` | pontuações geradas pela inferência |

---

## Documentação por Etapa

| Etapa | Documento |
|---:|---|
| 1 | `docs/etapa_1_fundacao_engenharia.md` |
| 2 | `docs/etapa_2_ingestao_contrato_dados.md` |
| 3 | `docs/etapa_3_rotulacao_alert_labeler.md` |
| 4 | `docs/etapa_4_eda_orientada_decisao.md` |
| 5 | `docs/etapa_5_feature_engineering.md` |
| 6 | `docs/etapa_6_validacao_temporal_baseline.md` |
| 7 | `docs/etapa_7_selecao_robusta_modelos.md` |
| 8 | `docs/etapa_8_otimizacao_metricas_hist_gbdt.md` |
| 9 | `docs/etapa_9_metricas_operacionais_confiaveis.md` |
| 10 | `docs/etapa_10_avaliacao_segmentada.md` |
| 11 | `docs/etapa_11_inferencia_operacional.md` |

Documentos de governança:

- `docs/politica_promocao_modelo.md`
- `docs/controle_alteracoes.md`
- `docs/benchmark_modelos_recomendacoes.md`
- `docs/revisao_geral_documentacao_2026-05-04.md`

---

## Caderno Técnico

Caderno ativo:

- `notebooks/main.ipynb` — visão técnico-executiva consolidada do projeto.

Referências antigas a notebooks `01..09` foram removidas da documentação porque
não refletem mais a estrutura atual versionada.

---

## Regras de Manutenção

Sempre que houver mudança em fluxo, comandos, métricas oficiais, artefato
promovido ou política de promoção, atualizar no mesmo conjunto de alterações:

1. `README.md`
2. `docs/README.md`
3. documento da etapa afetada em `docs/`
4. `docs/controle_alteracoes.md`, quando houver mudança metodológica

O `threshold` operacional deve vir do artefato promovido. Ele não deve ser
substituído por valor fixo em variável de ambiente.

---

<p align="center">
  <sub>Vale · Operações de Mineração · Antecipação de Alertas Críticos</sub>
</p>
