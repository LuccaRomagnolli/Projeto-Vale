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
  <img src="https://img.shields.io/badge/Modelo-Seleção-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Divisão-70/15/15-888780?style=flat-square"/>
</p>

<p align="center">
  <sub>
    Autor: <strong>Lucca Maximus Romagnolli</strong> · Universidade de Brasília (UnB)
  </sub>
</p>

---

## Visão Geral

Este projeto transforma registros históricos de operação de equipamentos de mineração em uma lista diária de prioridade para manutenção preventiva. A pergunta operacional é objetiva:

> **Quais equipamentos devem ser verificados primeiro para reduzir o risco de um alerta crítico nas próximas 4 horas?**

Em vez de tratar o problema apenas como uma classificação individual de ciclos, o projeto avalia se o modelo apoia a decisão real da operação: ordenar as `Tags` por risco e destacar, a cada dia, os equipamentos que merecem atenção prioritária.

O resultado é um artefato operacional promovido em `models/model_selected.joblib`, selecionado por validação temporal e avaliado com métricas de priorização `Top15 Tag-dia`.

---

## O Que o Projeto Entrega

| Entrega | Finalidade |
|---|---|
| Base rotulada | Indica se cada ciclo antecede um alerta crítico em até 4 horas |
| Variáveis de comportamento | Resume histórico recente de ciclos, alertas, duração, frequência e contexto |
| Modelos concorrentes | Compara três famílias supervisionadas oficiais sob a mesma regra de seleção |
| Artefato promovido | Modelo final, colunas esperadas e `threshold` calibrado |
| Relatórios operacionais | Métricas `TopK`, avaliação segmentada e evidências de estabilidade |
| Inferência | Geração de `score` e `prediction` para novos dados |

---

## Dados e Escopo

| Item | Valor |
|---|---:|
| Registros do conjunto de dados | `377.907` |
| `Tags` únicas | `47` |
| Frotas únicas | `5` |
| Tipos de equipamento | `2` |
| Eventos críticos finais | `107.002` |
| Positivos em `target_4h` | `70.811` (`18,74%`) |
| Variáveis totais | `57` colunas |
| Variáveis modeláveis | `44` |

O alvo principal é `target_4h`: vale `1` quando há um evento crítico em até 4 horas após o ciclo observado, e `0` caso contrário. Essa janela foi adotada por ser curta o suficiente para apoiar ação operacional e longa o bastante para permitir antecipação.

---

## Metodologias Utilizadas

A solução combina metodologias de ciência de dados, aprendizado de máquina e avaliação operacional. O foco não é apenas prever eventos críticos, mas transformar as previsões em uma priorização diária utilizável pela manutenção.

| Metodologia | Aplicação no projeto |
|---|---|
| Formulação orientada ao negócio | Conversão do desafio em pergunta operacional: quais `Tags` priorizar antes de um alerta crítico |
| Rotulação supervisionada temporal | Criação do alvo `target_4h`, indicando ocorrência de evento crítico nas 4 horas seguintes |
| Análise exploratória de dados | Investigação de distribuição temporal, prevalência do alvo, concentração por frota, tipo, classe e `Tag` |
| Engenharia de variáveis temporais | Construção de atributos históricos disponíveis até o momento da decisão, sem uso de informação futura |
| Prevenção de vazamento de informação | Remoção de colunas futuras, identificadores brutos e variáveis que contaminariam o treinamento |
| Validação temporal | Separação `70 / 15 / 15` em treino, validação e teste com preservação da ordem cronológica |
| Seleção robusta multifamília | Comparação dos candidatos oficiais sob mesma base, mesma métrica primária e mesma regra de seleção |
| Otimização de hiperparâmetros | Uso de Optuna com `30` trials por candidato, sem escolha prévia de modelo por preferência |
| Calibração de decisão | `threshold` definido na validação, mantendo o teste reservado para reporte final |
| Avaliação operacional TopK | Medição de `precision@k`, `recall@k` e `lift@k` para listas diárias priorizadas |
| Avaliação segmentada | Desempenho por frota, tipo e `Tag`, separando riscos globais de recortes inconclusivos |
| Governança e reprodutibilidade | Artefatos versionados, política de promoção, barreira de estabilidade e comandos via `tasks.py` e `Makefile` |

---

## Metodologia

### 1. Ingestão e Contrato dos Dados

Dados carregados, padronizados e validados antes de qualquer modelagem. Garante consistência de nomes de colunas, tipos, datas e campos obrigatórios.

**Arquivo principal:** `src/data_loader.py`

### 2. Rotulação dos Alertas

Identifica quais ciclos antecedem um alerta crítico dentro da janela de 4 horas. Colunas que olham para o futuro são usadas apenas para criar o alvo e removidas da modelagem na sequência.

**Arquivo principal:** `src/alert_labeler.py`

### 3. Análise Exploratória Orientada à Decisão

Verifica distribuição temporal, prevalência do alvo, concentração por `Tag`, frota, tipo de equipamento e padrões com impacto operacional. Intenção: descobrir riscos de modelagem, desbalanceamento e segmentos que exigem cuidado — não apenas descrever a base.

**Arquivo principal:** `src/eda/run_eda.py`

### 4. Engenharia de Variáveis

Variáveis construídas com informação disponível até o momento do ciclo: contagens de ciclos recentes, duração média, variação de duração, alertas recentes, sazonalidade, frequência histórica por `Tag` e codificações controladas de categorias.

**Arquivo principal:** `src/features/build_features.py`

### 5. Divisão Temporal

| Parte | Uso |
|---|---|
| Treino (70%) | Ajustar os modelos |
| Validação (15%) | Calibrar `threshold`, ajustar hiperparâmetros e selecionar modelo |
| Teste (15%) | Reportar desempenho final, sem influenciar escolhas |

Essa estratégia evita que informações do futuro influenciem decisões do passado e aproxima a avaliação do uso real em produção.

### 6. Seleção Robusta de Modelos

Três candidatos oficiais competem com a mesma base, mesma divisão temporal, mesma métrica primária e otimização via Optuna:

| Papel | Modelos |
|---|---|
| Candidatos oficiais | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna` |
| Referência diagnóstica | `logistic_regression_baseline` |
| Trials por candidato | `30` |

O modelo é selecionado por desempenho operacional na validação, sem preferência prévia. A regra vigente prioriza o maior `val_top15_recall_at_k` e desempata por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr`. A regressão logística permanece apenas como baseline diagnóstico e não disputa a promoção.

**Arquivo principal:** `src/models/model_selection.py`

### 7. Avaliação Operacional

A avaliação principal usa `TopK Tag-dia`: para cada dia, o modelo ordena as `Tags` por risco e mede a qualidade dos `K` primeiros equipamentos indicados.

| Métrica | Interpretação |
|---|---|
| `precision@k` | Dentre os equipamentos priorizados, quantos realmente tiveram alerta |
| `recall@k` | Dos equipamentos com alerta, quantos apareceram na lista priorizada |
| `lift@k` | Quanto a priorização melhora em relação a uma escolha aleatória |

Métricas ciclo-a-ciclo (`recall`, `precision`, `AUC-PR`, `AUC-ROC`) são calculadas como diagnóstico técnico auxiliar.

**Arquivos principais:** `src/evaluation/evaluate_model.py`, `src/evaluation/segment_analysis.py`

### 8. Promoção e Estabilidade

Antes do uso operacional assistido, o modelo passa por regras de promoção e estabilidade: metas mínimas no teste, aprovação do `gate` e tratamento explícito de segmentos raros.

**Documentos oficiais:** `docs/politica_promocao_modelo.md`, `docs/controle_alteracoes.md`

---

## Estratégias Técnicas

### Prevenção de Vazamento de Informação

Colunas que revelam o futuro ou identificam diretamente eventos posteriores são removidas antes da modelagem:

| Coluna | Motivo da remoção |
|---|---|
| `next_critical_event_time` | Informação futura direta |
| `tte_horas` | Tempo até evento futuro |
| `target_4h` | Variável alvo |
| `Id`, `Inicio`, `Fim` | Identificadores e dados brutos |
| `Tag`, `Classe` | Substituídas por variáveis históricas ou codificadas |

### Priorização em Vez de Alarme Indiscriminado

O projeto evita depender apenas de um `threshold` global ciclo-a-ciclo — o que pode gerar volume alto de alertas. A estratégia principal é produzir uma lista ordenada por risco e limitar a decisão ao orçamento diário de atenção, como `Top15 Tags por dia`.

### Avaliação por Segmentos

O desempenho é analisado por recortes operacionais: frota, tipo e `Tag`. Segmentos com baixa prevalência são marcados como inconclusivos, não como falha global, evitando conclusões fortes em amostras pequenas.

### Governança de Promoção

A promoção depende de evidências versionadas em `reports/`, política explícita em `docs/` e comandos reproduzíveis via `tasks.py` ou `Makefile`.

---

## Estado Atual

| Parâmetro | Valor |
|---|---|
| Janela do alvo | `target_4h` |
| Divisão temporal | `70 / 15 / 15` com embargo de `4h` nas fronteiras |
| Pool oficial vigente | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna` |
| Modelo selecionado nesta rodada | `hist_gbdt_optuna` |
| Artefato operacional | `models/model_selected.joblib` |
| Relatório de seleção | `reports/model_selection/model_selection_report.json` |
| `threshold` promovido | `0.378744` |

### Métricas Operacionais no Teste — Top15 Tag-dia

| Métrica | Valor |
|---|---:|
| `precision@15` | **0.8133** |
| `recall@15` | **0.8862** |
| `lift@15` | **2.5010** |

No teste temporal, a lista diária de 15 equipamentos atingiu precisão de ~81,3%, recuperou ~88,6% dos casos críticos cobertos pelo critério operacional e foi ~2,50× melhor que uma priorização aleatória.

> **Revisão de 29/08/2026.** Estes números substituem os anteriores
> (`precision@15 = 0.6756`), que foram produzidos sobre um dataset de features
> defeituoso: as duas fontes de dados tinham resoluções de datetime distintas
> (`ns` e `us`), o que zerava `n_alertas_4h/8h/24h` e corrompia
> `dias_desde_ultimo_alerta`. A maior parte do ganho vem desse reparo. Na mesma
> revisão os encodings categóricos passaram a ser ajustados apenas no treino e o
> split ganhou embargo — correção que **custou** cerca de `1` ponto percentual de
> precisão, medido por ablação controlada. A decomposição está em
> `docs/controle_alteracoes.md` (alteração 04) e é reproduzível por
> `make leakage-ablation`.

---

## Viabilidade Atual

O projeto está metodologicamente apto para **piloto operacional assistido**, desde que usado com monitoramento e acompanhamento humano.

**Motivos:**

1. A validação respeita a ordem temporal dos dados.
2. A seleção usa métrica aderente à decisão real da operação.
3. O teste não decide hiperparâmetros nem escolha de modelo.
4. Há regra oficial de promoção e barreira de estabilidade (`gate`).
5. A avaliação segmentada separa riscos globais de recortes inconclusivos.

**Limites conhecidos:**

- O uso ciclo-a-ciclo por `threshold` global ainda pode gerar volume alto de alertas.
- Segmentos raros precisam de trilha dedicada antes de decisões automáticas.
- O desempenho deve ser monitorado continuamente contra mudança de padrão, degradação de métricas e aumento de falsos alertas.

### Limitações Honestas da Abordagem

| Limitação | Risco | Mitigação |
|---|---|---|
| Janela histórica limitada | A generalização pode mudar em outras regiões, meses ou sazonalidades. | Backtesting contínuo e retreino por janela móvel. |
| Dados de manutenção preventiva ausentes | O modelo não sabe se uma intervenção recente reduziu o risco de uma `Tag`. | Integrar ordens de serviço e histórico de manutenção. |
| Custo real de parada ausente no dataset | O impacto financeiro e a priorização econômica ainda ficam aproximados. | Adicionar custo/hora por frota e severidade operacional. |
| SHAP explica associação, não causalidade | Sinais correlacionais podem ser interpretados incorretamente como causa de falha. | Validar achados com especialistas e experimentos controlados no piloto. |

### Trabalhos Futuros Concretos

| Extensão | Proposta | Justificativa |
|---|---|---|
| Novos dados | Integrar manutenção preventiva, falhas mecânicas, clima e contexto de mina. | Aumenta explicação causal e reduz falsos positivos. |
| Novas features | Criar features em tempo real de sensores e comportamento anonimizado do operador. | Captura degradação mais próxima do evento. |
| Modelos online | Implementar aprendizado incremental e monitoramento de drift com gatilho de retreino. | Mantém performance quando a operação muda. |
| Integração operacional | Publicar o Top15 por API ou job diário no sistema do dispatcher. | Fecha o ciclo entre score, inspeção, feedback e governança. |

---

## Estrutura do Repositório

```text
.
├── data/
│   ├── raw/                  # dados brutos
│   ├── processed/            # dados rotulados e variáveis geradas
│   └── external/             # dados auxiliares
├── docs/                     # documentação por etapa, governança e referências
│   └── reference/            # materiais externos de apoio
├── models/                   # artefatos treinados
├── notebooks/                # notebook principal do projeto
├── pictures/                 # imagens usadas na documentação
├── reports/                  # ranking final na raiz e evidências em subpastas
│   ├── daily_priority_top15.csv
│   ├── baseline/
│   ├── eda/
│   ├── inference/
│   ├── model_selection/
│   ├── modeling_legacy/
│   ├── operational/
│   └── segments/
├── src/                      # código-fonte do fluxo
│   ├── data/                 # preparação de dados
│   ├── eda/                  # análise exploratória
│   ├── evaluation/           # avaliação operacional e segmentada
│   ├── features/             # construção de variáveis
│   ├── models/               # treino, seleção e estabilidade
│   ├── ui/                   # interface operacional de engenharia de minas
│   └── utils/                # utilitários
├── tests/                    # testes automatizados
├── tasks.py                  # runner de comandos multiplataforma
├── Makefile                  # atalhos opcionais para tasks.py
├── pyproject.toml            # configuração do projeto Python
└── requirements.txt          # dependências
```

A raiz do repositório fica reservada para arquivos de entrada do projeto
(`README.md`, `Makefile`, `tasks.py` e configurações). A interface operacional
fica em `src/ui/`. Materiais externos de apoio ficam em `docs/reference/`.

---

## Fluxo do Projeto

| Etapa | Multiplataforma | Make | Arquivo principal | Saída esperada |
|---:|---|---|---|---|
| 1 | `python tasks.py label` | `make label` | `src/data/make_dataset.py` | base rotulada |
| 2 | `python tasks.py eda` | `make eda` | `src/eda/run_eda.py` | relatório exploratório |
| 3 | `python tasks.py features` | `make features` | `src/features/build_features.py` | variáveis modeláveis |
| 4 | `python tasks.py train` | `make train` | `src/models/train_baseline.py` e `src/models/model_selection.py` | referência inicial e seleção |
| 5 | `python tasks.py gate-stability` | `make gate-stability` | `src/models/stability_gate.py` | validação de estabilidade |
| 6 | `python tasks.py evaluate` | `make evaluate` | `src/evaluation/evaluate_model.py` | métricas operacionais |
| 7 | `python tasks.py evaluate-segments` | `make evaluate-segments` | `src/evaluation/segment_analysis.py` | análise por segmento |
| 8 | `python tasks.py infer` | `make infer` | `src/inference.py` | pontuações de inferência |

**Fluxo completo:**

```bash
python tasks.py run-all
# ou
make run-all
```

**Validação rápida** (testes + inferência + avaliação operacional + avaliação segmentada):

```bash
python tasks.py smoke
# ou
make smoke
```

---

## Configuração do Ambiente

**Requisito:** Python `>=3.11, <3.14`

### Com Conda

```bash
conda create -n vale312 python=3.12 -y
conda activate vale312

python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

> **Windows (PowerShell):** execute `conda init powershell` e reinicie o terminal antes de `conda activate`.  
> **macOS / Linux:** use o hook adequado ao seu shell, por exemplo `eval "$(conda shell.bash hook)"`.

### Variáveis de Ambiente de Execução

| Variável | Efeito |
|---|---|
| `LOG_LEVEL` | Nível mínimo de log (`INFO` por padrão) |
| `LOG_FILE` | Caminho para também gravar o log em arquivo, com carimbo de tempo |
| `RANDOM_STATE` | Semente do projeto (`42` por padrão) |
| `MLFLOW_TRACKING` | `0` desliga o rastreamento de experimentos |

O terminal continua exibindo `[OK]` e `[ERROR]` como antes; o arquivo recebe
formato com carimbo de tempo, nível e módulo, adequado ao job de lote diário.

### Com venv

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

---

## Comandos Principais

Use `python tasks.py ...` em Windows, macOS ou Linux. Se o `make` estiver
instalado, os comandos `make ...` são atalhos equivalentes.

| Ação | Multiplataforma | Make |
|---|---|---|
| Instalar dependências | `python tasks.py install` | `make install` |
| Formatar código | `python tasks.py format` | `make format` |
| Verificar lint | `python tasks.py lint` | `make lint` |
| Verificar tipos | `python tasks.py typecheck` | `make typecheck` |
| Executar testes | `python tasks.py test` | `make test` |
| Gerar base rotulada | `python tasks.py label` | `make label` |
| Executar EDA | `python tasks.py eda` | `make eda` |
| Abrir interface operacional | `python tasks.py dashboard` | `make dashboard` |
| Construir features | `python tasks.py features` | `make features` |
| Treinar seleção robusta | `python tasks.py train` | `make train` |
| Selecionar modelo | `python tasks.py model-selection` | `make model-selection` |
| Validar estabilidade | `python tasks.py gate-stability` | `make gate-stability` |
| Avaliar operação | `python tasks.py evaluate` | `make evaluate` |
| Avaliar segmentos | `python tasks.py evaluate-segments` | `make evaluate-segments` |
| Medir custo do fix de vazamento | `python tasks.py leakage-ablation` | `make leakage-ablation` |
| Gerar inferência | `python tasks.py infer` | `make infer` |
| Processar lotes diários | `python tasks.py batch` | `make batch` |
| Congelar referência de drift | `python tasks.py monitor-baseline` | `make monitor-baseline` |
| Verificar drift | `python tasks.py monitor` | `make monitor` |
| Validação rápida | `python tasks.py smoke` | `make smoke` |
| Fluxo completo | `python tasks.py run-all` | `make run-all` |
| Limpar caches locais | `python tasks.py clean` | `make clean` |

O `Makefile` delega para `tasks.py`, mantendo o mesmo comportamento entre
sistemas operacionais quando `make` estiver disponível.

---

## Contrato de Inferência

`src/inference.py` implementa o contrato mínimo para uso operacional:

1. Carrega o artefato promovido em `models/model_selected.joblib`.
2. Valida a presença de `model`, `feature_columns` e `threshold`.
3. Valida os **valores** do lote com `src/inference_contract.py` (nulos em campos
   de identidade, datas implausíveis, faixas numéricas impossíveis).
4. Aplica o encoder categórico ajustado no treino, quando o lote traz colunas
   categóricas cruas.
5. Alinha as colunas com `feature_columns` e **falha** se alguma estiver ausente.
6. Gera `score` e `prediction`.
6. Salva a saída em `reports/inference/inference_scores.parquet`.
7. Consolida o ranking diário `Top15 Tag-dia` em `reports/daily_priority_top15.csv`.

**Entrada esperada:** arquivo `.csv` ou `.parquet` com as variáveis modeláveis.

**Saída esperada:**

| Coluna | Significado |
|---|---|
| `score` | probabilidade estimada de alerta crítico na janela |
| `prediction` | decisão binária após aplicação do `threshold` |

A interface em `src/ui/` consome esse ranking, os ciclos rotulados e os
eventos críticos para a engenharia de minas priorizar inspeções, tratar
alertas Don't Go e acompanhar o processamento da frota.

**Degradação controlada.** Cada artefato é carregado isoladamente: um arquivo
ausente ou corrompido degrada apenas o painel que depende dele, em vez de
impedir o servidor de subir. `GET /api/health` reporta o estado real de cada
fonte e lista as degradadas.

```bash
python tasks.py dashboard
# abre http://127.0.0.1:8000
```

**Ranking operacional acionável:** `reports/daily_priority_top15.csv`

| Coluna | Significado |
|---|---|
| `data` | dia operacional do ranking |
| `rank` | posição da `Tag` no dia |
| `Tag` | equipamento priorizado |
| `score` | maior score diário da `Tag` |
| `Frota`, `Tipo`, `turno` | contexto operacional |
| `motivo_principal` | sinal resumido para triagem |
| `risco_segmento` | faixa de prioridade |
| `acao_recomendada` | sugestão de ação assistida |

---

## Relatórios e Evidências

| Arquivo | Conteúdo |
|---|---|
| `reports/model_selection/model_selection_report.json` | modelo selecionado, regra de seleção e métricas |
| `reports/model_selection/model_selection_trials.csv` | tentativas de otimização com Optuna |
| `reports/model_selection/model_selection_backtest_report.csv` | avaliação temporal retrospectiva |
| `reports/model_selection/model_selected_threshold_curve.csv` | curva de calibração de `threshold` |
| `reports/operational/operational_metrics_report.json` | métricas operacionais consolidadas |
| `reports/segments/segment_operational_report.json` | desempenho por segmentos |
| `reports/inference/inference_scores.parquet` | pontuações geradas pela inferência |
| `reports/daily_priority_top15.csv` | lista diária acionável de `Tags` priorizadas |

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

**Documentos de governança:**

- `docs/politica_promocao_modelo.md`
- `docs/controle_alteracoes.md`
- `docs/benchmark_modelos_recomendacoes.md`
- `docs/revisao_geral_documentacao_2026-05-04.md`

---

## Caderno Técnico

**Caderno ativo:** `notebooks/main.ipynb` — visão técnico-executiva consolidada do projeto.

> Referências antigas a notebooks `01..09` foram removidas da documentação por não refletirem a estrutura atual versionada.

---

## Regras de Manutenção

Sempre que houver mudança em fluxo, comandos, métricas oficiais, artefato promovido ou política de promoção, atualizar no mesmo conjunto de alterações:

1. `README.md`
2. `docs/README.md`
3. Documento da etapa afetada em `docs/`
4. `docs/controle_alteracoes.md` (quando houver mudança metodológica)

> O `threshold` operacional deve vir do artefato promovido. Não deve ser substituído por valor fixo em variável de ambiente.

---

<p align="center">
  <sub>Vale · Operações de Mineração · Antecipação de Alertas Críticos</sub><br>
  <sub>Autor: <strong>Lucca Maximus Romagnolli</strong> · Universidade de Brasília (UnB)</sub>
</p>
