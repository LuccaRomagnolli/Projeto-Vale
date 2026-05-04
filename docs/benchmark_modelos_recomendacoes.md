<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Mining Fleet Alert Anticipation</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-HistGBDT-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

# Benchmark de modelos e recomendacoes para ganho de metricas

Data: 02/05/2026

## Objetivo

Avaliar se outros modelos supervisionados conseguem melhorar a antecipacao de
alerta Don't Go na janela de predicao de 4 horas, mantendo as mesmas garantias
metodologicas da implementacao atual:

| Principio | Como foi preservado |
|---|---|
| Sem data leakage | Mesmo split temporal da Etapa 6 |
| Mesmo target | `target_4h` corrigido com eventos criticos reais |
| Mesmas features | 32 features numericas/booleanas ja aprovadas |
| Calibracao honesta | Threshold escolhido apenas na validacao |
| Selecao honesta | Campeao escolhido por metrica de validacao, nao de teste |

## Scorecard oficial do projeto

O benchmark passa a usar um scorecard unico com prioridade operacional:

1. **Metricas primarias (criterio executivo):**
   - `Precision@TopK Tag-dia`
   - `Recall@TopK Tag-dia`
   - `Lift@TopK Tag-dia`
2. **Metricas secundarias (diagnostico tecnico):**
   - `AUC-PR`, `Recall`, `Precision`, `AUC-ROC` ciclo-a-ciclo

Metas de aceite para piloto (split de teste temporal):

| TopK | Precision@K minima | Recall@K minima | Lift@K minimo |
|---:|---:|---:|---:|
| 10 | 0.60 | 0.45 | 1.70 |
| 15 | 0.60 | 0.70 | 1.90 |
| 20 | 0.55 | 0.85 | 1.70 |

## Modelos treinados

O benchmark foi implementado em `src/models/benchmark_models.py` e executado com:

```bash
make benchmark
```

Modelos avaliados:

| Modelo | Familia | Motivacao |
|---|---|---|
| `logistic_regression_balanced` | Linear regularizado | Baseline supervisionado interpretavel |
| `hist_gbdt_balanced` | Gradient boosting | Modelo principal atual |
| `hist_gbdt_regularized` | Gradient boosting regularizado | Reduzir overfitting e melhorar generalizacao |
| `lightgbm_balanced` | Gradient boosting LightGBM | Modelo tabular forte com suporte a desbalanceamento |
| `extra_trees_balanced` | Ensemble de arvores aleatorias | Capturar nao linearidades com maior variancia controlada |
| `random_forest_balanced` | Bagging de arvores | Referencia robusta e estavel para tabular |

`lightgbm` e incluido automaticamente pelo codigo quando estiver instalado no
ambiente. O ambiente atual foi corrigido com `lightgbm==4.6.0`, portanto o
benchmark final rodou com 6 candidatos.

## Controle de alteracao metodologica

Registro formal:

- `docs/controle_alteracoes.md`

Resumo:

| Antes | Depois | Motivo |
|---|---|---|
| Campeao escolhido por `val_auc_pr` | Campeao escolhido por `val_top15_recall_at_k`, com desempate por `val_top15_precision_at_k`, `val_top15_lift_vs_random` e `val_auc_pr` | A operacao usa o modelo como ranking diario de equipamentos, nao como classificador isolado de ciclos. |

## Regra de escolha do campeao

O campeao foi escolhido por:

1. Maior `val_top15_recall_at_k`.
2. Desempate por `val_top15_precision_at_k`.
3. Segundo desempate por `val_top15_lift_vs_random`.
4. AUC-PR de validacao (`val_auc_pr`) apenas como diagnostico tecnico final.

Essa regra e adequada para o problema porque reproduz a forma de uso real:
priorizar equipamentos por `Tag-dia` dentro de uma capacidade diaria. AUC-PR,
recall e precision ciclo-a-ciclo continuam no relatorio, mas nao sao mais o
criterio principal de promocao. O teste fica reservado para estimar desempenho
final, sem orientar a escolha.

Decisao de promocao para producao: alem de vencer a validacao, o modelo precisa
passar no gate de estabilidade temporal (`make gate-stability`) e cumprir metas
TopK por segmento critico.

## Resultado do benchmark

| Modelo | Val P@15 | Val R@15 | Val Lift@15 | Test P@15 | Test R@15 | Test Lift@15 | Val AUC-PR tecnica |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hist_gbdt_balanced` | 0.6619 | 0.7493 | 2.1231 | 0.6822 | 0.7433 | 2.0979 | 0.2523 |
| `hist_gbdt_regularized` | 0.6548 | 0.7412 | 2.1002 | 0.6689 | 0.7288 | 2.0569 | 0.2890 |
| `lightgbm_balanced` | 0.6524 | 0.7385 | 2.0925 | 0.6756 | 0.7361 | 2.0774 | 0.2498 |
| `logistic_regression_balanced` | 0.6238 | 0.7062 | 2.0009 | 0.6733 | 0.7337 | 2.0705 | 0.2344 |
| `extra_trees_balanced` | 0.6238 | 0.7062 | 2.0009 | 0.6578 | 0.7167 | 2.0227 | 0.2232 |
| `random_forest_balanced` | 0.6214 | 0.7035 | 1.9933 | 0.6689 | 0.7288 | 2.0569 | 0.2390 |

Campeao por validacao:

```text
hist_gbdt_balanced
```

Artefatos gerados:

| Artefato | Caminho |
|---|---|
| Relatorio JSON | `reports/model_benchmark_report.json` |
| Relatorio CSV | `reports/model_benchmark_report.csv` |
| Scores por modelo | `reports/model_benchmark_scores.parquet` |
| Artefato campeao | `models/model_benchmark_winner.joblib` |

Observacao metodologica: os scores salvos pelo benchmark agora carregam tambem
`Frota`, `Tipo`, `turno` e `Classe`, permitindo a mesma leitura setorizada usada
na avaliacao operacional.

## Leitura dos resultados

O `hist_gbdt_balanced` foi o melhor por validacao operacional, com
`Recall@Top15 Tag-dia=0.7493`, `Precision@Top15 Tag-dia=0.6619` e lift acima de
2.1. No teste, manteve `Recall@Top15 Tag-dia=0.7433`, acima da meta operacional
de 0.70.

O `hist_gbdt_regularized` continua forte em AUC-PR tecnica, mas perdeu no
scorecard operacional de validacao. Por isso, nao deve ser promovido apenas por
ter melhor AUC-PR ciclo-a-ciclo.

O `lightgbm_balanced` nao venceu o benchmark nesta configuracao inicial, mas
entregou a maior precision de teste entre os candidatos principais e deve ser
otimizado com Optuna antes de ser descartado.

O `extra_trees_balanced` entregou recall alto, mas com AUC-PR menor. Isso sugere
que ele pode ser util em uma estrategia de alerta extremamente conservadora, mas
nao e o melhor rankeador geral.

## Recomendacoes para aumentar metricas

### 1. Otimizar LightGBM com Optuna

LightGBM ja esta instalado e foi incluido no benchmark. O proximo ganho deve vir
de busca de hiperparametros, nao apenas da troca do algoritmo.

Recomendacao:

```bash
pip install optuna
make train
make benchmark
```

Por que isso deve ajudar:

| Motivo | Impacto esperado |
|---|---|
| LightGBM e forte em tabular | Melhor AUC-PR e ranking de risco apos tuning |
| `scale_pos_weight` nativo | Melhor tratamento de desbalanceamento |
| Optuna | Busca sistematica de hiperparametros |
| SHAP integrado ao LightGBM | Interpretabilidade mais confiavel |

### 2. Backtesting temporal com multiplas janelas

Hoje existe um split temporal fixo 70/15/15. Ele e valido, mas ainda pode estar
dependente de uma unica safra temporal.

Melhoria recomendada:

| Janela | Treino | Validacao | Teste |
|---|---|---|---|
| Fold 1 | Meses 1-3 | Mes 4 | Mes 5 |
| Fold 2 | Meses 1-4 | Mes 5 | Mes 6 |
| Fold 3 | Meses 2-4 | Mes 5 | Mes 6 |

Beneficio: medir estabilidade do modelo e evitar escolher um algoritmo que foi
bom apenas por coincidencia temporal.

### 3. Otimizar threshold por custo operacional

O threshold atual busca recall minimo de 80%. Para operacao real, o ideal e
calibrar pelo custo de falso negativo e falso positivo.

Exemplo de funcao objetivo:

```text
custo_total = FN * custo_parada_nao_planejada + FP * custo_parada_preventiva
```

Se a mina tolerar mais alertas falsos para evitar perda critica, o threshold pode
ser reduzido. Se a fila de manutencao estiver saturada, o threshold pode subir
para priorizar apenas risco mais alto.

### 4. Melhorar features precursoras

As features atuais ja usam historico operacional e alertas recentes. O maior
ganho potencial agora esta em enriquecer sinais que antecedem a falha.

Prioridade de novas features:

| Feature | Por que pode melhorar |
|---|---|
| Contagem de eventos por `TIPO/EVENTO/SITUACAO` em 1h, 2h, 4h | Captura degradacao recente |
| Taxa de crescimento de alertas por Tag | Detecta aceleracao de risco |
| Sequencias de classes de atividade | Alguns padroes operacionais antecedem alerta |
| Tempo desde ultima manutencao ou parada | Contexto mecanico forte, se disponivel |
| Features por frota/modelo | Evita tratar equipamentos diferentes como iguais |
| Agregacoes por operador anonimo | Captura padroes de uso sem expor identidade |

### 5. Separar avaliacao por frota e tipo de equipamento

Uma metrica global pode esconder problemas. O modelo pode ir bem em caminhoes e
mal em carregadeiras, ou vice-versa.

Relatorios recomendados:

| Corte | Metrica |
|---|---|
| `Frota` | Recall, Precision, AUC-PR |
| `Tipo` | Recall, Precision, AUC-PR |
| `Tag` com maior volume | Recall operacional |
| Turno | Recall por periodo do dia |
| Regiao Sul/Sudeste | Estabilidade geografica |

### 6. Revisar a fonte de rotulacao continuamente

O target corrigido aumentou muito a utilidade do projeto porque saiu de uma
amostra pequena para eventos reais de telemetria. Ainda assim, a qualidade do
label deve ser tratada como ativo vivo.

Checks recomendados:

| Check | Objetivo |
|---|---|
| Eventos criticos sem Tag correspondente | Identificar perda de join |
| Tags com alerta demais | Detectar regra ruidosa ou equipamento anomalo |
| Mudancas no catalogo de alarmes | Evitar drift de regra de negocio |
| Distribuicao mensal do target | Detectar mudanca operacional |

## Recomendacao pratica de curto prazo

Para a proxima versao, eu promoveria o `hist_gbdt_balanced` como candidato
operacional porque ele foi escolhido pela validacao `Top15 Tag-dia` e manteve
recall operacional acima de 0.70 no teste temporal.

Antes de colocar como modelo oficial, eu faria:

1. Rodar tuning Optuna para LightGBM com validacao temporal.
2. Adicionar backtesting com pelo menos 3 janelas temporais.
3. Gerar matriz de custo por threshold para alinhar decisao com manutencao.
4. Validar metricas por `Frota`, `Tipo` e top Tags.

## Status

Status: `CONCLUIDO`
