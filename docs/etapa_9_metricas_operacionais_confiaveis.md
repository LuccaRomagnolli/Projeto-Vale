# Etapa 9 - Metricas operacionais confiaveis

Data: 02/05/2026

## Motivacao

As metricas globais ciclo-a-ciclo mostravam um modelo aparentemente
desequilibrado:

| Metrica | Leitura |
|---|---|
| Recall alto | O modelo captura muitos positivos |
| Precision baixa | Gera muitos falsos positivos ciclo-a-ciclo |
| AUC-PR baixa | Ranking global ainda tem separacao limitada |

Essa leitura e tecnicamente correta, mas incompleta para operacao. O dispatcher
nao precisa necessariamente ver todos os ciclos positivos; ele precisa de um
ranking confiavel de equipamentos com maior risco para agir dentro de uma
capacidade diaria de manutencao.

Por isso, a avaliacao foi alterada para responder perguntas operacionais:

1. Se eu puder olhar apenas os top K equipamentos por dia, quantos alertas reais capturo?
2. O modelo prioriza melhor do que escolha aleatoria?
3. Quantos falsos alertas viram volume operacional real depois de deduplicacao por Tag?
4. Qual orcamento de alertas entrega melhor equilibrio entre recall e precision?

## Artefatos implementados

| Artefato | Caminho |
|---|---|
| Avaliador operacional | `src/evaluation/evaluate_model.py` |
| Testes | `tests/test_operational_evaluation.py` |
| Relatorio JSON | `reports/operational_metrics_report.json` |
| Budget por percentual | `reports/operational_budget_metrics.csv` |
| Top K Tag-dia | `reports/operational_daily_topk_metrics.csv` |
| Alertas deduplicados | `reports/operational_deduplicated_alerts.csv` |

Comando:

```bash
make evaluate
```

## Modelo avaliado

| Campo | Valor |
|---|---|
| Modelo | `hist_gbdt_tuned` |
| Features | 48 |
| Threshold calibrado | 0.180223 |
| Split analisado | Teste temporal |

## Metricas no threshold

No teste, o threshold calibrado gera:

| Metrica | Valor |
|---|---:|
| Linhas no teste | 56687 |
| Prevalencia no teste | 0.1498 |
| Predicoes positivas | 25527 |
| Taxa alertada | 0.4503 |
| True positives | 6910 |
| False positives | 18617 |
| False negatives | 1580 |
| Precision | 0.2707 |
| Recall | 0.8139 |
| Lift vs aleatorio | 1.8074 |

Leitura:

1. A precision ciclo-a-ciclo ainda e baixa.
2. O recall e alto porque o threshold alerta 45% dos ciclos.
3. O lift mostra que o modelo e 1.8x melhor que selecao aleatoria, mas isso ainda nao e suficiente como unica metrica executiva.

## Metricas por orcamento de alerta

Essa tabela responde: "se a operacao so puder analisar uma fracao dos ciclos com
maior score, qual qualidade teremos?"

| Budget | Alertas/dia | Precision | Recall | Lift | Positivos capturados |
|---:|---:|---:|---:|---:|---:|
| 1% | 19.6 | 0.3933 | 0.0263 | 2.6260 | 223 |
| 2% | 39.1 | 0.3695 | 0.0494 | 2.4670 | 419 |
| 5% | 97.8 | 0.3284 | 0.1097 | 2.1927 | 931 |
| 10% | 195.5 | 0.2785 | 0.1860 | 1.8597 | 1579 |
| 20% | 391.0 | 0.2320 | 0.3098 | 1.5488 | 2630 |
| 30% | 586.4 | 0.2488 | 0.4984 | 1.6611 | 4231 |

Leitura:

1. O modelo e mais confiavel quando usado como priorizador, nao como classificador binario amplo.
2. Nos top 1% dos ciclos, a precision sobe para 39.3%, com lift 2.63.
3. O recall fica baixo nos budgets pequenos porque o volume analisado e pequeno.

## Metricas por Tag-dia

Essa e a visao mais proxima do uso real: em vez de alertar todo ciclo, agregamos
por equipamento e dia. Para cada Tag-dia usamos o maior score do dia e o target
indica se aquela Tag teve ao menos um risco real.

| Top K Tags/dia | Alertas/dia | Precision | Recall | Lift | Positivos capturados |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 0.6667 | 0.1453 | 2.0500 | 60 |
| 5 | 5 | 0.6333 | 0.2300 | 1.9475 | 95 |
| 10 | 10 | 0.6767 | 0.4915 | 2.0808 | 203 |
| 15 | 15 | 0.6689 | 0.7288 | 2.0569 | 301 |
| 20 | 20 | 0.6167 | 0.8959 | 1.8963 | 370 |

Essa e a metrica que eu recomendo levar para a empresa.

Leitura:

1. Top 10 Tags/dia captura 49.2% dos casos Tag-dia positivos com precision de 67.7%.
2. Top 15 Tags/dia captura 72.9% dos casos Tag-dia positivos com precision de 66.9%.
3. Top 20 Tags/dia captura 89.6% dos casos Tag-dia positivos com precision de 61.7%.
4. O lift fica perto de 2x, ou seja, a priorizacao e aproximadamente duas vezes melhor que escolha aleatoria.

## Deduplicacao por Tag em 4 horas

Para reduzir spam operacional, tambem foi simulada deduplicacao por Tag com
cooldown de 4 horas.

| Metrica | Valor |
|---|---:|
| Predicoes positivas brutas | 25527 |
| Alertas deduplicados | 2537 |
| Alertas deduplicados/dia | 87.5 |
| Precision deduplicada | 0.2590 |
| Captura de positivos ciclo-a-ciclo apos dedup | 0.0774 |

Leitura:

1. Deduplicar reduz muito volume bruto.
2. Mesmo assim, threshold global ainda gera alertas demais para rotina manual.
3. Para operacao, top K Tag-dia e mais confiavel do que threshold global.

## Recomendacao de metrica principal

Trocar a metrica principal do projeto para:

```text
Precision@TopK Tag-dia + Recall@TopK Tag-dia + Lift@TopK Tag-dia
```

K deve ser definido pela capacidade operacional diaria:

| Capacidade do time | Metrica sugerida |
|---|---|
| 3 equipamentos/dia | `Precision@Top3 Tag-dia` |
| 5 equipamentos/dia | `Precision@Top5 Tag-dia` |
| 10 equipamentos/dia | `Recall@Top10 Tag-dia` com precision minima |
| 15 equipamentos/dia | `Recall@Top15 Tag-dia` para cobertura forte |

Minha recomendacao atual:

```text
Usar Top 10 ou Top 15 Tags/dia como criterio executivo.
```

Justificativa:

| Opcao | Vantagem | Risco |
|---|---|---|
| Top 10 | Precision maior e volume tratavel | Captura apenas metade dos casos Tag-dia |
| Top 15 | Captura 72.9% dos casos Tag-dia | Exige maior capacidade operacional |
| Top 20 | Recall muito alto | Pode sobrecarregar manutencao |

## Decisao metodologica

AUC-PR e precision global continuam sendo metricas tecnicas auxiliares, mas nao
devem ser usadas como criterio unico de sucesso do projeto. Para o uso real, o
modelo deve ser avaliado como ranking operacional de equipamentos.

Nova metrica de sucesso recomendada:

```text
Capturar pelo menos 70% dos Tag-dia positivos analisando no maximo 15 Tags por dia,
com precision@15 acima de 60%.
```

Resultado atual:

| Criterio | Meta | Atual | Status |
|---|---:|---:|---|
| Recall@Top15 Tag-dia | >= 0.70 | 0.7288 | OK |
| Precision@Top15 Tag-dia | >= 0.60 | 0.6689 | OK |
| Lift@Top15 Tag-dia | >= 1.50 | 2.0569 | OK |

## Proxima melhoria

Depois dessa troca de criterio, a proxima etapa deve ser segmentar essas metricas
por:

1. `Frota`
2. `Tipo`
3. `Tag`
4. Turno
5. Semana de operacao

Isso vai mostrar onde o ranking e confiavel e onde ainda precisa de features ou
regras especificas.

## Status

Status: `CONCLUIDA`
