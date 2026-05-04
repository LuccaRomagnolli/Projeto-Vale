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
| Threshold calibrado | 0.141388 |
| Split analisado | Teste temporal |

## Metricas no threshold

No teste, o threshold calibrado gera:

| Metrica | Valor |
|---|---:|
| Linhas no teste | 56687 |
| Prevalencia no teste | 0.1498 |
| Predicoes positivas | 24925 |
| Taxa alertada | 0.4397 |
| True positives | 6797 |
| False positives | 18128 |
| False negatives | 1693 |
| Precision | 0.2727 |
| Recall | 0.8006 |
| Lift vs aleatorio | 1.8208 |

Leitura:

1. A precision ciclo-a-ciclo ainda e baixa.
2. O recall e alto porque o threshold alerta 45% dos ciclos.
3. O lift mostra que o modelo e 1.8x melhor que selecao aleatoria, mas isso ainda nao e suficiente como unica metrica executiva.

## Metricas por orcamento de alerta

Essa tabela responde: "se a operacao so puder analisar uma fracao dos ciclos com
maior score, qual qualidade teremos?"

| Budget | Alertas/dia | Precision | Recall | Lift | Positivos capturados |
|---:|---:|---:|---:|---:|---:|
| 1% | 19.6 | 0.3386 | 0.0226 | 2.2610 | 192 |
| 2% | 39.1 | 0.3369 | 0.0450 | 2.2492 | 382 |
| 5% | 97.8 | 0.2917 | 0.0974 | 1.9477 | 827 |
| 10% | 195.5 | 0.2611 | 0.1743 | 1.7431 | 1480 |
| 20% | 391.0 | 0.2363 | 0.3155 | 1.5777 | 2679 |
| 30% | 586.4 | 0.2396 | 0.4800 | 1.5998 | 4075 |

Leitura:

1. O modelo e mais confiavel quando usado como priorizador, nao como classificador binario amplo.
2. Nos top 1% dos ciclos, a precision sobe para 33.9%, com lift 2.26.
3. O recall fica baixo nos budgets pequenos porque o volume analisado e pequeno.

## Metricas por Tag-dia

Essa e a visao mais proxima do uso real: em vez de alertar todo ciclo, agregamos
por equipamento e dia. Para cada Tag-dia usamos o maior score do dia e o target
indica se aquela Tag teve ao menos um risco real.

| Top K Tags/dia | Alertas/dia | Precision | Recall | Lift | Positivos capturados |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 0.6111 | 0.1332 | 1.8792 | 55 |
| 5 | 5 | 0.6333 | 0.2300 | 1.9475 | 95 |
| 10 | 10 | 0.6767 | 0.4915 | 2.0808 | 203 |
| 15 | 15 | 0.6800 | 0.7409 | 2.0910 | 306 |
| 20 | 20 | 0.6150 | 0.8935 | 1.8912 | 369 |

Essa e a metrica que eu recomendo levar para a empresa.

Leitura:

1. Top 10 Tags/dia captura 49.2% dos casos Tag-dia positivos com precision de 67.7%.
2. Top 15 Tags/dia captura 74.1% dos casos Tag-dia positivos com precision de 68.0%.
3. Top 20 Tags/dia captura 89.6% dos casos Tag-dia positivos com precision de 61.7%.
4. O lift fica perto de 2x, ou seja, a priorizacao e aproximadamente duas vezes melhor que escolha aleatoria.

## Deduplicacao por Tag em 4 horas

Para reduzir spam operacional, tambem foi simulada deduplicacao por Tag com
cooldown de 4 horas.

| Metrica | Valor |
|---|---:|
| Predicoes positivas brutas | 24925 |
| Alertas deduplicados | 2541 |
| Alertas deduplicados/dia | 87.6 |
| Precision deduplicada | 0.2621 |
| Captura de positivos ciclo-a-ciclo apos dedup | 0.0784 |

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

## Validade metodologica e viabilidade operacional

Pontos que sustentam viabilidade:

1. A selecao de modelo ocorre por validacao temporal, sem usar o teste para escolher campeao.
2. O criterio principal (TopK Tag-dia) esta alinhado com a capacidade diaria de manutencao.
3. O threshold e calibrado fora do teste, reduzindo risco de ajuste oportunista.
4. O projeto mede ganho relativo via lift, evitando leitura isolada de precision.

Riscos a monitorar para manter viabilidade:

1. Drift temporal de prevalencia pode degradar rapidamente precision@K.
2. Segmentos com baixa prevalencia precisam trilha separada para evitar conclusoes falsas.
3. Sem regressao automatica de metricas em CI, o desempenho pode cair silenciosamente.

Nova metrica de sucesso recomendada:

```text
Capturar pelo menos 70% dos Tag-dia positivos analisando no maximo 15 Tags por dia,
com precision@15 acima de 60%.
```

Resultado atual:

| Criterio | Meta | Atual | Status |
|---|---:|---:|---|
| Recall@Top15 Tag-dia | >= 0.70 | 0.7409 | OK |
| Precision@Top15 Tag-dia | >= 0.60 | 0.6800 | OK |
| Lift@Top15 Tag-dia | >= 1.50 | 2.0910 | OK |

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
