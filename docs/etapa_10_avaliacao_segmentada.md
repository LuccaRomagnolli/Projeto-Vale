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

# Etapa 10 - Avaliacao segmentada operacional

Data: 02/05/2026

## Objetivo

Entender onde o modelo `hist_gbdt_tuned` e confiavel e onde ele falha. A metrica
global TopK Tag-dia e util, mas ainda pode esconder problemas por frota, tipo de
equipamento, turno, classe operacional ou Tag especifica.

Esta etapa responde:

1. Em quais segmentos o modelo prioriza melhor?
2. Quais segmentos tem baixa cobertura?
3. Quais Tags positivas estao sendo perdidas no ranking Top15?
4. Onde devemos atacar com novas features, regras ou modelos especificos?

## Artefatos implementados

| Artefato | Caminho |
|---|---|
| Codigo | `src/evaluation/segment_analysis.py` |
| Testes | `tests/test_segment_analysis.py` |
| Relatorio JSON | `reports/segment_operational_report.json` |
| Threshold por segmento | `reports/segment_threshold_metrics.csv` |
| TopK Tag-dia por segmento | `reports/segment_topk_tag_day_metrics.csv` |
| Hotspots por Tag | `reports/segment_tag_hotspots.csv` |

Comando:

```bash
make evaluate-segments
```

## Segmentos avaliados

| Segmento | Origem |
|---|---|
| `Frota` | Reconstruida das colunas one-hot `Frota_*` |
| `Tipo` | Reconstruido das colunas one-hot `Tipo_*` |
| `turno` | Feature temporal criada na engenharia |
| `Classe` | Classe operacional do ciclo |
| `Tag` | Usada para hotspots e falsos negativos recorrentes |

## Resultado no threshold por segmento

### Frota

| Frota | Linhas | Prevalencia | Precision | Recall | Lift | Taxa alertada |
|---|---:|---:|---:|---:|---:|---:|
| `793-D 5S` | 21363 | 0.2136 | 0.2259 | 0.9025 | 1.0578 | 0.8532 |
| `793-D 4S` | 14082 | 0.1915 | 0.4282 | 0.7171 | 2.2356 | 0.3208 |
| `LeTourneau L 1850` | 10358 | 0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `793-D 2S` | 7388 | 0.0850 | 0.1816 | 0.4841 | 2.1364 | 0.2266 |
| `793-D 3S` | 3496 | 0.1696 | 0.4995 | 0.9342 | 2.9451 | 0.3172 |

Leitura:

1. `793-D 3S` tem melhor lift e recall no threshold.
2. `793-D 4S` tem boa precision relativa e lift alto.
3. `793-D 5S` tem recall alto, mas taxa alertada muito alta, sugerindo threshold permissivo demais para essa frota.
4. `LeTourneau L 1850` quase nao tem positivos no teste; nao deve ser julgado pela mesma regra global.

### Tipo

| Tipo | Linhas | Prevalencia | Precision | Recall | Lift | Taxa alertada |
|---|---:|---:|---:|---:|---:|---:|
| `Caminhao` | 46329 | 0.1831 | 0.2707 | 0.8148 | 1.4787 | 0.5510 |
| `Escavadeira` | 10358 | 0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Leitura:

1. O modelo atual e, na pratica, um modelo para caminhoes.
2. Escavadeira tem baixa prevalencia de positivos no teste e precisa de analise propria.

### Turno

| Turno | Linhas | Prevalencia | Precision | Recall | Lift | Taxa alertada |
|---|---:|---:|---:|---:|---:|---:|
| `tarde` | 19537 | 0.1435 | 0.2614 | 0.8377 | 1.8216 | 0.4599 |
| `manha` | 18858 | 0.1501 | 0.2745 | 0.8233 | 1.8294 | 0.4500 |
| `noite` | 18292 | 0.1561 | 0.2770 | 0.7812 | 1.7739 | 0.4404 |

Leitura:

1. Turnos sao relativamente equilibrados no threshold.
2. `noite` tem melhor precision, mas menor recall.
3. `tarde` tem maior recall, mas precision levemente menor.

### Classe

| Classe | Linhas | Prevalencia | Precision | Recall | Lift | Taxa alertada |
|---|---:|---:|---:|---:|---:|---:|
| `Operando` | 27396 | 0.1878 | 0.2747 | 0.8140 | 1.4628 | 0.5564 |
| `Parado` | 15387 | 0.1834 | 0.2775 | 0.8331 | 1.5129 | 0.5507 |
| `Hibernando` | 9066 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `Manutenção` | 4838 | 0.1083 | 0.2055 | 0.7099 | 1.8976 | 0.3741 |

Leitura:

1. `Operando` e `Parado` tem comportamento parecido.
2. `Manutenção` tem lift alto, mas precision baixa.
3. `Hibernando` nao tem positivos no teste e deve ser tratado como segmento especial.

## TopK Tag-dia por segmento

### Turno

| Turno | TopK | Precision | Recall | Lift | Positivos capturados |
|---|---:|---:|---:|---:|---:|
| `manha` | 15 | 0.4500 | 0.7975 | 2.2500 | 189 |
| `tarde` | 15 | 0.4299 | 0.7633 | 2.1529 | 187 |
| `noite` | 15 | 0.5200 | 0.7290 | 2.0525 | 234 |

Leitura:

1. `noite` tem maior precision no ranking Top15.
2. `manha` tem maior recall no ranking Top15.
3. Todos os turnos mantem lift acima de 2.0, o que e operacionalmente bom.

### Classe

| Classe | TopK | Precision | Recall | Lift | Positivos capturados |
|---|---:|---:|---:|---:|---:|
| `Operando` | 15 | 0.6578 | 0.7475 | 1.3172 | 296 |
| `Parado` | 15 | 0.6378 | 0.7674 | 1.3369 | 287 |
| `Manutenção` | 15 | 0.3303 | 0.9363 | 1.1867 | 147 |
| `Hibernando` | 15 | 0.0000 | 0.0000 | 0.0000 | 0 |

Leitura:

1. `Operando` e `Parado` sao os melhores segmentos para priorizacao.
2. `Manutenção` captura quase todos os positivos, mas com precision baixa.
3. `Hibernando` nao deve entrar no mesmo painel de risco se nao houver positivos historicos.

## Cuidados com TopK por Frota e Tipo

TopK por Frota/Tipo pode ficar artificial quando o segmento tem poucas Tags por
dia. Por exemplo, `Top15` pode selecionar praticamente todas as Tags de uma frota
em um dia, fazendo o recall chegar a 1.0 sem que isso represente priorizacao
real.

Recomendacao:

| Segmento | Metrica mais confiavel |
|---|---|
| `turno` | TopK Tag-dia |
| `Classe` | TopK Tag-dia |
| `Frota` | Threshold + hotspots de Tag |
| `Tipo` | Threshold + analise de prevalencia |
| `Tag` | Hotspots de falso negativo/falso positivo |

## Hotspots por Tag

Tags com mais dias positivos perdidos no Top15:

| Tag | Positivos perdidos | Falsos positivos selecionados | Dias positivos | Precision quando selecionada |
|---|---:|---:|---:|---:|
| `CA65921` | 25 | 0 | 25 | 0.000 |
| `CA5926` | 18 | 3 | 18 | 0.000 |
| `CA65789` | 14 | 0 | 14 | 0.000 |
| `CA65790` | 13 | 0 | 13 | 0.000 |
| `CA65915` | 10 | 0 | 10 | 0.000 |
| `CA65793` | 9 | 0 | 9 | 0.000 |
| `CA65916` | 9 | 0 | 9 | 0.000 |
| `CA65937` | 7 | 4 | 15 | 0.667 |
| `CA65909` | 2 | 3 | 18 | 0.842 |
| `CA65932` | 1 | 6 | 18 | 0.739 |

Leitura:

1. `CA65921`, `CA5926`, `CA65789`, `CA65790`, `CA65915`, `CA65793` e `CA65916` sao as principais perdas do ranking Top15.
2. Algumas Tags com muitos positivos nao entram no ranking, indicando que o modelo nao aprendeu bem seu padrao precursor.
3. Essas Tags devem ser investigadas antes de qualquer nova troca de algoritmo.

## Decisoes recomendadas

| Achado | Decisao |
|---|---|
| Caminhao concentra o sinal util | Manter modelo principal focado em caminhoes |
| Escavadeira quase nao tem positivos | Criar regra/avaliacao separada ou coletar mais historico |
| Turnos tem lift > 2 no Top15 | Ranking e operacionalmente confiavel por turno |
| `Manutenção` tem recall alto e precision baixa | Avaliar threshold especifico por Classe |
| Algumas Tags positivas sao sempre perdidas | Criar features/tag rules especificas para hotspots |

## Politica para baixa prevalencia

A avaliacao segmentada agora classifica segmentos em tres estados:

- `ok`: amostra e prevalencia suficientes para conclusao.
- `inconclusivo_baixa_amostra`: volume insuficiente para conclusao robusta.
- `inconclusivo_baixa_prevalencia`: positivos raros, risco de metrica instavel.

Segmentos inconclusivos saem da metrica primaria de aceite e entram em trilha
separada (coleta adicional, heuristica local ou threshold dedicado).

## Proxima etapa recomendada

Criar uma etapa de calibracao por segmento:

1. Threshold especifico por `Frota` e `Classe`.
2. Regras de boost de score para Tags hotspot.
3. Painel TopK com filtros por turno e classe.
4. Reavaliacao das metricas TopK apos calibracao.

## Status

Status: `CONCLUIDA`
