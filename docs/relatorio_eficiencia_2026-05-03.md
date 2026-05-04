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

# Relatorio de eficiencia e consistencia

Data: 03/05/2026

## Objetivo

Avaliar eficiencia ponta a ponta do projeto, verificando se os artefatos e
metricas sustentam uso em piloto operacional assistido.

## Escopo avaliado

| Comando | Finalidade | Status |
|---|---|---|
| `make run-all` | Executar pipeline completo | `APROVADO` |
| `make test` | Validar suite automatizada | `APROVADO` |
| `make infer` | Gerar scores operacionais | `APROVADO` |

## Resultados consolidados

| Area | Resultado |
|---|---|
| Dados | `377907` registros e `70811` positivos |
| Features | `57` colunas totais e `48` features modelaveis |
| Baseline | AUC-PR teste `0.1498` |
| Modelo principal | AUC-PR teste `0.2810` |
| Benchmark | Campeao `hist_gbdt_balanced` |
| Tuning | Promovido `hist_gbdt_tuned_04` |
| Operacional | `Precision@15=0.6800`, `Recall@15=0.7409`, `Lift@15=2.0910` |
| Estabilidade | Gate aprovado |
| Segmentos | Raros marcados como inconclusivos |

## Leitura de eficiencia

O projeto e mais eficiente como ferramenta de priorizacao diaria do que como
classificador binario ciclo-a-ciclo. A estrategia recomendada e ordenar Tags por
risco e acionar a capacidade operacional diaria, especialmente `Top15 Tag-dia`.

## Riscos remanescentes

1. Threshold global ciclo-a-ciclo gera alto volume bruto de alertas.
2. Segmentos raros como `Escavadeira` e `LeTourneau L 1850` exigem trilha
   separada.
3. Tags com falsos negativos recorrentes devem entrar em rotina de monitoramento.
4. Drift de prevalencia e mudanca operacional precisam de acompanhamento.

## Decisao

Status: `APROVADO PARA PILOTO ASSISTIDO`. A promocao deve manter governanca,
monitoramento e revisao periodica de segmentos.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
