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

# Revisao geral da documentacao

Data: 04/05/2026

## Objetivo

Registrar a padronizacao da documentacao do projeto, alinhando todos os
documentos de `docs/` ao frame visual do `README.md` e aos resultados oficiais
dos artefatos atuais.

## Escopo revisado

| Grupo | Documentos |
|---|---|
| Etapas | `etapa_1` a `etapa_11` |
| Governanca | `politica_promocao_modelo.md`, `controle_alteracoes.md` |
| Modelagem | `benchmark_modelos_recomendacoes.md` |
| Operacao | `relatorio_eficiencia_2026-05-03.md` |
| Notebook | `notebook_codigo_e_contexto_projeto.md` |
| Indice | `docs/README.md` |

## Ajustes aplicados

1. Inclusao do logo Vale e cabecalho padronizado em todos os Markdown de `docs/`.
2. Recriacao dos documentos deletados no working tree.
3. Inclusao de etapa final de inferencia operacional.
4. Correcao de referencias antigas a notebooks `01..09`.
5. Consolidacao dos resultados oficiais atuais de dados, features, modelo,
   threshold, TopK e estabilidade.

## Diagnostico metodologico

| Dimensao | Situacao |
|---|---|
| Split temporal | Adequado e documentado |
| Calibracao | Threshold definido na validacao |
| Metrica primaria | `Top15 Tag-dia`, aderente ao uso operacional |
| Benchmark | Comparacao consistente entre candidatos |
| Estabilidade | Gate temporal aprovado |
| Segmentos raros | Tratados como inconclusivos |

## Recomendacoes

1. Automatizar checagem documental para garantir logo e cabecalho padrao.
2. Manter regressao de `precision@15`, `recall@15` e `lift@15`.
3. Criar monitoramento diario de drift, prevalencia e volume de alertas.
4. Manter trilha dedicada para segmentos raros e Tags com falsos negativos.

## Decisao

Status: `CONCLUIDA`. A documentacao esta padronizada para leitura
tecnico-executiva e alinhada aos artefatos atuais do projeto.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
