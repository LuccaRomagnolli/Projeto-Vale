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

# Etapa 1 - Fundacao de engenharia e qualidade

Data: 04/05/2026

## Objetivo

Criar uma base tecnica reprodutivel para desenvolver, testar e executar o
pipeline de antecipacao de alertas criticos com baixo risco operacional.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Configuracao de projeto Python | `pyproject.toml` |
| Dependencias | `requirements.txt`, `constraints.txt` |
| Automacao de comandos | `Makefile` |
| Codigo fonte modular | `src/` |
| Testes automatizados | `tests/` |
| Documentacao raiz | `README.md` |

## Comandos principais

```bash
make lint
make test
make run-all
make smoke
```

## Resultado

| Item | Valor |
|---|---|
| Linguagem | Python `>=3.11, <3.14` |
| Pipeline completo | `make run-all` |
| Validacao rapida | `make smoke` |
| Estrutura de testes | `pytest` com cobertura |
| Qualidade estatica | `ruff` e `black` |

## Evidencias

- `Makefile` contem targets para ingestao, EDA, features, treino, benchmark,
  tuning, gate, avaliacao, segmentos, inferencia e testes.
- `tests/` cobre os modulos de dados, features, modelos, avaliacao e inferencia.
- `README.md` documenta setup, comandos e politica operacional.

## Decisao

Status: `CONCLUIDA`. A fundacao suporta execucao local, auditoria de artefatos e
evolucao incremental do pipeline.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
