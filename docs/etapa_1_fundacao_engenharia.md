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
| Automacao de comandos | `tasks.py`, `Makefile` |
| Codigo fonte modular | `src/` |
| Testes automatizados | `tests/` |
| Documentacao raiz | `README.md` |

## Comandos principais

```bash
python tasks.py lint
python tasks.py test
python tasks.py run-all
python tasks.py smoke
```

## Resultado

| Item | Valor |
|---|---|
| Linguagem | Python `>=3.11, <3.14` |
| Pipeline completo | `python tasks.py run-all` |
| Validacao rapida | `python tasks.py smoke` |
| Estrutura de testes | `pytest` com cobertura |
| Qualidade estatica | `ruff` e `black` |

## Evidencias

- `tasks.py` contem tarefas multiplataforma para ingestao, EDA, features,
  treino, selecao, gate, avaliacao, segmentos, inferencia, notebook e testes.
- `Makefile` mantem atalhos opcionais que delegam para `tasks.py`.
- `tests/` cobre os modulos de dados, features, modelos, avaliacao e inferencia.
- `README.md` documenta setup, comandos e politica operacional.

## Decisao

Status: `CONCLUIDA`. A fundacao suporta execucao local, auditoria de artefatos e
evolucao incremental do pipeline.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
