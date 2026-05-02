# Guia de Notebooks (Workflow Completo)

## Trilha executiva (entrada principal)

1. `09_manager_full_walkthrough.ipynb`
- Narrativa ponta a ponta para gestor.
- Leitura completa sem terminal.
- Gráficos e outputs já renderizados.

## Trilha técnica (suporte e auditoria)

## Ambiente recomendado antes dos notebooks

Para evitar erro de build em dependencias pesadas, use o mesmo padrao do projeto:

```bash
eval "$(/opt/miniconda3/bin/conda shell.zsh hook)"
conda activate base
conda create -n vale312 python=3.12 -y
conda activate vale312
python -m pip install -U pip wheel "setuptools<82"
echo "setuptools<82" > constraints.txt
python -m pip install -r requirements.txt -c constraints.txt --prefer-binary
```

Validação rápida:

```bash
make smoke
```

Sequência técnica recomendada:

1. `01_business_understanding.ipynb`
- Problema, objetivo, KPI oficial e critérios de aceite.

2. `02_data_understanding_eda.ipynb`
- Qualidade de dados, cobertura temporal e distribuição do target.

3. `03_data_preparation.ipynb`
- Preparação, risco de leakage e famílias de features.

4. `04_modeling_benchmark.ipynb`
- Validação temporal, comparação de modelos e decisão de campeão.

5. `05_operational_evaluation_application.ipynb`
- Aplicação prática com TopK Tag-dia e orçamento operacional.

6. `06_segment_analysis_and_risk.ipynb`
- Segmentação, baixa prevalência, riscos e hotspots de Tag.

7. `07_model_governance_and_promotion.ipynb`
- Gate de estabilidade, política de promoção e rastreabilidade.

8. `08_executive_readout_for_head_of_tech.ipynb`
- Resumo final para defesa com Head de Tecnologia.

## Notebooks legados
- `00_extracao_head_dados.ipynb`: exploração inicial de extração.
- `01_eda.ipynb`: notebook exploratório anterior.
