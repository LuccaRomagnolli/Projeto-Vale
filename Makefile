PYTHON ?= python3

.PHONY: help install format lint test label eda dashboard features train model-selection benchmark tune-hist-gbdt gate-stability evaluate evaluate-segments infer smoke run-all clean

help:
	@echo "Targets:"
	@echo "  make install   - install dependencies"
	@echo "  make format    - format python files with black"
	@echo "  make lint      - run ruff checks"
	@echo "  make test      - run unit tests with coverage"
	@echo "  make label     - run data labeling pipeline"
	@echo "  make eda       - run EDA stage"
	@echo "  make dashboard - run Streamlit EDA dashboard"
	@echo "  make features  - run feature engineering stage"
	@echo "  make train     - train baseline and robust model selection"
	@echo "  make model-selection - tune and select among four supervised model families"
	@echo "  make benchmark - legacy alias for model-selection"
	@echo "  make tune-hist-gbdt - legacy alias for model-selection"
	@echo "  make gate-stability - validate temporal stability before promotion"
	@echo "  make evaluate  - run model evaluation"
	@echo "  make evaluate-segments - run segmented operational evaluation"
	@echo "  make infer     - run inference on test split with promoted artifact"
	@echo "  make smoke     - run quick validation (test + infer + evaluate + evaluate-segments)"
	@echo "  make run-all   - run full pipeline"
	@echo "  make clean     - remove generated local artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

format:
	$(PYTHON) -m black src tests

lint:
	$(PYTHON) -m ruff check src tests

test:
	COVERAGE_FILE=.pytest_cache/.coverage $(PYTHON) -m pytest tests -v --cov=src --cov-config=.coveragerc --cov-report=term-missing

label:
	$(PYTHON) -m src.data.make_dataset

eda:
	$(PYTHON) -m src.eda.run_eda

dashboard:
	streamlit run streamlit_app.py

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train_baseline
	$(PYTHON) -m src.models.model_selection

model-selection:
	$(PYTHON) -m src.models.model_selection

benchmark: model-selection

tune-hist-gbdt: model-selection

gate-stability:
	$(PYTHON) -m src.models.stability_gate

evaluate:
	$(PYTHON) -m src.evaluation.evaluate_model

evaluate-segments:
	$(PYTHON) -m src.evaluation.segment_analysis

infer:
	$(PYTHON) -m src.inference

smoke: test infer evaluate evaluate-segments

run-all: label eda features train gate-stability evaluate evaluate-segments

clean:
	rm -rf .pytest_cache .ruff_cache .coverage .coverage.*
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
