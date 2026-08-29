PYTHON ?= python
TASKS := $(PYTHON) tasks.py

.PHONY: help install format lint typecheck test label eda dashboard features train train-baseline model-selection gate-stability gate-promotion evaluate evaluate-segments leakage-ablation infer batch monitor-baseline monitor simulate notebook smoke run-all clean
.SILENT:

help:
	$(TASKS) --list

install:
	$(TASKS) install

format:
	$(TASKS) format

lint:
	$(TASKS) lint

typecheck:
	$(TASKS) typecheck

test:
	$(TASKS) test

label:
	$(TASKS) label

eda:
	$(TASKS) eda

dashboard:
	$(TASKS) dashboard

features:
	$(TASKS) features

train:
	$(TASKS) train

train-baseline:
	$(TASKS) train-baseline

model-selection:
	$(TASKS) model-selection

gate-stability:
	$(TASKS) gate-stability

gate-promotion:
	$(TASKS) gate-promotion

evaluate:
	$(TASKS) evaluate

evaluate-segments:
	$(TASKS) evaluate-segments

leakage-ablation:
	$(TASKS) leakage-ablation

infer:
	$(TASKS) infer

batch:
	$(TASKS) batch

monitor-baseline:
	$(TASKS) monitor-baseline

monitor:
	$(TASKS) monitor

simulate:
	$(TASKS) simulate

notebook:
	$(TASKS) notebook

smoke:
	$(TASKS) smoke

run-all:
	$(TASKS) run-all

clean:
	$(TASKS) clean
