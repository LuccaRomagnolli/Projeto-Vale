PYTHON ?= python
TASKS := $(PYTHON) tasks.py

.PHONY: help install format lint test label eda dashboard features train train-baseline model-selection gate-stability evaluate evaluate-segments infer notebook smoke run-all clean
.SILENT:

help:
	$(TASKS) --list

install:
	$(TASKS) install

format:
	$(TASKS) format

lint:
	$(TASKS) lint

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

evaluate:
	$(TASKS) evaluate

evaluate-segments:
	$(TASKS) evaluate-segments

infer:
	$(TASKS) infer

notebook:
	$(TASKS) notebook

smoke:
	$(TASKS) smoke

run-all:
	$(TASKS) run-all

clean:
	$(TASKS) clean
