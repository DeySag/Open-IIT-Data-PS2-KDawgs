# Makefile for PS2 RPC
# Deterministic commands for data, train, eval, test

.PHONY: help install data-dev data-full train-baselines train-models eval smoke lint typecheck test clean

help:
	@echo "PS2 RPC - Available commands:"
	@echo "  install         - Install dependencies"
	@echo "  data-dev        - Generate dev dataset (~5k borrowers)"
	@echo "  data-full       - Generate full dataset (~100k borrowers)"
	@echo "  train-baselines - Train baseline models"
	@echo "  train-models    - Train all models"
	@echo "  eval            - Run evaluation against baselines"
	@echo "  smoke           - End-to-end smoke test (event in, decision out)"
	@echo "  lint            - Run ruff linting"
	@echo "  typecheck       - Run mypy type checking"
	@echo "  test            - Run pytest suite"
	@echo "  clean           - Remove generated data and artifacts"

install:
	pip install -e ".[dev]"

data-dev:
	python -m src.rpc.sim.generate --config configs/sim.yaml --scale dev --output data/dev.parquet

data-full:
	python -m src.rpc.sim.generate --config configs/sim.yaml --scale full --output data/full.parquet

train-baselines:
	python -m src.rpc.models.train_baselines --data data/dev.parquet --output models/baselines/

train-models:
	python -m src.rpc.models.train --data data/dev.parquet --config configs/sim.yaml --output models/v0/

eval:
	python -m src.rpc.eval.run --data data/dev.parquet --models models/v0/ --baselines models/baselines/ --output eval/

smoke:
	python -m src.rpc.serve.smoke_test

lint:
	ruff check src tests

typecheck:
	mypy src

test:
	pytest tests -v

clean:
	rm -rf data/*.parquet models/ eval/ __pycache__ .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true