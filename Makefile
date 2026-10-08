# Makefile for PS2 RPC
# Deterministic commands for data, train, eval, test

.PHONY: help install data data-check features train-baselines train-models train-addons eval eval-inputs smoke serve lint typecheck test clean

help:
	@echo "PS2 RPC - Available commands:"
	@echo "  install         - Install dependencies"
	@echo "  data            - Ingest all official extracts into data/event_store.duckdb"
	@echo "  data-check      - Verify official extracts are present (datasets/, gitignored)"
	@echo "  train-baselines - Train baseline models"
	@echo "  train-models    - Train all models"
	@echo "  eval            - Run evaluation against baselines"
	@echo "  smoke           - End-to-end smoke test (event in, decision out)"
	@echo "  serve           - Run the serving API (uvicorn, port 8000)"
	@echo "  lint            - Run ruff linting"
	@echo "  typecheck       - Run mypy type checking"
	@echo "  test            - Run pytest suite"
	@echo "  clean           - Remove generated data and artifacts"

install:
	pip install -e ".[dev]"

DATASETS ?= datasets
DB ?= data/event_store.duckdb

data:
	@test -n "$$CN_HASH_PEPPER" || echo "WARNING: CN_HASH_PEPPER unset - contact refs fall back to unpeppered sha256 (see docs/dataset_audit.md s11)"
	python -m src.rpc.ingest --datasets $(DATASETS) --db $(DB)

data-check:
	@test -f $(DATASETS)/dial_attempts.csv || (echo "missing datasets/dial_attempts.csv (gitignored official extracts)" && exit 1)
	@test -f $(DATASETS)/accounts.csv || (echo "missing $(DATASETS)/accounts.csv" && exit 1)
	@echo "official extracts present (see docs/dataset_audit.md)"

features:
	python -m src.rpc.features.build --store data/event_store.duckdb --accounts datasets/accounts.csv --as-of-range 2026-05-01 2026-06-29 7 --out data/features_official.parquet

train-baselines:
	python -m src.rpc.models.train_baselines --data data/dev.parquet --output models/baselines/

train-models:
	python -m src.rpc.models.train --data data/dev.parquet --config configs/state_tracker.yaml --output models/v0/

train-addons:
	python -m src.rpc.models.train_addons --datasets datasets --out artifacts --fit-cap 2026-05-26

eval:
	python -m src.rpc.eval.run --config configs/eval.yaml

eval-inputs:
	python -m src.rpc.eval.prepare --db data/event_store.duckdb --datasets datasets --out-dir data

smoke:
	python -m src.rpc.serve.smoke_test

serve:
	python -m uvicorn src.rpc.serve.app:app --host 0.0.0.0 --port 8000

lint:
	ruff check src tests

typecheck:
	mypy src

test:
	pytest tests -v

clean:
	rm -rf data/*.parquet models/ eval/ __pycache__ .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
