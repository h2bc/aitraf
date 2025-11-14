PYTHON ?= uv run python
RUFF ?= uv run ruff
JUPYTER ?= uv run jupyter notebook

.PHONY: lint format data jupyter

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

data:
	$(PYTHON) scripts/data_pipeline.py

jupyter:
	$(JUPYTER)
