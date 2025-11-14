PYTHON ?= uv run python
RUFF ?= uv run ruff
HYDRA_ENTRY := scripts/data_pipeline.py

.PHONY: install lint format data
install:
	uv sync

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

data:
	$(PYTHON) $(HYDRA_ENTRY) $(ARGS)
