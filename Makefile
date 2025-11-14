PYTHON ?= uv run python
HYDRA_ENTRY := scripts/data_pipeline.py

.PHONY: data
data:
	$(PYTHON) $(HYDRA_ENTRY) $(ARGS)
