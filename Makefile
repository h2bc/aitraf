PYTHON ?= uv run python
RUFF ?= uv run ruff
JUPYTER ?= uv run jupyter notebook

.PHONY: lint format data jupyter train-video-mae

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

data:
	$(PYTHON) scripts/data_pipeline.py

jupyter:
	$(JUPYTER)

train-video-mae:
	$(PYTHON) scripts/video_mae/train.py
