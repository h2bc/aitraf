PYTHON ?= uv run python
RUFF ?= uv run ruff
JUPYTER ?= uv run jupyter notebook

.PHONY: lint format data jupyter train-video-mae

install:
	uv sync

	uv pip install --force-reinstall \
		--index-url https://download.pytorch.org/whl/cu126 \
		torch \
		torchvision \
		torchcodec

	conda install "ffmpeg<8"

	

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
