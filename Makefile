PYTHON ?= uv run python
RUFF ?= uv run ruff
JUPYTER ?= uv run jupyter notebook

.PHONY: lint format data jupyter train-video-mae

install:
	uv sync
	uv pip install torch==2.7.0 torchvision==0.22.0 torchcodec==0.3.0 --index-url https://download.pytorch.org/whl/cu126
	sudo apt-get update
	sudo apt-get install -y ffmpeg

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
