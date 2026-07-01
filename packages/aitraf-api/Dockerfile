# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS demo-clips

WORKDIR /workspace

COPY packages/aitraf-api/src/aitraf_api/features/demo_videos/filter.py ./demo_filter.py
COPY data/manifests/trick_classification/test.jsonl ./data/manifests/trick_classification/
COPY data/manifests/score_prediction_ordinal/test.jsonl ./data/manifests/score_prediction_ordinal/

RUN --mount=from=aitraf_clips,target=/local-clips,readonly python - <<'PY'
import json
import shutil
from pathlib import Path

from demo_filter import filter_demo_video_ids


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


classification_rows = read_jsonl(
    Path("data/manifests/trick_classification/test.jsonl")
)
aqa_rows = read_jsonl(Path("data/manifests/score_prediction_ordinal/test.jsonl"))
video_ids = filter_demo_video_ids(
    classification_rows=classification_rows,
    aqa_rows=aqa_rows,
)

clips_dir = Path("/local-clips")
output_dir = Path("/demo-clips")
output_dir.mkdir(parents=True, exist_ok=True)

for video_id in video_ids:
    source = clips_dir / video_id
    shutil.copy2(source, output_dir / video_id)
PY

FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_LINK_MODE=copy \
    PATH=/workspace/.venv/bin:/usr/local/bin:${PATH}

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY packages/aitraf-core/pyproject.toml ./packages/aitraf-core/pyproject.toml
COPY packages/aitraf-core/src ./packages/aitraf-core/src
COPY packages/aitraf-api/pyproject.toml ./packages/aitraf-api/pyproject.toml
COPY packages/aitraf-api/src ./packages/aitraf-api/src
COPY data ./data
COPY --from=demo-clips /demo-clips ./storage/data/clips

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package aitraf-api

WORKDIR /workspace/packages/aitraf-api

EXPOSE 8000

CMD ["uvicorn", "aitraf_api.app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
