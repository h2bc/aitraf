# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_LINK_MODE=copy \
    PATH=/workspace/.venv/bin:/usr/local/bin:${PATH}

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -1sLf "https://dl.cloudsmith.io/public/task/task/setup.deb.sh" | bash \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends task \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY packages/aitraf-core ./packages/aitraf-core
COPY packages/aitraf-ml-core ./packages/aitraf-ml-core
COPY packages/aitraf-train ./packages/aitraf-train

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package aitraf-train

WORKDIR /workspace/packages/aitraf-train

CMD ["sleep", "infinity"]
