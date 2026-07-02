# syntax=docker/dockerfile:1.7

FROM python:3.13-slim

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_LINK_MODE=copy \
    PATH=/workspace/.venv/bin:/usr/local/bin:${PATH}

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY packages/aitraf-api/pyproject.toml ./packages/aitraf-api/pyproject.toml
COPY packages/aitraf-api/src ./packages/aitraf-api/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package aitraf-api

WORKDIR /workspace/packages/aitraf-api

EXPOSE 8000

CMD ["uvicorn", "aitraf_api.app:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
