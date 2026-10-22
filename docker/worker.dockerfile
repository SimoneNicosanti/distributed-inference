# syntax=docker/dockerfile:1.7

FROM python:3.14.8-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY services/worker/pyproject.toml services/worker/pyproject.toml
COPY packages/artifacts/pyproject.toml packages/artifacts/pyproject.toml
COPY packages/catalog/pyproject.toml packages/catalog/pyproject.toml
COPY packages/integration/pyproject.toml packages/integration/pyproject.toml
COPY packages/lifecycle/pyproject.toml packages/lifecycle/pyproject.toml
COPY packages/scheduling/pyproject.toml packages/scheduling/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project services/worker \
        --no-dev \
        --no-install-local \
        --no-install-project

COPY services/worker/src services/worker/src
COPY packages/artifacts/src packages/artifacts/src
COPY packages/catalog/src packages/catalog/src
COPY packages/integration/src packages/integration/src
COPY packages/lifecycle/src packages/lifecycle/src
COPY packages/scheduling/src packages/scheduling/src
COPY packages/shared/src packages/shared/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project services/worker \
        --locked \
        --no-dev \
        --no-editable

FROM python:3.14.8-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 worker \
    && useradd --system --uid 10001 --gid worker \
        --home-dir /nonexistent --shell /usr/sbin/nologin worker

COPY --from=builder /opt/venv /opt/venv

USER worker
WORKDIR /app

EXPOSE 9000 8815 8816 8817 8818

CMD ["python", "-m", "worker.main"]
