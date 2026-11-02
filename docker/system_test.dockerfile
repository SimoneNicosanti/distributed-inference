# syntax=docker/dockerfile:1.7

FROM python:3.14.8-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# External test-client dependencies and runtime dependencies of local packages.
# Boundary packages used to build requests, artifacts, and catalog assertions.
COPY test/system/pyproject.toml test/system/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml
COPY packages/integration/pyproject.toml packages/integration/pyproject.toml
COPY packages/artifacts/pyproject.toml packages/artifacts/pyproject.toml
COPY packages/catalog/pyproject.toml packages/catalog/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project test/system \
        --no-dev \
        --no-install-local \
        --no-install-project

COPY packages/shared/src packages/shared/src
COPY packages/integration/src packages/integration/src
COPY packages/artifacts/src packages/artifacts/src
COPY packages/catalog/src packages/catalog/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project test/system \
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
    && groupadd --system --gid 10001 system-test \
    && useradd --system --uid 10001 --gid system-test \
        --home-dir /nonexistent --shell /usr/sbin/nologin system-test \
    && mkdir /app \
    && chown system-test:system-test /app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=system-test:system-test pyproject.toml pyproject.toml
COPY --chown=system-test:system-test test test

USER system-test

CMD ["python", "-m", "pytest", "-q", "test/system"]
