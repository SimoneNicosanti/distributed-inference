# syntax=docker/dockerfile:1.7

FROM python:3.14.8-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy manifests first to preserve the dependency layer when source changes.
COPY services/front_end/pyproject.toml services/front_end/pyproject.toml
COPY packages/artifacts/pyproject.toml packages/artifacts/pyproject.toml
COPY packages/integration/pyproject.toml packages/integration/pyproject.toml
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project services/front_end \
        --no-dev \
        --no-install-local \
        --no-install-project

COPY services/front_end/src services/front_end/src
COPY packages/artifacts/src packages/artifacts/src
COPY packages/integration/src packages/integration/src
COPY packages/shared/src packages/shared/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --project services/front_end \
        --locked \
        --no-dev \
        --no-editable

FROM python:3.14.8-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 front-end \
    && useradd --system --uid 10001 --gid front-end \
        --home-dir /nonexistent --shell /usr/sbin/nologin front-end

COPY --from=builder /opt/venv /opt/venv

USER front-end
WORKDIR /app

EXPOSE 8000

CMD ["python", "-m", "front_end.main"]
