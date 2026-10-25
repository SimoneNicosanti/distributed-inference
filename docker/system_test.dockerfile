# syntax=docker/dockerfile:1.7

FROM python:3.14.8-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN uv venv /opt/venv

# External test-client dependencies and runtime dependencies of local packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        "aiofiles>=25.1.0" \
        "aiorwlock>=1.5.1" \
        "boto3>=1.43.116" \
        "httpx>=0.28.1" \
        "numpy>=2.0" \
        "onnx>=1.17" \
        "pyarrow>=25.0.0" \
        "pydantic>=2.8" \
        "pytest>=8.0" \
        "pytest-asyncio>=0.24" \
        "pytest-timeout>=2.3" \
        "redis>=8.0.1" \
        "tenacity>=9.1.2"

# Boundary packages used to build requests, artifacts, and catalog assertions.
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml
COPY packages/integration/pyproject.toml packages/integration/pyproject.toml
COPY packages/artifacts/pyproject.toml packages/artifacts/pyproject.toml
COPY packages/catalog/pyproject.toml packages/catalog/pyproject.toml

COPY packages/shared/src packages/shared/src
COPY packages/integration/src packages/integration/src
COPY packages/artifacts/src packages/artifacts/src
COPY packages/catalog/src packages/catalog/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
        ./packages/shared \
        ./packages/integration \
        ./packages/artifacts \
        ./packages/catalog

FROM python:3.14.8-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 system-test \
    && useradd --system --uid 10001 --gid system-test \
        --home-dir /nonexistent --shell /usr/sbin/nologin system-test \
    && mkdir /app \
    && chown system-test:system-test /app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=system-test:system-test pyproject.toml pyproject.toml
COPY --chown=system-test:system-test test test

USER system-test

# ENTRYPOINT ["pytest"]
# CMD ["-q", "test/system"]
