# trunk-ignore-all(checkov/CKV_DOCKER_2)
## Using this image because it has an embedded non root user
FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

USER root

## Installing uv package system wide
# trunk-ignore(hadolint/DL3013)
# trunk-ignore(hadolint/DL3042)
RUN pip install --upgrade pip \
    && pip install uv

## Installing utilities system wide
# trunk-ignore(hadolint/DL3008)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        iproute2 \
        iputils-ping \
    && rm -rf /var/lib/apt/lists/*

USER vscode
