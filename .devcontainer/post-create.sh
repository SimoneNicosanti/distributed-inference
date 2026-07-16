#!/usr/bin/env bash
set -euo pipefail

## Creating the uv.lock file
## This is used for reproducing the environment across multiple builds
uv lock

## Syncing the uv.lock file
## Installing all the dependencies declared in the pyproject.toml
## This automatically creates a .venv with all dependencies
uv sync \
	--group dev \
	--group runtime \
	--group control \
	--group measurement \
	--group model-management \
	--group gcp
