#!/usr/bin/env bash
set -euo pipefail

## Changing persmissions to .claude cache for persistent volume
sudo chown -R vscode:vscode /home/vscode/.claude
## Changing permission to .cache for persistent volume
sudo chown -R vscode:vscode /home/vscode/.cache

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

## Install caveman plugin
claude plugin marketplace add JuliusBrussee/caveman
claude plugin install caveman@caveman
