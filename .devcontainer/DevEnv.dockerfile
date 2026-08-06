# syntax=docker/dockerfile:1
# ^^^ DEVE essere la primissima riga: le parser directive di Docker vengono
# ignorate appena compare un qualsiasi altro commento. Serve per garantire un
# frontend che supporti uid=/gid= sui cache mount.
# trunk-ignore-all(checkov/CKV_DOCKER_2)
# Immagine di sviluppo per distributed-inference.
#
# NOTA SUL BUILD CONTEXT: questo Dockerfile richiede "context": ".." in
# devcontainer.json, perché copia pyproject.toml e uv.lock dalla radice del
# repo. Serve anche un .dockerignore alla radice, altrimenti il daemon riceve
# .git, .venv e i modelli.
#
# Pinnare per digest (`FROM ...:3.14-bookworm@sha256:...`) rende il build
# davvero riproducibile; il tag mobile no. Trade-off: aggiornamenti manuali.
FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

USER root

# ---------------------------------------------------------------------------
# 1. Pacchetti di sistema PRIMA di uv: questo layer cambia raramente, quindi
#    bumpare uv non deve invalidarlo (nella versione precedente accadeva).
# ---------------------------------------------------------------------------
# trunk-ignore(hadolint/DL3008)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        iproute2 \
        iputils-ping \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        ripgrep \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
# libglib2.0-0 aggiunto: libgl1 da solo spesso non basta a opencv (arriva con
#   ultralytics) e l'errore che ottieni è un ImportError poco leggibile.
# build-essential: verifica se ti serve ancora con
#   uv sync --verbose 2>&1 | grep -i building
# Se non compila nulla da sorgente, rimuovilo e l'immagine si alleggerisce
# di parecchie centinaia di MB.

# ---------------------------------------------------------------------------
# 2. uv dall'immagine ufficiale, versione pinnata.
#    Sostituisce `pip install --upgrade pip && pip install uv`: un layer in
#    meno, versione deterministica, e cadono i trunk-ignore DL3013/DL3042.
# ---------------------------------------------------------------------------
# TODO PINNA QUESTA VERSIONE. Verifica prima i tag disponibili:
#   docker pull ghcr.io/astral-sh/uv:latest && docker run --rm ghcr.io/astral-sh/uv:latest --version
# poi sostituisci :latest con la versione esatta (es. :0.9.2).
# Un tag mobile in un'immagine di sviluppo vanifica la riproducibilità.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ---------------------------------------------------------------------------
# 3. Ambiente uv
# ---------------------------------------------------------------------------
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=/usr/local/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
# UV_PROJECT_ENVIRONMENT: venv fuori dal bind mount (era in /workspace).
# UV_LINK_MODE=copy: cache (volume) e venv (overlayfs) sono su fs diversi,
#   gli hardlink fallirebbero con un warning a ogni sync.
# UV_COMPILE_BYTECODE: .pyc all'install -> startup più rapido di mypy/pytest.
# UV_PYTHON_DOWNLOADS=never: impedisce a uv di scaricare un SECONDO interprete
#   quando quello di sistema (3.14) è già adeguato.

# ---------------------------------------------------------------------------
# 4. Directory con ownership corretta PRIMA che i volumi vengano montati.
#    I volumi Docker ereditano contenuto e ownership dal path nell'immagine:
#    facendolo qui, i `chown -R` in post-create diventano superflui (erano
#    l'operazione più lenta dello script, su una cache uv con decine di
#    migliaia di file).
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/venv \
             /home/vscode/.cache/uv \
             /home/vscode/.cache/mypy \
             /home/vscode/.cache/ruff \
             /home/vscode/.claude \
             /home/vscode/.codex \
    && chown -R vscode:vscode /opt/venv /home/vscode/.cache /home/vscode/.claude /home/vscode/.codex

USER vscode
WORKDIR /workspace/distributed-inference

# ---------------------------------------------------------------------------
# 5. Dipendenze al BUILD TIME. È l'intervento che cambia i tempi di creazione
#    del container: prima ogni `postCreate` scaricava e installava tutto da
#    zero (dev + tutti i profili, con torch, pandas, kubernetes, stack onnx).
#
#    --no-install-project: installa solo le dipendenze, non il pacchetto. Il
#    layer si invalida quando cambiano uv.lock o pyproject.toml, NON a ogni
#    modifica del codice sorgente.
#
#    --frozen: usa uv.lock così com'è, senza ri-risolvere. Fallisce se il lock
#    è disallineato dal pyproject: è il comportamento che vuoi in un build.
#
#    ATTENZIONE: il set di extras/gruppi qui deve corrispondere ESATTAMENTE a
#    quello di post-create.sh. `uv sync` è dichiarativo e disinstalla ciò che
#    non è nel set richiesto: flag diversi = il post-create butta via il
#    lavoro fatto qui.
# ---------------------------------------------------------------------------
#    Il cache mount di BuildKit sopravvive all'invalidazione del layer: quando
#    aggiungi una dipendenza il RUN riparte, ma uv trova i wheel già scaricati
#    e paga solo la copia nel venv (nessun traffico di rete).
#    Attenzione: il cache mount è LOCALE AL BUILDER e non viaggia con
#    l'immagine, quindi cacheFrom non lo trasporta. Su un runner CI pulito o
#    dopo `docker builder prune` si torna al download completo.
#
#    I bind mount al posto di COPY evitano un layer e tengono pyproject.toml e
#    uv.lock fuori dall'immagine (li ha già il bind mount del workspace).
#    L'invalidazione resta legata al loro contenuto: se iteri spesso su
#    [tool.ruff]/[tool.mypy], valuta di spostarli in ruff.toml e mypy.ini per
#    non invalidare il layer delle dipendenze a ogni modifica di lint.
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --all-extras
# --all-extras include anche network-agent, che nella lista precedente mancava
# (pyroute2 non era installato, quindi mypy non type-checkava davvero i moduli
# che lo importano).
# Il gruppo `export` (torch, transformers, ultralytics, ...) NON è incluso:
# serve solo a tools/ ed experiments/, esclusi da mypy. Chi ne ha bisogno:
#   uv sync --frozen --all-extras --group export
