#!/usr/bin/env bash
# Eseguito una volta alla creazione del container.
# Le dipendenze sono già installate nell'immagine (vedi DevEnv.dockerfile),
# quindi qui resta solo l'installazione del progetto e i task per-utente.
set -euo pipefail

# IMPORTANTE: deve corrispondere esattamente al set usato nel Dockerfile.
# `uv sync` è dichiarativo e disinstalla ciò che non è nel set richiesto:
# se qui il set è più ristretto, uv rimuove quanto il build aveva preparato.
# Aggiungi --group export se lavori sulla conversione dei modelli.
UV_SYNC_ARGS=(--frozen --all-extras)

# ---------------------------------------------------------------------------
# 1. Permessi
# ---------------------------------------------------------------------------
# I volumi ereditano l'ownership dall'immagine (le dir sono create con
# chown nel Dockerfile), quindi di norma non c'è nulla da fare. Serve solo
# quando updateRemoteUserUID rimappa l'uid su host Linux con uid != 1000.
#
# Il chown è CONDIZIONALE: `chown -R` non condizionato su una cache uv a
# regime significa attraversare decine di migliaia di file a ogni creazione.
fix_owner() {
	local path="$1"
	[[ -e ${path} ]] || return 0
	if [[ "$(stat -c %u "${path}")" != "$(id -u)" ]]; then
		echo "==> Correzione ownership su ${path}"
		sudo chown -R "$(id -u):$(id -g)" "${path}"
	fi
}

fix_owner /home/vscode/.cache
fix_owner /home/vscode/.claude
fix_owner /home/vscode/.codex
fix_owner /opt/venv

# ---------------------------------------------------------------------------
# 2. Dipendenze
# ---------------------------------------------------------------------------
# `uv lock` è stato RIMOSSO da qui. Ri-risolveva le dipendenze a ogni
# creazione del container: due sviluppatori in due giorni diversi ottenevano
# versioni diverse, cioè l'opposto della riproducibilità che il commento
# originale dichiarava.
#
# `uv lock` va lanciato A MANO quando cambi le dipendenze, e uv.lock va
# committato in git. Qui usiamo --frozen, che legge il lock senza risolvere e
# fallisce se è disallineato dal pyproject.toml: così il disallineamento è un
# errore visibile e non un aggiornamento silenzioso.
echo "==> Sincronizzazione dipendenze"
uv sync "${UV_SYNC_ARGS[@]}"

# ---------------------------------------------------------------------------
# 3. Warm-up della cache mypy
# ---------------------------------------------------------------------------
# Popola MYPY_CACHE_DIR (sul volume) così il primo avvio del daemon usato
# dall'estensione VS Code parte da cache calda, invece di analizzare tutto il
# progetto al primo salvataggio.
echo "==> Warm-up cache mypy"
mypy src test || true

# ---------------------------------------------------------------------------
# 4. Hook pre-commit
# ---------------------------------------------------------------------------
# pre-commit era fra le dipendenze dev ma non veniva mai installato: il gate
# non esisteva. Scegli UNA opzione (due gestori di hook sullo stesso repo sono
# una fonte garantita di confusione):
#
#   A) pre-commit -> scommenta la riga sotto e lascia trunk-fmt-pre-commit
#      disabilitato in trunk.yaml
#   B) Trunk -> sposta trunk-fmt-pre-commit in `enabled` in trunk.yaml e
#      rimuovi pre-commit dal pyproject
#
# if [[ -f .pre-commit-config.yaml ]]; then
# 	pre-commit install --install-hooks
# fi


# ---------------------------------------------------------------------------
# 5. Codex Configuration
# ---------------------------------------------------------------------------
sh ./.devcontainer/codex-prepare.sh

# ---------------------------------------------------------------------------
# 6. Verifica
# ---------------------------------------------------------------------------
echo "==> Ambiente pronto"
echo "    venv:   ${VIRTUAL_ENV:-non impostato}"
python --version
mypy --version # deve riportare "compiled: yes"
ruff --version
uv --version
