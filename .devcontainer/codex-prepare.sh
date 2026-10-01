#!/usr/bin/env bash
set -euo pipefail

mkdir -p ~/.codex


# ---------------------------------------------------------------------------
# Codex baseline
# ---------------------------------------------------------------------------

cat > ~/.codex/config.toml <<'EOF'
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
personality = "pragmatic"

approval_policy = "on-request"
approvals_reviewer = "user"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = false

[features]
guardian_approval = false

[projects."/workspace/distributed-inference"]
trust_level = "trusted"
EOF

# All hooks are derived from provisioning.
rm -f ~/.codex/hooks.json


# ---------------------------------------------------------------------------
# token-optimizer-mcp (ooples)
#
# Role:
# - smart_read / smart_grep / smart_glob
# - Wiki / knowledge graph
# - MCP retrieval and storage
#
# Keep default MODE=assist: don't let it compete with RTK for shell calls.
# ---------------------------------------------------------------------------

codex mcp remove token-optimizer \
    >/dev/null 2>&1 || true

codex plugin remove token-optimizer@token-optimizer \
    >/dev/null 2>&1 || true

codex plugin marketplace remove token-optimizer \
    >/dev/null 2>&1 || true

codex plugin marketplace add ooples/token-optimizer-mcp
codex plugin add token-optimizer@token-optimizer

# The plugin should bundle the MCP server, but explicitly register it
# if Codex does not expose it.
if ! codex mcp get token-optimizer >/dev/null 2>&1; then
    codex mcp add token-optimizer -- \
        npx -y @ooples/token-optimizer-mcp@latest
fi


# ---------------------------------------------------------------------------
# token-optimizer (alexgreensh)
#
# Role:
# - session continuity
# - compaction recovery
# - quality scoring
# - loop / waste detection
# - context auditing
#
# Use balanced, NOT aggressive:
# RTK owns Bash PreToolUse compression.
# ---------------------------------------------------------------------------

codex plugin remove token-optimizer@alexgreensh-token-optimizer \
    >/dev/null 2>&1 || true

codex plugin marketplace remove alexgreensh-token-optimizer \
    >/dev/null 2>&1 || true

codex plugin marketplace add alexgreensh/token-optimizer

TOKEN_OPTIMIZER_PLUGIN_JSON="$(
    codex plugin add \
        token-optimizer@alexgreensh-token-optimizer \
        --json
)"

TOKEN_OPTIMIZER_ROOT="$(
    printf '%s' "$TOKEN_OPTIMIZER_PLUGIN_JSON" |
        python3 -c \
        'import json,sys; print(json.load(sys.stdin)["installedPath"])'
)"

TOKEN_OPTIMIZER_MEASURE="$TOKEN_OPTIMIZER_ROOT/skills/token-optimizer/scripts/measure.py"

TOKEN_OPTIMIZER_RUNTIME=codex \
    python3 "$TOKEN_OPTIMIZER_MEASURE" \
    codex-install --profile balanced


# ---------------------------------------------------------------------------
# RTK
#
# Role:
# - Bash / CLI compression
# - git, pytest, mypy, ruff, build output, logs, etc.
# ---------------------------------------------------------------------------

if command -v rtk >/dev/null 2>&1; then
    rtk init -g --codex --uninstall \
        >/dev/null 2>&1 || true
fi

curl -fsSL \
    https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh \
    | sh

export PATH="$HOME/.local/bin:$PATH"

rtk init -g --codex


# ---------------------------------------------------------------------------
# Caveman
#
# Role:
# - concise response style
# - no proxy/native integration
# ---------------------------------------------------------------------------

npx -y skills@latest add JuliusBrussee/caveman \
    --skill caveman \
    --agent codex \
    --global \
    --yes


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

echo "==> token-optimizer-mcp"
codex mcp get token-optimizer

echo "==> RTK"
rtk --version
rtk init --show --codex

echo "==> alexgreensh/token-optimizer"
TOKEN_OPTIMIZER_RUNTIME=codex \
    python3 "$TOKEN_OPTIMIZER_MEASURE" \
    codex-doctor --project "$PWD"

echo "==> Setup complete"
