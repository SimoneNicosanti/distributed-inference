codex_home="${CODEX_HOME:-/home/vscode/.codex}"

install -d -m 0755 "${codex_home}/hooks"

seed_file() {
    local source_path="$1"
    local destination_path="$2"
    local mode="${3:-0644}"

    if [[ ! -e "${destination_path}" ]]; then
        install -m "${mode}" "${source_path}" "${destination_path}"
    fi
}

seed_file \
    .devcontainer/codex/AGENTS.md \

    "${codex_home}/AGENTS.md"

seed_file \
    .devcontainer/codex/caveman-hook.json \
    "${codex_home}/hooks.json"

seed_file \
    .devcontainer/codex/caveman-enable.py \
    "${codex_home}/hooks/caveman-enable.py" \
    0755


## Caveman Installation for Codex
npx -y skills@latest add JuliusBrussee/caveman \
    --skill caveman \
    --agent codex \
    --global \
    --yes

## RTK Installation for Codex
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
rtk init -g --codex

## token-optimizer-mcp Installation for Codex
codex plugin remove token-optimizer@token-optimizer \
    >/dev/null 2>&1 || true
codex plugin marketplace remove token-optimizer \
    >/dev/null 2>&1 || true
codex plugin marketplace add ooples/token-optimizer-mcp
codex plugin add token-optimizer@token-optimizer
# Ensure its MCP server is registered
codex mcp remove token-optimizer >/dev/null 2>&1 || true
codex mcp add token-optimizer -- \
    npx -y @ooples/token-optimizer-mcp@latest
