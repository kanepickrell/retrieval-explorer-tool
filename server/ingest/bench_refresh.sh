#!/usr/bin/env bash
# Refresh the bench dataset: pull new pi sessions + feedback from the Pi and the bench's
# Open WebUI chats from this desktop, rebuild
# the docs, and sync both indexes (changed docs re-inserted, removed docs dropped).
#
# Data stays on this machine. The H100 is used only for inference (embeddings +
# entity extraction), reached through the Pi's tunnel; it stores nothing.
#
#   BENCH_HOST   the Pi (default gopher.local)
#   BENCH_VAULT  local synced vault (default ~/BenchVault)
#   INFER_PORT   local port forwarded to the H100 Ollama (default 11435)
#
# Usage: ingest/bench_refresh.sh        (from server/, or anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${BENCH_HOST:-gopher.local}
VAULT=${BENCH_VAULT:-$HOME/BenchVault}
PORT=${INFER_PORT:-11435}
DS=datasets/incoming/bench
RAW=$DS/raw
PY=.venv/bin/python

export PROTOGRAPH_EMBED_HOST=http://localhost:$PORT
export PROTOGRAPH_LLM=${PROTOGRAPH_LLM:-qwen3.6:latest}
export PROTOGRAPH_LLM_THINK=${PROTOGRAPH_LLM_THINK:-false}

# inference tunnel: desktop:PORT -> Pi:11435 -> H100 Ollama
if ! curl -s -m 3 "$PROTOGRAPH_EMBED_HOST/api/tags" >/dev/null; then
  ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L "$PORT:localhost:11435" "$HOST"
  sleep 1
fi

mkdir -p "$RAW"
rsync -a --delete "$HOST:.pi/agent/sessions/" "$RAW/pi-sessions/"
rsync -a "$HOST:bench/feedback" "$HOST:bench/order_list.md" "$RAW/"

# the bench's Open WebUI chats (it moved there 2026-11-14): AI Bench, not archived
OWUI_URL=${OWUI_URL:-http://localhost:3080} $PY ingest/owui_to_docs.py "$RAW/owui-chats" \
  || echo "  (Open WebUI chats skipped: is it up, and are WEBUI_ADMIN_* in ~/repos/resnovae_chat/.env?)"

$PY ingest/bench_to_docs.py "$DS/docs" --vault "$VAULT" --sessions "$RAW/pi-sessions" --owui "$RAW/owui-chats" \
  --extra "$RAW/order_list.md" "$RAW/feedback/open_issues.md"

EXTRACT=(--entity-guidance ingest/bench_entities.txt --drop-entities ingest/bench_drop_entities.txt)
$PY ingest/build_index.py "$DS/docs" "$DS/storage" --sync --max-async 8 "${EXTRACT[@]}"
# the structure-aware comparison index (ProtoGraph's "bench · structured"), same docs and settings
$PY ingest/build_index.py "$DS/docs" "$DS/storage-structured" --sync --max-async 8 --chunker structured "${EXTRACT[@]}"
