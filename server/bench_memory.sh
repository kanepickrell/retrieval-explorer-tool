#!/usr/bin/env bash
# Run the bench memory service on this desktop.
#
#   this desktop   index + chunks + this backend (:8020). The data lives here.
#   H100           inference only (embeddings, LLM), via the Pi's tunnel:
#                  desktop:11435 -> Pi:11435 -> H100 Ollama
#   Pi             reaches the backend through a reverse tunnel:
#                  Pi:localhost:8010 -> desktop:8020   (bench hub's memory_search)
#
#   BENCH_HOST  the Pi (default gopher.local)
#   PORT        backend port (default 8020; 8000 is taken by another service here)
set -euo pipefail
cd "$(dirname "$0")"

HOST=${BENCH_HOST:-gopher.local}
export PORT=${PORT:-8020}
export PROTOGRAPH_EMBED_HOST=http://localhost:11435
export PROTOGRAPH_LLM=${PROTOGRAPH_LLM:-qwen3.6:latest}
export PROTOGRAPH_LLM_THINK=${PROTOGRAPH_LLM_THINK:-false}

FWD=(-R 8010:localhost:"$PORT")
# reuse an inference tunnel that's already up (e.g. one bench_refresh.sh opened)
curl -s -m 3 "$PROTOGRAPH_EMBED_HOST/api/tags" >/dev/null || FWD+=(-L 11435:localhost:11435)
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "${FWD[@]}" "$HOST" &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null' EXIT
sleep 1
kill -0 $TUNNEL 2>/dev/null || { echo "tunnel to $HOST failed (is 11435 or Pi:8010 already in use?)"; exit 1; }

exec .venv/bin/python app.py
