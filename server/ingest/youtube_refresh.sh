#!/usr/bin/env bash
# Keep the "youtube" dataset (Ping's video_search) in step with the owner's liked-video
# transcripts: rebuild the docs, then sync the index (new videos inserted, re-fetched ones
# re-inserted, removed ones dropped). Run by the youtube-graph.timer (systemd/), or by hand.
#
# Entity extraction runs on arete:27b on the H100 Ollama behind the Pi tunnel (GPU 3), one video at a
# time, so Ping's qwen3.6 (h100:11434, GPUs 0-2) is not queued behind it. (:11434 cannot hold arete: its scheduler overcounts GPU 2 free memory and spills it to CPU). Embeddings: nomic-embed-text, the same
# model the backend queries with. Data stays on this desktop; the H100 stores nothing.
#
#   YT_TRANSCRIPTS   youtube-mcp's transcripts dir (default ~/repos/resnovae_chat/youtube/data/transcripts)
#   YT_INFER_HOST    Ollama for extraction + embeddings (default http://localhost:11435: the desktop tunnel -> Pi -> h100:4003, GPU 3)
#   YT_LLM           extraction model (default arete:27b)
set -euo pipefail
cd "$(dirname "$0")/.."

exec 9>/tmp/youtube-graph.lock
flock -n 9 || { echo "another youtube refresh is running"; exit 0; }

SRC=${YT_TRANSCRIPTS:-$HOME/repos/resnovae_chat/youtube/data/transcripts}
DS=datasets/incoming/youtube
PY=.venv/bin/python

export PROTOGRAPH_EMBED_HOST=${YT_INFER_HOST:-http://localhost:11435}
export PROTOGRAPH_LLM=${YT_LLM:-arete:27b}
export PROTOGRAPH_LLM_THINK=false
export PROTOGRAPH_NUM_CTX=${YT_NUM_CTX:-16384}

$PY ingest/youtube_to_docs.py "$SRC" "$DS/docs"
# one index, structure-aware chunks (whole [mm:ss] paragraphs, title on every chunk)
$PY ingest/build_index.py "$DS/docs" "$DS/storage" --sync --max-async 2 --chunker structured \
  --entity-guidance ingest/youtube_entities.txt --drop-entities ingest/youtube_drop_entities.txt
