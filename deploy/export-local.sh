#!/usr/bin/env bash
# Export the LOCAL Open WebUI + Agent Hub stack into retrieval-explorer's datasets,
# pseudonymised, in the same shape as the shared server's export_package.py.
#
#   bash deploy/export-local.sh [suffix]      (run in WSL, where docker lives)
#
# Writes server/datasets/incoming/agenthub-export-<date>-<suffix>/ (suffix default
# "local"), with uploads/ holding the original uploaded files. Then run the ingest
# steps on it (see server/ingest/README or the summary this prints).
#
# The pseudonym map (real user id -> user-NN) is the one file that undoes the
# pseudonymisation: it lives in ~/.agenthub-local/pseudonyms.json (WSL home, mode
# 600), never in the export or either repo, and is reused so pseudonyms stay stable.
set -euo pipefail

SUFFIX="${1:-local}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # retrieval-explorer
STACK="${STACK_DIR:-$HERE/../agenthub-chime-in}"
CONTAINER="${OWUI_CONTAINER:-open-webui}"
PRIVATE="$HOME/.agenthub-local"
WORK=/tmp/agenthub-export-local

HUB_PW="$(grep -E '^HUB_DB_PASSWORD=' "$STACK/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)"
HUB_PW="${HUB_PW:-hub}"

mkdir -p "$PRIVATE" && chmod 700 "$PRIVATE"
docker exec "$CONTAINER" rm -rf "$WORK"
docker exec "$CONTAINER" mkdir -p "$WORK/out"
docker cp "$HERE/server/ingest/export_local.py" "$CONTAINER:$WORK/export_local.py"
[ -f "$PRIVATE/pseudonyms.json" ] && docker cp "$PRIVATE/pseudonyms.json" "$CONTAINER:$WORK/pseudonyms.json"

cleanup() { docker exec "$CONTAINER" rm -rf "$WORK" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker exec \
  -e HUB_DSN="postgresql://hub:${HUB_PW}@hub-db:5432/hub" \
  -e PSEUDONYMS="$WORK/pseudonyms.json" \
  -e OUT="$WORK/out" \
  "$CONTAINER" python "$WORK/export_local.py"

docker cp "$CONTAINER:$WORK/pseudonyms.json" "$PRIVATE/pseudonyms.json"
chmod 600 "$PRIVATE/pseudonyms.json"

SRC="$(docker exec "$CONTAINER" sh -c "ls -d $WORK/out/agenthub-export-* 2>/dev/null | head -1")"
[ -n "$SRC" ] || { echo "export produced nothing" >&2; exit 1; }
NAME="$(basename "$SRC")-$SUFFIX"
DEST="$HERE/server/datasets/incoming/$NAME"
rm -rf "$DEST"
docker cp "$CONTAINER:$SRC" "$DEST"

echo
echo "export:      $DEST"
echo "pseudonyms:  $PRIVATE/pseudonyms.json (keep private)"
echo "dataset id:  $NAME"
