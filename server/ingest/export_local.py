"""Export a LOCAL Open WebUI + Agent Hub stack as a pseudonymised Agent Hub export,
in the same shape as export_package.py on the shared server, so the rest of the
ingest pipeline (ingest/run_ingest.py: export_to_docs -> tag_docs -> build_passages -> build_metadata_graph -> eval sets)
runs on it unchanged.

Differences from export_package.py:
  * reads the LIVE webui.db through SQLite's backup API (a consistent copy taken
    at export time), not a snapshot someone made earlier
  * request texts (user-tasks-*) come from pgvector, the slim build's vector store,
    instead of Chroma
  * copies each upload's ORIGINAL file into uploads/<file_id>_<name>, so Ping's
    panel can open the real PDF (export_package.py drops file bytes)
  * skips Open WebUI's built-in User Guide files: app documentation, not user data

Meant to run INSIDE the open-webui container (it has python, sqlite3, psycopg, the
data folder and the compose network); deploy/export-local.sh does that. Read-only
on every source.

Pseudonyms: user-NN by account creation date, "system" for the hub seed owner,
kept in PSEUDONYMS (outside the export) so later exports reuse them. Name and
email fields are dropped; ids, emails, usernames and full names inside other
values are replaced. First names left in free text are counted in the manifest.

Env:
  OWUI_DB       /app/backend/data/webui.db
  OWUI_DATA     /app/backend/data              (upload paths are resolved against it)
  HUB_DSN       postgresql://hub:<pw>@hub-db:5432/hub
  VECTOR_DSN    $PGVECTOR_DB_URL               (optional; no request texts without it)
  PSEUDONYMS    path of the mapping file       (required; never inside OUT)
  OUT           output folder for agenthub-export-<date>/

Usage:
  python export_local.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

OWUI_DB = Path(os.getenv('OWUI_DB', '/app/backend/data/webui.db'))
OWUI_DATA = Path(os.getenv('OWUI_DATA', '/app/backend/data'))
HUB_DSN = os.environ['HUB_DSN']
VECTOR_DSN = os.getenv('VECTOR_DSN') or os.getenv('PGVECTOR_DB_URL') or ''
PSEUDONYMS = Path(os.environ['PSEUDONYMS'])
OUT_ROOT = Path(os.environ['OUT'])
SYSTEM_IDS = {'hub-seed': 'system'}
SKIP_FILE_PREFIXES = ('user-guide',)   # Open WebUI's built-in User Guide

DROP = {'user_name', 'user_email', 'owner_name', 'email', 'name_of_user'}
OWUI_TABLES = {
    'chat': 'select * from chat',
    'chat_message': 'select * from chat_message',
    'tag': 'select * from tag',
    'chat_review': 'select * from chat_review',
    'file': 'select * from file',
    'chat_file': 'select * from chat_file',
}
HUB_TABLES = ['agent', 'agent_version', 'step', 'step_version', 'hub_tool', 'hub_tool_version',
              'hub_install', 'agent_review', 'hub_tool_vote']


def _maybe_json(v):
    if isinstance(v, str) and v[:1] in '{[':
        try:
            return json.loads(v)
        except ValueError:
            pass
    return v


def _plain(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (bytes, memoryview)):
        return None
    if type(v).__name__ == 'UUID':
        return str(v)
    return v


class Pseudonymiser:
    def __init__(self, owui: sqlite3.Connection):
        self.map: dict[str, str] = json.loads(PSEUDONYMS.read_text()) if PSEUDONYMS.exists() else {}
        self.map.update(SYSTEM_IDS)
        users = owui.execute('select id, name, email, username from user order by created_at, id').fetchall()
        for uid, *_ in users:
            self.add(uid)
        # ids, emails, full names and usernames found inside free text; longest first
        self.swap: dict[str, str] = {}
        for uid, name, email, username in users:
            p = self.map[uid]
            for s in (uid, email, name, username):
                if s and len(s) >= 4:
                    self.swap[s] = p
        self.first_names = {n.split()[0] for _, n, *_ in users if n and ' ' in n and len(n.split()[0]) >= 3}

    def add(self, uid: str) -> str:
        if uid not in self.map:
            used = [int(v.split('-')[1]) for v in self.map.values() if re.fullmatch(r'user-\d+', v)]
            self.map[uid] = f'user-{max(used, default=0) + 1:02d}'
        return self.map[uid]

    def pattern(self):
        keys = sorted(self.swap, key=len, reverse=True)
        return re.compile('|'.join(re.escape(k) for k in keys), re.I) if keys else None

    def save(self):
        PSEUDONYMS.parent.mkdir(parents=True, exist_ok=True)
        PSEUDONYMS.write_text(json.dumps(self.map, indent=2))
        os.chmod(PSEUDONYMS, 0o600)


def _scrub(value, ps: Pseudonymiser, pat, lower, key=None):
    if isinstance(value, dict):
        return {k: _scrub(v, ps, pat, lower, k) for k, v in value.items() if k not in DROP}
    if isinstance(value, list):
        return [_scrub(v, ps, pat, lower) for v in value]
    if isinstance(value, str):
        if key and (key == 'user_id' or key.endswith('owner_id') or key == 'created_by') and value:
            return ps.map.get(value) or ps.add(value)
        if pat:
            value = pat.sub(lambda m: lower[m.group(0).lower()], value)
        return value
    return value


def _snapshot(src: Path) -> Path:
    """A consistent copy of the live SQLite database (backup API; safe while in use)."""
    dst = Path(tempfile.mkdtemp(prefix='owui-snap-')) / 'webui.db'
    with sqlite3.connect(f'file:{src}?mode=ro', uri=True) as live, sqlite3.connect(dst) as snap:
        live.backup(snap)
    return dst


def _upload_path(row: dict) -> Path | None:
    p = row.get('path') or ''
    if not p or '://' in p:          # s3 / gcs / azure storage: not a local file
        return None
    path = Path(p)
    if not path.is_absolute():
        path = OWUI_DATA / path
    return path if path.is_file() else None


def build() -> Path:
    import psycopg

    snap = _snapshot(OWUI_DB)
    owui = sqlite3.connect(f'file:{snap}?mode=ro', uri=True)
    owui.row_factory = sqlite3.Row
    ps = Pseudonymiser(owui)
    tables: dict[str, list[dict]] = {}
    for name, sql in OWUI_TABLES.items():
        rows = [{k: _maybe_json(r[k]) for k in r.keys()} for r in owui.execute(sql)]
        if name == 'file':
            # failed uploads (no text was ever extracted) are dead copies of a file the user
            # re-attached; keeping them would add empty duplicate cards to the graph
            failed = {r['id'] for r in rows if isinstance(r.get('data'), dict)
                      and r['data'].get('status') == 'failed' and not r['data'].get('content')}
            rows = [r for r in rows if not str(r.get('id', '')).startswith(SKIP_FILE_PREFIXES)
                    and r['id'] not in failed]
        if name == 'chat_file':   # links to skipped files would dangle
            kept = {r['id'] for r in tables['openwebui/file']}
            rows = [r for r in rows if r.get('file_id') in kept]
        tables[f'openwebui/{name}'] = rows
    owui.close()
    shutil.rmtree(snap.parent, ignore_errors=True)

    with psycopg.connect(HUB_DSN, options='-c default_transaction_read_only=on') as hub:
        for name in HUB_TABLES:
            cur = hub.execute(f'select * from "{name}"')
            cols = [c.name for c in cur.description]
            tables[f'hub/{name}'] = [{c: _maybe_json(_plain(v)) for c, v in zip(cols, row)} for row in cur]

    # Request texts: the per-user "user-tasks-<id>" collections in pgvector (text + metadata, no vectors).
    requests = []
    if VECTOR_DSN.startswith('postgres'):
        dsn = VECTOR_DSN.replace('postgresql+psycopg://', 'postgresql://').replace('postgresql+psycopg2://', 'postgresql://')
        try:
            with psycopg.connect(dsn, options='-c default_transaction_read_only=on') as vec:
                for coll, text, meta in vec.execute(
                        "select collection_name, text, vmetadata from document_chunk "
                        "where collection_name like 'user-tasks-%'"):
                    meta = meta if isinstance(meta, dict) else (_maybe_json(meta) or {})
                    requests.append({'user_id': coll.removeprefix('user-tasks-'), 'text': text,
                                     **{k: v for k, v in meta.items() if k not in ('text',)}})
        except Exception as e:   # no collections yet, or the table doesn't exist
            print(f'note: no request texts ({type(e).__name__}: {e})')
    tables['requests/user_tasks'] = requests

    pat = ps.pattern()
    lower = {k.lower(): v for k, v in ps.swap.items()}
    files_raw = {r['id']: r for r in tables['openwebui/file']}
    tables = {name: [_scrub(r, ps, pat, lower) for r in rows] for name, rows in tables.items()}
    ps.save()

    blob = json.dumps(tables, default=str)
    leftovers = {n: len(re.findall(rf'\b{re.escape(n)}\b', blob)) for n in sorted(ps.first_names)}
    leftover_ids = [u for u in ps.map if u not in SYSTEM_IDS and u in blob]

    stamp = dt.date.today().isoformat()
    folder = OUT_ROOT / f'agenthub-export-{stamp}'
    if folder.exists():
        shutil.rmtree(folder)
    for name, rows in tables.items():
        path = folder / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))

    # Original upload bytes, named <file_id>_<name> (what the Ping engine looks for).
    copied, missing = 0, []
    up = folder / 'uploads'
    scrubbed_names = {r['id']: r.get('filename') for r in tables['openwebui/file']}
    for fid, row in files_raw.items():
        src = _upload_path(row)
        if not src:
            missing.append(fid)
            continue
        # the pseudonymised filename: the raw one can carry a real name ("dave_healthcheck.py").
        # The bytes themselves are copied as uploaded and are NOT scrubbed.
        name = scrubbed_names.get(fid) or src.name.split('_', 1)[-1]
        name = re.sub(r'[^\w.\- ]', '_', name)
        up.mkdir(exist_ok=True)
        shutil.copyfile(src, up / f'{fid}_{name}')
        copied += 1

    manifest = {
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'source': 'local stack (export_local.py)',
        'counts': {name: len(rows) for name, rows in tables.items()},
        'uploads_copied': copied,
        'people': sorted(set(ps.map.values())),
        'pseudonyms': 'user-NN by account creation date; "system" is the hub seed owner. '
                      'Stable across exports; the mapping is not included.',
        'removed': 'name and email fields; user ids, emails, usernames and full names wherever they appeared',
        'not_included': 'user accounts, embeddings/vectors, LiteLLM data, settings, the built-in User Guide',
        'caveat': 'uploads/ holds the original file bytes as uploaded; names inside them are not pseudonymised',
        'first_names_still_in_text': {f'name-{i + 1}': n for i, n in enumerate(leftovers.values())},
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'export': str(folder), **manifest, 'uploads_missing': len(missing),
                      'raw_user_ids_left': len(leftover_ids)}, indent=2))
    return folder


if __name__ == '__main__':
    build()
