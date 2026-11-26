"""
Pull the bench's Open WebUI chats (the AI bench moved there on 2026-11-14) into raw JSON,
one file per chat, for bench_to_docs.py --owui to turn into chat-<id>.md docs.

Since 2026-11-15 there is one chat a day (meta.day), which becomes one dated doc.

Only what's worth remembering:
  - chats with the bench's models (default: ai-bench), not archived (archived = tests /
    put away), owned by the signed-in user
  - the branch that was kept (history.currentId back to the root), not regenerated
    alternatives
  - tool calls as name + arguments + a short result; the text the model wrote

Files under <out>/ mirror the chats that exist now: a chat that was archived or deleted
since the last run has its file removed, so its doc drops out of the index too.

Sign-in: OWUI_TOKEN, or WEBUI_ADMIN_EMAIL / WEBUI_ADMIN_PASSWORD from --env (default
~/repos/resnovae_chat/.env). Nothing leaves this machine here; it reads Open WebUI's API.

Usage:
  python owui_to_docs.py <out_dir> [--url http://localhost:3080] [--models ai-bench,planner]
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DETAILS = re.compile(r"<details\b[^>]*>.*?</details>", re.S)
TOOL_DETAILS = re.compile(r'<details\s+type="tool_calls"([^>]*)>', re.S)
ATTR = re.compile(r'(\w+)="([^"]*)"')


def api(base: str, path: str, token: str | None = None, body: dict | None = None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def env_file(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def thread(history: dict) -> list[dict]:
    msgs = history.get("messages") or {}
    cur, out = history.get("currentId"), []
    while cur and cur in msgs:
        out.append(msgs[cur])
        cur = msgs[cur].get("parentId")
    return list(reversed(out))


def tools_of(m: dict) -> list[dict]:
    """Tool calls of one assistant message: from the native-call output list, else the
    <details type="tool_calls"> blocks in its content."""
    calls, results = [], {}
    for o in m.get("output") or []:
        if o.get("type") == "function_call":
            calls.append({"id": o.get("call_id") or o.get("id"), "name": o.get("name", ""), "args": o.get("arguments") or ""})
        elif o.get("type") == "function_call_output":
            out = o.get("output")
            if isinstance(out, list):
                out = " ".join(p.get("text", "") for p in out if isinstance(p, dict))
            results[o.get("call_id")] = str(out or "")
    if calls:
        return [{"name": c["name"], "args": c["args"], "result": results.get(c["id"], "")} for c in calls]
    found = []
    for attrs in TOOL_DETAILS.findall(m.get("content") or ""):
        a = {k: html.unescape(v) for k, v in ATTR.findall(attrs)}
        found.append({"name": a.get("name", ""), "args": a.get("arguments", ""), "result": a.get("result", "")})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default=os.getenv("OWUI_URL", "http://localhost:3080"))
    ap.add_argument("--models", default="ai-bench", help="comma list of model ids to keep")
    ap.add_argument("--env", default=str(Path.home() / "repos" / "resnovae_chat" / ".env"))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    want = {m.strip() for m in a.models.split(",") if m.strip()}

    token = os.getenv("OWUI_TOKEN")
    if not token:
        e = env_file(Path(a.env))
        if not (e.get("WEBUI_ADMIN_EMAIL") and e.get("WEBUI_ADMIN_PASSWORD")):
            print("set OWUI_TOKEN, or WEBUI_ADMIN_EMAIL / WEBUI_ADMIN_PASSWORD in --env", file=sys.stderr)
            return 1
        token = api(base, "/api/v1/auths/signin", body={"email": e["WEBUI_ADMIN_EMAIL"], "password": e["WEBUI_ADMIN_PASSWORD"]})["token"]

    listing, page = [], 1
    while True:
        rows = api(base, f"/api/v1/chats/?page={page}", token)
        if not rows:
            break
        listing += rows
        page += 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, written = set(), 0
    for row in listing:
        c = api(base, f"/api/v1/chats/{row['id']}", token)
        chat = c.get("chat") or {}
        models = set(chat.get("models") or [])
        turns = []
        for m in thread(chat.get("history") or {}):
            if m.get("model"):
                models.add(m["model"])
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            text = m.get("content") or ""
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
            turns.append({"role": role, "text": DETAILS.sub("", text).strip(), "at": m.get("timestamp"),
                          "model": m.get("model") or "", "tools": tools_of(m) if role == "assistant" else []})
        if not (models & want) or not any(t["role"] == "user" for t in turns):
            continue
        doc = {"id": c["id"], "title": c.get("title") or chat.get("title") or "", "created_at": c.get("created_at"),
               "updated_at": c.get("updated_at"), "models": sorted(models), "turns": turns,
               # one chat a day (Open WebUI routers/bench.py /day): the date it belongs to
               "day": (c.get("meta") or {}).get("day")}
        f = out / f"{c['id']}.json"
        text = json.dumps(doc, ensure_ascii=False, indent=1)
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            written += 1
        kept.add(f.name)
    gone = [f for f in out.glob("*.json") if f.name not in kept]
    for f in gone:
        f.unlink()
    print(f"{len(kept)} chats ({', '.join(sorted(want))}) of {len(listing)}, {written} written, {len(gone)} removed -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
