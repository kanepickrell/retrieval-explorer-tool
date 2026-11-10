"""
Convert an Agent Hub export (export_package.py output) into one markdown doc
per corpus item, ready for LightRAG ingestion.

  chats      -> chat-<chat_id>.md     (title, tags, user/assistant turns)
  files      -> file-<file_id>.md     (extracted text of uploaded / KB files)
  agents     -> agent-<slug>.md       (latest manifest: description, system prompt, steps, tools)
  steps      -> step-<slug>.md
  hub tools  -> tool-<slug>.md

Trivial chats (greetings / thanks with no real request) are skipped. Doc ids
keep the source ids so eval gold labels can point at them (chat-<chat_id>.md).

Usage:
  python export_to_docs.py <export_dir> <out_docs_dir>
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
MIN_USER_CHARS = 40       # chats whose user turns total less than this are small talk
MAX_MSG_CHARS = 8000      # cap very long pastes / tool dumps per message


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def loose(v):
    """Decode a field that may be JSON, a Python repr, or already a value."""
    if v is None or isinstance(v, (dict, list)):
        return v
    s = str(v)
    if s in ("null", "None", ""):
        return None
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:
            pass
    return s


def msg_text(m: dict) -> str:
    c = loose(m.get("content"))
    if isinstance(c, str) and c.strip():
        return c
    out = loose(m.get("output")) or []
    parts = []
    for o in out if isinstance(out, list) else []:
        for c in (o.get("content") or []) if isinstance(o, dict) else []:
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"])
    return "\n".join(parts)


def clip(s: str) -> str:
    s = JWT.sub("[REDACTED-JWT]", s or "").strip()
    return s if len(s) <= MAX_MSG_CHARS else s[:MAX_MSG_CHARS] + "\n…[truncated]"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60] or "untitled"


def main(export_dir: str, out_dir: str) -> None:
    ex, out = Path(export_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written, skipped = defaultdict(int), []

    # ── chats ──
    msgs = defaultdict(list)
    for m in load(ex / "openwebui/chat_message.json"):
        msgs[m["chat_id"]].append(m)
    for c in load(ex / "openwebui/chat.json"):
        turns = sorted(msgs.get(c["id"], []), key=lambda m: int(m.get("created_at") or 0))
        user_chars = sum(len(msg_text(m)) for m in turns if m["role"] == "user")
        if user_chars < MIN_USER_CHARS:
            skipped.append(c["id"])
            continue
        meta = loose(c.get("meta")) or {}
        tags = ", ".join(meta.get("tags", []) if isinstance(meta, dict) else [])
        lines = [f"# {c.get('title') or 'Chat'}", ""]
        if tags:
            lines += [f"Tags: {tags}", ""]
        for m in turns:
            t = clip(msg_text(m))
            if t:
                lines += [f"## {'User' if m['role'] == 'user' else 'Assistant'}", "", t, ""]
        (out / f"chat-{c['id']}.md").write_text("\n".join(lines), encoding="utf-8")
        written["chat"] += 1

    # ── files (extracted text) ──
    for f in load(ex / "openwebui/file.json"):
        data = loose(f.get("data")) or {}
        text = data.get("content") if isinstance(data, dict) else str(data)
        if not (text or "").strip():
            continue
        body = f"# {f.get('filename') or f['id']}\n\n{clip(text) if len(text) < 60000 else text[:60000]}"
        (out / f"file-{f['id']}.md").write_text(JWT.sub("[REDACTED-JWT]", body), encoding="utf-8")
        written["file"] += 1

    # ── hub: agents / steps / tools (latest version manifest) ──
    def latest(rows, key):
        best = {}
        for r in rows:
            if r[key] not in best or r["version"] > best[r[key]]["version"]:
                best[r[key]] = r
        return best.values()

    for kind, path, key in (("agent", "hub/agent_version.json", "agent_id"),
                            ("step", "hub/step_version.json", "step_id"),
                            ("tool", "hub/hub_tool_version.json", "tool_id")):
        for v in latest(load(ex / path), key):
            man = loose(v.get("manifest")) or {}
            name = man.get("name") or man.get("slug") or v[key]
            lines = [f"# {name} ({kind})", ""]
            if man.get("description"):
                lines += [man["description"], ""]
            if man.get("tags"):
                lines += [f"Tags: {', '.join(man['tags'])}", ""]
            for field in ("system", "instructions", "prompt"):
                if man.get(field):
                    lines += [f"## {field.title()}", "", clip(str(man[field])), ""]
            rest = {k: val for k, val in man.items()
                    if k not in ("name", "slug", "description", "tags", "system", "instructions", "prompt", "manifest_version")
                    and val not in (None, [], {}, "")}
            if rest:
                lines += ["## Definition", "", "```json", clip(json.dumps(rest, indent=2, ensure_ascii=False)), "```", ""]
            (out / f"{kind}-{slug(man.get('slug') or name)}.md").write_text("\n".join(lines), encoding="utf-8")
            written[kind] += 1

    total = sum(p.stat().st_size for p in out.glob("*.md"))
    print(f"wrote {dict(written)} -> {out}  ({total/1024:.0f} KB)")
    print(f"skipped {len(skipped)} small-talk chats")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
