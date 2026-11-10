"""
Build a gold eval set from an Agent Hub export's automatic chat reviews.

Selection: chats the review scored 2, plus chats scored 1 with no pain points,
that made it into the indexed docs (export_to_docs.py skips small talk).

Per chat:
  query      = the first user message
  reference  = the assistant's first reply to it  (used by the reference-based
               Ragas metrics; reflects what the assistant said, not verified truth)
  gold_ids   = the chat's own doc + any files attached to that chat

Output matches metrics.py's gold format, so point the backend at it with
PROTOGRAPH_GOLD=<out.json>.

Usage:
  python build_eval_set.py <export_dir> <docs_dir> <out.json>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_to_docs import load, msg_text  # noqa: E402

MAX_QUERY_CHARS = 600
MIN_QUERY_CHARS = 20   # drops greetings like "hi there"


def main(export_dir: str, docs_dir: str, out_path: str) -> None:
    ex, docs = Path(export_dir), {p.name for p in Path(docs_dir).glob("*.md")}

    msgs = defaultdict(list)
    for m in load(ex / "openwebui/chat_message.json"):
        msgs[m["chat_id"]].append(m)
    files = defaultdict(list)
    for cf in load(ex / "openwebui/chat_file.json"):
        files[cf["chat_id"]].append(cf["file_id"])
    titles = {c["id"]: c.get("title") for c in load(ex / "openwebui/chat.json")}

    reviews = {r["chat_id"]: r for r in load(ex / "openwebui/chat_review.json")}
    picked = [r for r in reviews.values()
              if r["score"] == 2 or (r["score"] == 1 and not r["pain_points"])]

    out, dropped = [], []
    for r in picked:
        cid = r["chat_id"]
        if f"chat-{cid}.md" not in docs:
            dropped.append((cid, "not indexed"))
            continue
        turns = sorted(msgs[cid], key=lambda m: int(m.get("created_at") or 0))
        first_user = next((i for i, m in enumerate(turns) if m["role"] == "user" and msg_text(m).strip()), None)
        if first_user is None:
            dropped.append((cid, "no user turn"))
            continue
        reply = next((m for m in turns[first_user + 1:] if m["role"] == "assistant" and msg_text(m).strip()), None)
        if reply is None:
            dropped.append((cid, "no assistant reply"))
            continue
        q = msg_text(turns[first_user]).strip()
        if len(q) < MIN_QUERY_CHARS:
            dropped.append((cid, f"query too short ({q!r})"))
            continue
        gold = [f"chat-{cid}.md"] + [f"file-{f}.md" for f in files[cid] if f"file-{f}.md" in docs]
        out.append({
            "id": f"q-{cid[:8]}",
            "query": q[:MAX_QUERY_CHARS],
            "query_truncated": len(q) > MAX_QUERY_CHARS,
            "reference": msg_text(reply).strip(),
            "gold_ids": gold,
            "review_score": r["score"],
            "chat_title": titles.get(cid),
            "query_type": "real_user_request",
        })

    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} eval queries -> {out_path}  "
          f"(score2={sum(e['review_score'] == 2 for e in out)}, "
          f"truncated queries={sum(e['query_truncated'] for e in out)})")
    for cid, why in dropped:
        print(f"  dropped {cid[:8]}: {why}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
