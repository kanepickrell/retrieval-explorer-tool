"""
Chime-in assist: as a conversation unfolds, decide what to look up, retrieve
with any technique, and surface docs that would help -- the "retrieval explorer
chimes in" mode of the chat panel.

  GET  /api/assist/chats?dataset=   replayable held-out chats (real export chats)
  POST /api/assist                  one chime-in decision for the conversation so far

Per turn:
  1. build a search query from the conversation, by strategy:
       last     the latest user message
       window   the last few messages, concatenated
       rewrite  LLM writes one standalone search query for what would help now,
                or NONE when nothing is worth looking up (the assistant stays quiet)
  2. retrieve (run_unified, any technique), excluding the replayed chat's own doc
     and anything already suggested in this conversation
  3. filter candidate docs:
       score    cosine(query, chunk) >= min_score
       llm      the judge marks each candidate useful or not, with a one-line
                reason that becomes the suggestion's "why" (toggleable, to measure
                precision and latency with and without it)
Everything is timed per stage and the retrieval is a normal run (run_id), so a
chime-in opens in the Explorer's graph / chunks / steps / metrics panels.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import datasets as DS
from graph_search import run_unified

router = APIRouter(prefix="/api/assist", tags=["Assist (chime-in)"])

WINDOW_TURNS = 3
WINDOW_CHARS = 1500
MAX_QUERY_CHARS = 600


class Turn(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class AssistRequest(BaseModel):
    dataset: Optional[str] = None
    turns: List[Turn]
    technique: str = "bm25_vector"
    query_strategy: str = Field("rewrite", pattern="^(last|window|rewrite)$")
    limit: int = Field(5, ge=1, le=20)
    top_k: Optional[int] = Field(None, le=200)
    llm_filter: bool = True
    min_score: float = 0.5
    max_suggestions: int = Field(3, ge=1, le=10)
    exclude_docs: List[str] = []          # e.g. the replayed chat's own doc
    already_suggested: List[str] = []     # docs shown earlier in this conversation


def _text_of(turns: List[Turn]) -> str:
    return "\n".join(f"{t.role.upper()}: {t.content.strip()}" for t in turns)


async def _build_query(req: AssistRequest) -> tuple:
    """-> (query or None, strategy detail, ms)."""
    t0 = time.perf_counter()
    users = [t for t in req.turns if t.role == "user" and t.content.strip()]
    if not users:
        return None, "no user message yet", 0.0
    if req.query_strategy == "last":
        q, detail = users[-1].content.strip(), "latest user message"
    elif req.query_strategy == "window":
        recent = req.turns[-WINDOW_TURNS:]
        q, detail = " ".join(t.content.strip() for t in recent)[-WINDOW_CHARS:], f"last {len(recent)} messages"
    else:
        from retrieval import lightrag_provider as lp
        convo = _text_of(req.turns[-6:])[-4000:]
        out = await lp.llm(
            "Below is the latest part of a conversation between a user and an AI assistant. "
            "Write ONE standalone search query (at most 20 words) for internal documents or past "
            "conversations that would help the user with what they are doing right now. Use the "
            "specific names, tools, IDs and errors mentioned. If nothing would plausibly help "
            "(small talk, thanks, a finished task), reply with exactly NONE.\n\n"
            f"CONVERSATION:\n{convo}\n\nSEARCH QUERY:",
            system_prompt="You write search queries. Output only the query, or NONE.")
        q = (out or "").strip().strip('"').splitlines()[0].strip() if (out or "").strip() else ""
        detail = "LLM rewrite of the recent conversation"
        if not q or q.upper().startswith("NONE"):
            return None, detail + " → NONE (nothing worth looking up)", round((time.perf_counter() - t0) * 1000, 1)
    return q[:MAX_QUERY_CHARS], detail, round((time.perf_counter() - t0) * 1000, 1)


def _json_list(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except Exception:
        return []


async def _judge(req: AssistRequest, query: str, cands: List[dict]) -> List[dict]:
    """LLM marks each candidate useful (bool) with a one-line reason."""
    from retrieval import lightrag_provider as lp
    convo = _text_of(req.turns[-4:])[-3000:]
    listing = "\n\n".join(f"[{i + 1}] {c['title']}\n{c['snippet']}" for i, c in enumerate(cands))
    out = await lp.llm(
        "A user is in the conversation below. For each numbered document excerpt, decide whether "
        "showing it to the user RIGHT NOW would genuinely help with what they are doing (not merely "
        "on the same general topic). Return ONLY a JSON array with one object per excerpt, in order: "
        '{"i": <number>, "useful": true|false, "why": "<one short sentence addressed to the user>"}.\n\n'
        f"CONVERSATION:\n{convo}\n\nDOCUMENT EXCERPTS:\n{listing}",
        system_prompt="You judge whether documents would help a user right now. Output JSON only.")
    verdicts = {int(v.get("i", 0)): v for v in _json_list(out) if isinstance(v, dict)}
    for i, c in enumerate(cands, 1):
        v = verdicts.get(i)
        c["useful"] = bool(v.get("useful")) if v else None   # None = judge gave no verdict
        c["why"] = str(v.get("why") or "").strip() if v else ""
    return cands


@router.post("")
async def assist(req: AssistRequest):
    t_all = time.perf_counter()
    ds = DS.get(req.dataset)
    query, q_detail, q_ms = await _build_query(req)
    stages = [{"label": "Build query", "detail": q_detail, "stat": query or "(stay quiet)", "ms": q_ms}]
    if not query:
        return {"query": None, "stay_quiet": True, "reason": q_detail, "suggestions": [],
                "stages": stages, "retrieval": None, "total_ms": round((time.perf_counter() - t_all) * 1000, 1)}

    exclude = set(req.exclude_docs)
    retrieval = await run_unified(query, req.limit, req.technique, ds.id, req.top_k,
                                  exclude_docs=exclude or None)
    stages.append({"label": f"Retrieve ({req.technique})", "detail": f"{len(retrieval['results'])} chunks",
                   "stat": f"excluding {len(exclude)} doc(s)", "ms": retrieval["time_ms"]})

    # candidates: distinct docs, best-ranked chunk first, not already suggested
    seen, cands = set(req.already_suggested), []
    for r in retrieval["results"]:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        cands.append({"doc": r["id"], "chunk_id": r.get("chunk_id"), "title": r["label"],
                      "snippet": (r.get("description") or "")[:400], "score": r.get("relevance_score"),
                      "rank": r.get("rank"), "via": r.get("via")})
    above = [c for c in cands if c["score"] is not None and c["score"] >= req.min_score]
    stages.append({"label": "Score threshold", "detail": f"cosine(query, chunk) ≥ {req.min_score}",
                   "stat": f"{len(above)} of {len(cands)} new docs pass", "ms": 0.0})

    shown = above
    if req.llm_filter and above:
        t = time.perf_counter()
        judged = await _judge(req, query, above[: req.max_suggestions + 2])
        shown = [c for c in judged if c["useful"]]
        stages.append({"label": "LLM usefulness check", "detail": "judge: would this help right now?",
                       "stat": f"{len(shown)} of {len(judged)} judged useful",
                       "ms": round((time.perf_counter() - t) * 1000, 1)})
    shown = shown[: req.max_suggestions]
    return {
        "query": query, "stay_quiet": not shown,
        "reason": None if shown else "nothing passed the filters",
        "suggestions": shown, "candidates": cands,
        "stages": stages, "retrieval": retrieval,
        "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
    }


@router.get("/chats")
def replayable_chats(dataset: Optional[str] = Query(None), min_user_turns: int = Query(2, ge=1)):
    """Real chats from the dataset's export that can be replayed turn by turn.
    Each comes with the doc id to exclude so the chat can't retrieve itself."""
    ds = DS.get(dataset)
    if not ds.docs_dir:
        return []
    ex = ds.docs_dir.parent
    if not (ex / "openwebui" / "chat.json").exists():
        return []
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))
    from export_to_docs import load, msg_text
    msgs = {}
    for m in load(ex / "openwebui" / "chat_message.json"):
        msgs.setdefault(m["chat_id"], []).append(m)
    indexed = {p.name for p in ds.docs_dir.glob("chat-*.md")}
    out = []
    for c in load(ex / "openwebui" / "chat.json"):
        doc = f"chat-{c['id']}.md"
        if doc not in indexed:
            continue
        turns = [{"role": m["role"], "content": msg_text(m).strip()}
                 for m in sorted(msgs.get(c["id"], []), key=lambda m: int(m.get("created_at") or 0))]
        turns = [t for t in turns if t["content"]]
        n_user = sum(1 for t in turns if t["role"] == "user")
        if n_user >= min_user_turns:
            out.append({"chat_id": c["id"], "doc": doc, "title": c.get("title") or "Chat",
                        "author": c.get("user_id"), "user_turns": n_user, "turns": turns})
    out.sort(key=lambda x: -x["user_turns"])
    return out
