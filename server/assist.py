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

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

import datasets as DS
from graph_search import run_unified

router = APIRouter(prefix="/api/assist", tags=["Assist (chime-in)"])

WINDOW_TURNS = 3
WINDOW_CHARS = 1500
MAX_QUERY_CHARS = 600

# doc id -> human title, per dataset. Chat docs carry real titles in the export's
# chat.json (a uuid can't be prettified); other doc kinds fall back to their label
# and the client prettifies them. Cached by storage path.
_TITLES_CACHE: dict = {}


def _doc_titles(ds) -> dict:
    key = str(getattr(ds, "storage", ds))
    if key in _TITLES_CACHE:
        return _TITLES_CACHE[key]
    titles: dict = {}
    docs_dir = getattr(ds, "docs_dir", None)
    if docs_dir:
        chat_json = Path(docs_dir).parent / "openwebui" / "chat.json"
        if chat_json.exists():
            try:
                for c in json.loads(chat_json.read_text(encoding="utf-8")):
                    cid, t = c.get("id"), (c.get("title") or "").strip()
                    if cid and t:
                        titles[f"chat-{cid}.md"] = t
            except Exception:
                pass
    _TITLES_CACHE[key] = titles
    return titles


def doc_title(ds, doc: str, markdown: Optional[str] = None) -> str:
    """Human title for a doc: the chat's real title, else the doc's first heading."""
    t = _doc_titles(ds).get(doc)
    if t and t.strip().lower() not in _GENERIC_TITLES:
        return t
    if t:   # "Greeting", "hi there": name it after what the user actually asked
        return _first_request(markdown if markdown is not None else _doc_md(ds, doc)) or t
    if markdown is None and getattr(ds, "docs_dir", None) and (Path(ds.docs_dir) / doc).is_file():
        markdown = (Path(ds.docs_dir) / doc).read_text(encoding="utf-8", errors="replace")[:2000]
    # first heading that is a real title, not a filename like "05-files-and-code-in-chat.md"
    for h in re.findall(r"^#\s+(.+)$", markdown or "", re.M):
        if not re.search(r"\.[A-Za-z0-9]{1,5}$", h.strip()):
            return h.strip()
    m = re.search(r"^#\s+(.+)$", markdown or "", re.M)
    return m.group(1).strip() if m else doc


_GENERIC_TITLES = {"greeting", "hi there", "hi", "hello", "test message", "test", "new chat",
                   "empty conversation", "chat", "untitled"}


def _doc_md(ds, doc: str) -> str:
    d = getattr(ds, "docs_dir", None)
    f = Path(d) / doc if d else None
    return f.read_text(encoding="utf-8", errors="replace") if f and f.is_file() else ""


def _first_request(md: str) -> str:
    """First substantive user message of a chat doc (for chats titled 'Greeting' etc.)."""
    for sec in re.split(r"^## User\s*$", md, flags=re.M)[1:]:
        msg = sec.split("\n## ", 1)[0].strip()
        if len(msg) >= 25 and not msg.lower().startswith(("hi", "hello", "test")):
            return re.sub(r"\s+", " ", msg)[:90]
    return ""


def _judge_text(ds, c: dict) -> str:
    """What the judge reads for a candidate: the matched chunk, plus -- for a past chat --
    the assistant's answer that follows it, since the chunk is often just the question."""
    chunk = c.get("text") or c.get("snippet") or ""
    if not c["doc"].startswith("chat-"):
        # a guide / file / agent: its headings tell the judge what the whole doc covers
        heads = re.findall(r"^#{2,3}\s+(.+)$", _doc_md(ds, c["doc"]), re.M)[:16]
        return (f"(sections: {'; '.join(h.strip() for h in heads)})\n" if heads else "") + chunk[:900]
    about = _first_request(_doc_md(ds, c["doc"]))
    head = f"(this chat began with: {about})\n" if about else ""
    md = _doc_md(ds, c["doc"])
    probe = re.sub(r"\s+", " ", chunk[:80]).strip()
    flat = re.sub(r"\s+", " ", md)
    at = flat.find(probe) if probe else -1
    rest = flat[at + len(probe):] if at >= 0 else flat
    m = re.search(r"## Assistant (.{0,900})", rest)
    return head + chunk[:700] + ("\nANSWER IN THAT CHAT: " + m.group(1) if m else "")


# Ping curation, per dataset: <export>/ping.json says which docs Ping may never
# suggest (index pages, greetings, tests, off-topic chats) -- the corpus and its
# index stay untouched, so the Explorer still sees everything.
#   {"exclude_prefixes": ["person-", ...], "exclude_docs": {"chat-<id>.md": "why", ...}}
def _ping_excluded(ds) -> set:
    docs_dir = getattr(ds, "docs_dir", None)
    if not docs_dir:
        return set()
    f = Path(docs_dir).parent / "ping.json"
    if not f.exists():
        return set()
    try:
        rules = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return set()
    out = set(rules.get("exclude_docs") or {})
    prefixes = tuple(rules.get("exclude_prefixes") or ())
    if prefixes:
        out |= {p.name for p in Path(docs_dir).glob("*.md") if p.name.startswith(prefixes)}
    return out


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
    # Ping's automatic mode: only search once the conversation has made clear what
    # the user is working on and what they need (rewrite strategy only). A manual
    # Ping skips this and searches right away.
    require_context: bool = False
    # Apply the dataset's ping.json curation (docs Ping must never suggest).
    curated: bool = False


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
    elif req.require_context:
        return await _gated_query(req, t0)
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


_GATE_PROMPT = (
    "A user is chatting with an AI assistant. Read the conversation and fill in what is known so far "
    "about the user's situation. Do not guess: leave a field \"\" until the user has actually said it.\n"
    "  who   - the system, tool, codebase, file, agent, team or role involved\n"
    "  what  - the concrete task, question or problem (an error, a failing step, something to build)\n"
    "  why   - the goal or reason, or the specifics that pin it down (error text, names, constraints)\n"
    "  topic - \"work\" if it concerns technical or professional work (IT, security, red/blue team, "
    "exercises, code, data, Agent Hub agents/tools/workflows, files); \"chitchat\" for greetings, "
    "thanks, tests or vague openers; \"other\" for trivia, general knowledge, personal life, "
    "study or writing help unrelated to work\n"
    "  query - ONE search query of at most 20 words for the team's internal knowledge base (past chats, "
    "uploaded files, agents, tools), built from the specific names, tools, files, IDs and errors "
    "mentioned; \"\" if the task is not known yet\n\n"
    "Reply with JSON only: {{\"who\": \"\", \"what\": \"\", \"why\": \"\", \"topic\": \"\", \"query\": \"\"}}\n\n"
    "CONVERSATION:\n{convo}\n\nJSON:"
)


def _filled(v: dict, k: str) -> bool:
    x = str(v.get(k) or "").strip().lower()
    return len(x) > 2 and x not in ("unknown", "none", "n/a", "not specified", "not stated", "?")


async def _gated_query(req: AssistRequest, t0: float) -> tuple:
    """Rewrite strategy that waits until the conversation says who/what/why.

    The LLM only extracts; the decision is made here, because small models are
    unreliable at judging their own readiness. Search once it is work and the
    task (what) is known along with either who/what system or why/specifics."""
    from retrieval import lightrag_provider as lp
    convo = _text_of(req.turns[-6:])[-4000:]
    out = await lp.llm(_GATE_PROMPT.format(convo=convo),
                       system_prompt="You extract facts from conversations. Output JSON only.")
    m = re.search(r"\{.*\}", out or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except Exception:
        v = {}
    ms = round((time.perf_counter() - t0) * 1000, 1)
    topic = str(v.get("topic") or "").strip().lower()
    known = " · ".join(f"{k}: {str(v.get(k) or '').strip()[:80] or '?'}" for k in ("who", "what", "why"))
    q = str(v.get("query") or "").strip().strip('"')
    if not q and _filled(v, "what"):   # small models sometimes skip the query field
        q = " ".join(str(v.get(k) or "") for k in ("who", "what", "why") if _filled(v, k))[:200]
    if topic != "work":
        return None, f"not a work question (topic: {topic or '?'}; {known})", ms
    # who + what + why all stated, or two of them in a substantial request (a long first
    # message can carry everything; a short opener like "I'm working on X" cannot).
    n = sum(_filled(v, k) for k in ("who", "what", "why"))
    last_user = next((t.content for t in reversed(req.turns) if t.role == "user"), "")
    detailed = len(last_user.split()) >= 15
    if not _filled(v, "what") or not (n == 3 or (n == 2 and detailed)) or not q:
        return None, f"waiting for context ({known})", ms
    return q[:MAX_QUERY_CHARS], f"context clear ({known})", ms


def _json_list(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:   # one candidate: models often answer with a bare object
        m1 = re.search(r"\{.*\}", text or "", re.S)
        try:
            v = json.loads(m1.group(0)) if m1 else None
        except Exception:
            v = None
        return [v] if isinstance(v, dict) else []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except Exception:
        return []


_KIND = {"chat": "past chat", "file": "uploaded file", "agent": "Agent Hub agent",
         "step": "workflow step", "tool": "Agent Hub tool", "catalog": "catalog page", "person": "user activity"}


async def _judge(req: AssistRequest, query: str, cands: List[dict]) -> List[dict]:
    """LLM marks each candidate useful (bool) with a one-line reason."""
    from retrieval import lightrag_provider as lp
    convo = _text_of(req.turns[-4:])[-3000:]
    ds = DS.get(req.dataset)
    listing = "\n\n".join(f"[{i + 1}] {_KIND.get(c['doc'].split('-', 1)[0], 'document')}: {c['title']}\n"
                           f"{_judge_text(ds, c)}" for i, c in enumerate(cands))
    out = await lp.llm(
        "A user is in the conversation below. Each numbered excerpt comes from the team's internal "
        "knowledge base. Mark an excerpt useful ONLY if showing it to the user right now would move "
        "their task forward: it holds a concrete answer, fix, code, configuration, procedure, prior "
        "discussion of the same problem, or names an agent/tool/workflow built for this job. Being "
        "on the same general topic is NOT enough; an excerpt that merely repeats what the "
        "conversation already established is NOT useful. Judge the document as a whole from its "
        "title and opening request -- the excerpt is only a sample of it. A past chat where someone asked the same or "
        "a similar question IS useful when its answer helps -- that is the point of surfacing it. "
        "When unsure, mark it not useful.\n"
        "Return ONLY a JSON array with one object per excerpt, in order: "
        '{"i": <number>, "useful": true|false, "why": "<one short sentence telling the user what '
        'in it helps, starting with a verb and never with \'This excerpt\', e.g. \'Shows the fix for the '
        'same NTLM 400 error\'>"}.\n\n'
        f"CONVERSATION:\n{convo}\n\nEXCERPTS:\n{listing}",
        system_prompt="You judge whether documents would help a user right now. Output JSON only.")
    verdicts = {int(v.get("i", 0)): v for v in _json_list(out) if isinstance(v, dict)}
    for i, c in enumerate(cands, 1):
        v = verdicts.get(i)
        c["useful"] = bool(v.get("useful")) if v else None   # None = judge gave no verdict
        c["why"] = str(v.get("why") or "").strip() if v else ""
    return cands


@router.post("")
async def assist(req: AssistRequest, request: Request):
    """Run the chime-in and log it to the Live feed (see live.py)."""
    import live
    ua = request.headers.get("user-agent")
    try:
        resp = await _assist(req)
    except Exception as e:
        live.record(req, {}, ua, error=f"{type(e).__name__}: {e}")
        raise
    live.record(req, resp, ua)
    return resp


async def _assist(req: AssistRequest):
    t_all = time.perf_counter()
    ds = DS.get(req.dataset)
    query, q_detail, q_ms = await _build_query(req)
    stages = [{"label": "Build query", "detail": q_detail, "stat": query or "(stay quiet)", "ms": q_ms}]
    if not query:
        return {"query": None, "stay_quiet": True, "reason": q_detail, "suggestions": [],
                "stages": stages, "retrieval": None, "total_ms": round((time.perf_counter() - t_all) * 1000, 1)}

    exclude = set(req.exclude_docs)
    if req.curated:
        exclude |= _ping_excluded(ds)
    retrieval = await run_unified(query, req.limit, req.technique, ds.id, req.top_k,
                                  exclude_docs=exclude or None)
    stages.append({"label": f"Retrieve ({req.technique})", "detail": f"{len(retrieval['results'])} chunks",
                   "stat": f"excluding {len(exclude)} doc(s)", "ms": retrieval["time_ms"]})

    # candidates: distinct docs, best-ranked chunk first, not already suggested
    titles = _doc_titles(ds)
    seen, cands = set(req.already_suggested), []
    for r in retrieval["results"]:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        cands.append({"doc": r["id"], "chunk_id": r.get("chunk_id"),
                      "title": doc_title(ds, r["id"]) or r["label"],
                      "snippet": (r.get("description") or "")[:400],
                      "text": (r.get("description") or "")[:1200], "score": r.get("relevance_score"),
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
    for c in cands:
        c.pop("text", None)
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
