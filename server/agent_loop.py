"""
Routing agent: works out what the user is after, hypothesises where the answer
lives, checks those hypotheses against the metadata graph, and only then dives
into content -- scoped to the records the graph points at.

  1. goal       LLM reads the conversation: the user's goal, what they need from the
                knowledge base now, and whether that is clear yet (if not, it waits)
  2. hypothesise 1-3 claims about where the answer lives, each with an answer type
                (records = a set/count of records from metadata, document = a
                particular chat/file/agent, detail = a fact inside a document), a
                target kind and a scope (users, teams, agents, files, terms)
  3. validate   deterministic, against the graph: do the named records exist, which
                docs fall in the scope, do the best-matching record cards land inside it
  4. revise     if no hypothesis holds, the LLM sees that evidence and revises (once)
  5. route      records -> answer from the graph itself (traversal, exact)
                document -> record cards, restricted to the scope
                detail -> chunk search, scope first, then everything else
  6. dive       the top docs' best passages become evidence
  7. check      LLM: does the evidence answer the need? if not, try the next hypothesis

Two layers per export: the metadata graph (<export>~metadata, built by
ingest/build_metadata_graph.py) and a chunk index (the selected dataset, or the
export's default one when the metadata dataset itself is selected).

Every step is an event on the run, so the Agent page can show the loop live:

  POST /api/agent/run          start a run ({dataset, turns | query}); wait=true blocks
  GET  /api/agent/runs         recent runs, newest first (any source: UI, search, bench)
  GET  /api/agent/runs/{id}    one run's events (?after=<i> for new ones) and result
  GET  /api/agent/runs/{id}/graph  the run as a traversal graph, with per-step roles

A run whose question is in a gold set carries the expected docs and whether its
top 5 reached them, so real traffic and replays can be monitored at a glance.

graph_search exposes it as the "agent" technique, so the Explorer, Compare and
the benchmark run it like any other retrieval.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import datasets as DS
from retrieval import bm25_provider as bp

router = APIRouter(prefix="/api/agent", tags=["Agent (routing loop)"])

MAX_RUNS = 60          # runs kept in memory
CONVO_TURNS = 6
CONVO_CHARS = 4000
DEEP = 200             # chunk rows fetched for scoped search (the export has ~700 chunks)
SCOPE_CAP = 25         # records returned by a records answer
CARD_OK = 0.55         # cosine of the best record card that counts as "the cards back this up"
EVIDENCE_DOCS = 3
EXCERPT_CHARS = 700

ANSWER_TYPES = ("records", "document", "detail")
TARGETS = ("chat", "file", "agent", "step", "tool", "user", "any")
PREFIX = {"chat": "chat-", "file": "file-", "agent": "agent-", "step": "step-", "tool": "tool-", "user": "person-"}


# ───────────────────────── the two layers ─────────────────────────

_LAYERS: Dict[str, dict] = {}     # graphml path -> loaded layer (reloaded when the file changes)
_TEXTS: Dict[str, dict] = {}      # docs dir -> {doc: lowercased full text}


def _layers(dataset_id: Optional[str]) -> dict:
    import networkx as nx
    ds = DS.get(dataset_id)
    base = ds.id.split("~")[0]
    gds = DS.get(f"{base}~metadata")
    if gds.id != f"{base}~metadata":
        raise HTTPException(400, f"'{base}' has no metadata graph; run ingest/build_metadata_graph.py on it")
    cds = DS.get(base) if ds.storage.name == "storage-metadata" else ds
    f = gds.storage / "graph_chunk_entity_relation.graphml"
    key, mtime = str(f), f.stat().st_mtime
    L = _LAYERS.get(key)
    if not L or L["mtime"] != mtime:
        G = nx.read_graphml(f)
        full = json.loads((gds.storage / "kv_store_full_docs.json").read_text(encoding="utf-8"))
        doc_nodes = defaultdict(list)
        for n, d in G.nodes(data=True):
            if d.get("file_path"):
                doc_nodes[d["file_path"]].append(n)
        L = _LAYERS[key] = {"mtime": mtime, "G": G, "doc_nodes": dict(doc_nodes),
                            "cards": {v["file_path"]: v["content"] for v in full.values()}}
    texts = _TEXTS.get(str(cds.docs_dir))
    if texts is None and cds.docs_dir and cds.docs_dir.is_dir():
        texts = _TEXTS[str(cds.docs_dir)] = {p.name: p.read_text(encoding="utf-8", errors="replace").lower()
                                             for p in cds.docs_dir.glob("*.md")}
    # duplicate uploads are indexed once, under the original: answer with the original
    return {**L, "graph_ds": gds, "chunk_ds": cds, "texts": texts or {}, "alias": DS.duplicate_aliases(gds)}


def _title(L: dict, doc: str) -> str:
    card = L["cards"].get(doc, "")
    return card.split("\n", 1)[0].lstrip("# ").strip() or doc


def _vocabulary(L: dict) -> str:
    """The graph's record names, so the LLM maps the conversation onto real records."""
    G, by = L["G"], defaultdict(list)
    for n, d in G.nodes(data=True):
        by[d.get("entity_type")].append(n)
    short = lambda n: n.split(": ", 1)[-1].rsplit(" [", 1)[0]
    lines = [f"users: {', '.join(sorted(by['User']))}"]
    if by["Team"]:
        lines.append(f"teams: {', '.join(short(t) for t in sorted(by['Team']))}")
    for t, label in (("Agent", "agents"), ("Skill", "skills"), ("Step", "workflow steps"), ("Tool", "tools")):
        if by[t]:
            lines.append(f"{label}: {', '.join(sorted(short(x) for x in by[t]))}")
    files = sorted({short(f) for f in by["File"]})
    lines.append(f"files: {', '.join(files[:40])}{' …' if len(files) > 40 else ''}")
    topics = sorted(by["Topic"], key=lambda n: -G.degree(n))[:30]
    lines.append(f"chat topics: {', '.join(short(t) for t in topics)}")
    return "\n".join(lines)


# ───────────────────────── runs + events ─────────────────────────

_RUNS: "OrderedDict[str, AgentRun]" = OrderedDict()
_IDS = itertools.count(1)


class AgentRun:
    def __init__(self, dataset: str, turns: List[dict], source: str):
        self.id = f"a{next(_IDS)}"
        self.at = time.time()
        self.t0 = time.perf_counter()
        self.dataset, self.turns, self.source = dataset, turns, source
        self.status = "running"            # running | done | waiting | error
        self.events: List[dict] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self._last_ms = 0
        _RUNS[self.id] = self
        while len(_RUNS) > MAX_RUNS:
            _RUNS.popitem(last=False)

    def emit(self, kind: str, title: str, detail: str = "", **data) -> dict:
        ms = round((time.perf_counter() - self.t0) * 1000)
        ev = {"i": len(self.events) + 1, "ms": ms, "took_ms": ms - self._last_ms,
              "kind": kind, "title": title, "detail": detail, "data": data}
        self._last_ms = ms
        self.events.append(ev)
        return ev

    @property
    def query(self) -> str:
        return next((t["content"] for t in reversed(self.turns) if t["role"] == "user"), "")

    def summary(self) -> dict:
        goal = next((e["data"].get("goal") for e in self.events if e["kind"] == "goal"), None)
        route = next((e["data"].get("route") for e in reversed(self.events) if e["kind"] == "route"), None)
        return {"id": self.id, "at": self.at, "source": self.source, "dataset": self.dataset,
                "status": self.status, "query": self.query[:300], "goal": goal, "route": route,
                "events": len(self.events), "ms": self.events[-1]["ms"] if self.events else 0,
                "gold": _gold_info(self)}


# ───────────────────────── LLM steps ─────────────────────────

async def _llm_json(prompt: str, system: str) -> dict:
    from retrieval import lightrag_provider as lp
    out = await lp.llm(prompt, system_prompt=system)
    m = re.search(r"\{.*\}", out or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except Exception:
        v = {}
    return v if isinstance(v, dict) else {}


_HYP_FORMAT = (
    '{"claim": "<where the answer lives, at most 15 words>", '
    '"answer_type": "records" | "document" | "detail", '
    '"target": "chat" | "file" | "agent" | "step" | "tool" | "user" | "any", '
    '"scope": {"users": [], "teams": [], "agents": [], "files": [], "chats": [], "terms": []}, '
    '"query": "<search query, at most 15 words>"}'
)

_GOAL_PROMPT = """You route questions for a team's internal knowledge base. Before anything is searched, work out what the user is actually after.

The knowledge base is a graph of records, each linking to its full source:
users (pseudonyms user-NN), teams, chats (past conversations), files (uploads), Agent Hub agents with their skills, workflow steps and tools, and chat topics.
KNOWN RECORDS:
{vocab}

CONVERSATION (latest last):
{convo}

Reply with JSON only:
{{"goal": "<one sentence: what the user is trying to accomplish>",
 "need": "<what they need from the knowledge base right now>",
 "clear": true | false,
 "missing": "<if not clear: what is still unknown>",
 "hypotheses": [{hyp}, ...]}}

Rules:
- clear is about whether you know what the user is trying to do, NOT whether you have everything needed to do it (missing attachments or code don't make it unclear). clear is false only for greetings, tests and openers that don't say what they need yet. Do not invent a goal.
- answer_type: "records" when the answer is a set, list or count of records built from who/what/where metadata (someone's chats, the files in a chat, who used an agent, which agents have a feature); "document" when they need the particular chat, file or agent that covers something; "detail" when they need a specific fact, code, setting or step written inside a document.
- target: the kind of record the answer consists of ("user" when the answer is people).
- Chats cover whatever users asked about (study help, writing, trivia, code, security work), not only Agent Hub. Don't assume the answer involves agents or files unless the user says so.
- query: reuse the user's own distinctive words; don't replace them with generic ones.
- scope: chats are chat titles the user quoted. ONLY names the user actually wrote in the conversation, spelled as in KNOWN RECORDS where they match. Never add an agent, file or user the user did not mention. terms are distinctive words the right records must contain (product, tool or file names, IDs, error text) -- not generic words.
- Give 1 or 2 hypotheses that differ in where they look, most likely first. Keep every field short."""

_REVISE_PROMPT = """You are routing a question for a team's knowledge base.
GOAL: {goal}
NEED: {need}

Your hypotheses about where the answer lives were checked against the knowledge base:
{evidence}

KNOWN RECORDS:
{vocab}

None of them held up. Revise them using the evidence: drop names that don't exist, widen or narrow the scope, or change the answer type.
Reply with JSON only: {{"note": "<what you changed and why, one sentence>", "hypotheses": [{hyp}, ...]}}"""

_CHECK_PROMPT = """GOAL: {goal}
NEED: {need}
HYPOTHESIS: {claim}

Excerpts retrieved from the knowledge base:
{excerpts}

Does this evidence give the user what they need? Reply with JSON only:
{{"verdict": "answers" | "partial" | "no", "best": <number of the most useful excerpt, 0 if none>, "why": "<one sentence>"}}"""


def _convo(turns: List[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['content'].strip()}" for t in turns[-CONVO_TURNS:])[-CONVO_CHARS:]


def _clean_hyps(raw) -> List[dict]:
    out = []
    for h in raw if isinstance(raw, list) else []:
        if not isinstance(h, dict):
            continue
        sc = h.get("scope") if isinstance(h.get("scope"), dict) else {}
        scope = {k: [str(x).strip() for x in (sc.get(k) or []) if str(x).strip()][:6]
                 for k in ("users", "teams", "agents", "files", "chats", "terms")}
        out.append({"claim": str(h.get("claim") or "").strip()[:300],
                    "answer_type": h.get("answer_type") if h.get("answer_type") in ANSWER_TYPES else "document",
                    "target": h.get("target") if h.get("target") in TARGETS else "any",
                    "scope": scope, "query": str(h.get("query") or "").strip()[:200]})
    return out[:2]


# ───────────────────────── validation against the graph ─────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def scope_terms(scope: dict) -> List[str]:
    return list(scope.get("_terms") or [])


def _resolve(L: dict, scope: dict) -> tuple:
    """Map scope names onto graph nodes / docs. -> (resolved, unresolved names)."""
    G = L["G"]
    T = lambda n: G.nodes[n].get("entity_type")
    short = lambda n: n.split(": ", 1)[-1].rsplit(" [", 1)[0]
    res = {"users": [], "teams": [], "agents": [], "files": [], "chats": [], "topics": [], "terms": []}
    missing = []
    scope = {**scope, "files": list(scope["files"]), "agents": list(scope["agents"]), "terms": []}
    named = {_norm(short(n)): n for n in G if T(n) in ("File", "Agent", "Step", "Tool", "Topic")}
    for term in scope_terms(scope):
        n = named.get(_norm(term))
        if n and T(n) == "File":
            scope["files"].append(term)
        elif n and T(n) == "Topic":
            res["topics"].append(n)
        elif n:
            scope["agents"].append(term)
        else:
            scope["terms"].append(term)
    for u in scope["users"]:
        m = re.search(r"user[\s_-]*0*(\d+)", u.lower())
        nid = f"user-{int(m.group(1)):02d}" if m else None
        (res["users"].append(nid) if nid in G else missing.append(u))
    for t in scope["teams"]:
        hit = [n for n in G if T(n) == "Team" and (_norm(t) in _norm(short(n)) or _norm(short(n)) in _norm(t))]
        (res["teams"].extend(hit) if hit else missing.append(t))
    for a in scope["agents"]:
        want = set(_norm(a).split()) - {"agent", "the"}
        best, score = None, 0.0
        for n in G:
            if T(n) not in ("Agent", "Step", "Tool", "Skill"):
                continue
            have = set(_norm(short(n)).split())
            s = len(want & have) / max(1, len(want | have))
            if s > score:
                best, score = n, s
        (res["agents"].append(best) if best and score >= 0.5 else missing.append(a))
    for f in scope["files"]:
        hit = [n for n in G if T(n) == "File" and (_norm(f) in _norm(short(n)) or _norm(short(n)) in _norm(f))][:6]
        (res["files"].extend(hit) if hit else missing.append(f))
    for c in scope.get("chats") or []:
        hit = [n for n in G if T(n) == "Chat" and len(_norm(c)) >= 4
               and (_norm(c) in _norm(short(n)) or _norm(short(n)) in _norm(c))][:6]
        (res["chats"].extend(hit) if hit else missing.append(c))
    for term in scope["terms"]:
        t = term.lower()
        docs = sorted(d for d, txt in L["texts"].items() if t in txt)
        (res["terms"].append((term, docs)) if docs else missing.append(term))
    return res, missing


def _near(L: dict, n: str, hops: int = 1) -> set:
    """Docs of a node and of everything within `hops` of it."""
    G, seen, frontier = L["G"], {n}, {n}
    for _ in range(hops):
        frontier = {m for x in frontier for m in G.neighbors(x)} - seen
        seen |= frontier
    return {G.nodes[x].get("file_path") for x in seen} - {"", None}


def _grounded(name: str, convo: str) -> bool:
    """A scope name counts only if the user actually said it (most of its words)."""
    n, c = _norm(name), _norm(convo)
    if not n:
        return False
    if n in c:
        return True
    words = [w for w in n.split() if len(w) >= 3 and w not in ("agent", "the", "file", "chat")]
    return bool(words) and sum(w in c.split() for w in words) / len(words) >= 0.5


_TARGET_WORDS = (
    ("user", re.compile(r"^\s*(who|whose)\b|\bwhich (users|people|person)\b", re.I)),
    ("chat", re.compile(r"\b(chats?|conversations?|discussions?)\b", re.I)),
    ("file", re.compile(r"\b(files?|uploads?|uploaded|attachments?|attached)\b", re.I)),
    ("agent", re.compile(r"\bwhich agents\b|\bwhat agents\b", re.I)),
)


def _question_target(q: str) -> Optional[str]:
    """What kind of record the question asks for, from its wording: "who" -> users,
    otherwise the first record noun ("what FILES were attached in the chat" -> files).
    None for counts and when the wording doesn't say. Overrides the model's guess."""
    q = q or ""
    if re.match(r"\s*how many\b", q, re.I):
        return None   # a count: the answer is a number, whatever record holds it
    hits = [(m.start(), target) for target, rx in _TARGET_WORDS for m in [rx.search(q)] if m]
    return min(hits)[1] if hits else None


def _scope_docs(L: dict, res: dict, target: str, use_terms: bool) -> Optional[set]:
    docs = _scope_docs_raw(L, res, target, use_terms)
    return None if docs is None else {L["alias"].get(d, d) for d in docs}


def _scope_docs_raw(L: dict, res: dict, target: str, use_terms: bool) -> Optional[set]:
    """Docs the hypothesis points at: union within a group, intersection across
    groups. Named records (users, teams, agents, files) scope hard; terms only
    define the record set of a records answer that names nothing else."""
    G = L["G"]
    groups = []
    if res["users"]:
        groups.append(set().union(*(_near(L, u) for u in res["users"])))
    if res["teams"]:
        groups.append(set().union(*(_near(L, t, 2) for t in res["teams"])))
    if res["agents"]:
        groups.append(set().union(*(_near(L, a) for a in res["agents"])))
    if res["files"]:
        groups.append(set().union(*(_near(L, f) for f in res["files"])))
    if res["chats"]:
        groups.append(set().union(*(_near(L, c) for c in res["chats"])))
    if res["topics"]:
        groups.append(set().union(*(_near(L, t) for t in res["topics"])))
    if not groups and use_terms and res["terms"]:
        groups.append(set().union(*(set(d) for _, d in res["terms"])))
    if not groups:
        return None
    docs = set.intersection(*groups)
    if target == "any":
        return docs
    if target == "user":   # the people behind the scoped records
        people = set()
        for d in docs:
            for n in L["doc_nodes"].get(d, []):
                cands = [n] if G.nodes[n].get("entity_type") == "User" else list(G.neighbors(n))
                people |= {f"person-{m}.md" for m in cands if G.nodes[m].get("entity_type") == "User"}
        return {p for p in people if p in L["cards"]}
    pre = PREFIX[target]
    direct = {d for d in docs if d.startswith(pre)}
    if direct:
        return direct
    # e.g. "agents user-02 used": the agents are neighbours of the scoped chats, not in the scope
    return {m for d in docs for n in L["doc_nodes"].get(d, []) for m in _near(L, n) if m.startswith(pre)}


_SEARCH_CACHE: Dict[tuple, list] = {}


def _cached(kind: str, storage: str, query: str, fn) -> list:
    key = (kind, storage, query)
    if key not in _SEARCH_CACHE:
        if len(_SEARCH_CACHE) > 500:
            _SEARCH_CACHE.clear()
        _SEARCH_CACHE[key] = fn()
    return _SEARCH_CACHE[key]


def _cards(L: dict, query: str, exclude: set, n: int = 100) -> List[dict]:
    sd = str(L["graph_ds"].storage)
    rows = _cached("cards", sd, query, lambda: bp.search_fused(query, n, sd))
    return [r for r in rows if r["id"] not in exclude]


def _chunks(L: dict, query: str, exclude: set) -> List[dict]:
    sd = str(L["chunk_ds"].storage)
    rows = _cached("chunks", sd, query, lambda: bp.search_fused(query, DEEP, sd, depth=DEEP))
    return [r for r in rows if r["id"] not in exclude]


def _content(L: dict, queries: List[str], exclude: set) -> List[dict]:
    """Chunk search for content routes: Reciprocal Rank Fusion of the hypothesis'
    query and the user's own message, so a rewrite can't lose the user's specifics."""
    qs = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    lists = [_chunks(L, q, exclude) for q in qs]
    if len(lists) == 1:
        return lists[0]
    fused: Dict[str, dict] = {}
    for lst in lists:
        for rank, r in enumerate(lst, 1):
            f = fused.setdefault(r["chunk_id"], {**r, "_rrf": 0.0})
            f["_rrf"] += 1.0 / (60 + rank)
    return sorted(fused.values(), key=lambda r: -r["_rrf"])


def _best_chunk(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for r in rows:
        out.setdefault(r["id"], r)
    return out


def _validate(L: dict, h: dict, convo: str, user_q: str, exclude: set, min_agree: int = 1) -> dict:
    """Check a hypothesis against the graph (and, for content, the chunk index)."""
    sc = h["scope"]
    dropped = [x for k in ("users", "teams", "agents", "files", "chats", "terms") for x in sc[k] if not _grounded(x, convo)]
    sc = {k: [x for x in v if _grounded(x, convo)] for k, v in sc.items()}
    h = {**h, "scope": sc}
    res, missing = _resolve(L, {**sc, "_terms": sc["terms"], "terms": []})
    retarget = None
    if h["answer_type"] == "records":
        want = _question_target(user_q)
        if want and want != h["target"]:
            retarget = f"target {h['target']} → {want} (the question asks for {want}s)"
            h = {**h, "target": want}
    named = sum(len(sc[k]) for k in ("users", "teams", "agents", "files", "chats")) + len(res["topics"])
    found = named - sum(1 for k in ("users", "teams", "agents", "files", "chats") for x in sc[k] if x in missing)
    records = h["answer_type"] == "records"
    scope = _scope_docs(L, res, h["target"], use_terms=records)
    if scope is not None:
        scope -= exclude
    query = h["query"] or h["claim"]
    cards = _cards(L, query, exclude)
    note = (f"dropped {', '.join(dropped)} (not in the conversation); " if dropped else "") + \
           (f"{retarget}; " if retarget else "")
    out = {"dropped": dropped, "missing": missing,
           "resolved": {k: ([t for t, _ in v] if k == "terms" else v) for k, v in res.items() if v},
           "retarget": retarget,
           "term_docs": {t: len(d) for t, d in res["terms"]},
           "scope": sorted(scope) if scope is not None else None,
           "cards": [{"doc": c["id"], "title": _title(L, c["id"]), "score": c.get("relevance_score"),
                      "in_scope": bool(scope and c["id"] in scope)} for c in cards[:5]],
           "_cards": cards, "_h": h,
           "named_scope": any(res[k] for k in ("users", "teams", "agents", "files", "chats", "topics"))}
    if named and not found:
        return {**out, "verdict": "rejected", "score": 0.0,
                "why": note + f"none of the named records exist ({', '.join(missing)})"}
    if scope is not None and not scope:
        return {**out, "verdict": "rejected", "score": 0.0, "why": note + "no record matches the whole scope"}
    if records:
        if scope:
            return {**out, "verdict": "validated", "score": 2.0 + found / max(1, named),
                    "why": note + f"{len(scope)} matching record(s) in the graph"
                           + (f" ({found} of {named} named record(s) found)" if named else " (from the terms)")}
        return {**out, "verdict": "weak", "score": 0.5, "why": note + "a records answer needs something to scope by"}
    # document / detail: does the content exist, and does it sit where the hypothesis says?
    rows = _content(L, [query, user_q], exclude)
    top = rows[:5]
    best = max((r.get("relevance_score") or 0.0 for r in top), default=0.0)
    inside = sum(1 for r in top if scope and r["id"] in scope)
    out["passages"] = [{"doc": r["id"], "title": _title(L, r["id"]), "score": r.get("relevance_score"),
                        "in_scope": bool(scope and r["id"] in scope)} for r in top]
    if scope:
        ok = inside > 0
        why = f"{inside} of the top 5 passages are in the {len(scope)} scoped doc(s); best passage {best:.2f}"
    elif h.get("baseline"):
        ok, why = True, f"the user's own words; best passage {best:.2f}"
    else:
        # no records to check it against: does the hypothesis still look where the user's own
        # words look? A rewrite whose top docs share nothing with them has drifted.
        own = list(_best_chunk(_chunks(L, user_q, exclude)))[:5]
        mine = list(_best_chunk(_chunks(L, query, exclude)))[:5]
        agree = len(set(own) & set(mine))
        ok = agree >= min_agree
        why = (f"no named records to scope by; its query agrees with the user's own words on {agree} of the top 5 docs"
               if ok else f"drifted: only {agree} of its top 5 docs are among those the user's own words find "
                          f"(needs {min_agree})")
    return {**out, "verdict": "validated" if ok else "weak",
            "score": round(inside / 5 + best + (0.5 if scope else 0), 3), "why": note + why}


def _execute(run: AgentRun, L: dict, h: dict, v: dict, user_q: str, exclude: set) -> List[dict]:
    """-> ranked result docs [{doc, score, text, chunk_id, via}] for the hypothesis."""
    scope = set(v["scope"]) if v["scope"] is not None else None
    query = h["query"] or h["claim"]
    seeds = {G_doc for k in ("users", "teams", "agents", "files", "chats")
             for n in (v.get("resolved") or {}).get(k, []) for G_doc in [L["G"].nodes[n].get("file_path")] if G_doc}
    hard = h["answer_type"] == "records" and scope and v.get("named_scope") and not scope <= seeds
    if h["answer_type"] == "records" and scope and not hard:
        run.emit("fallback", "Soft scope: searching content, scope first",
                 "the scope comes only from terms" if not v.get("named_scope")
                 else "the scope is just the named record itself", layer="chunks")
    if hard:
        order = {c["id"]: i for i, c in enumerate(v["_cards"])}
        ranked = sorted(scope, key=lambda d: (order.get(d, 10 ** 6), d))[:SCOPE_CAP]
        run.emit("search", f"Graph traversal: {len(scope)} record(s)",
                 "the answer is the scoped record set itself, ranked by how well each record card matches",
                 layer="graph", scoped=True, hits=len(ranked))
        score = {c["id"]: c.get("relevance_score") for c in v["_cards"]}
        return [{"doc": d, "score": score.get(d), "text": L["cards"].get(d, ""),
                 "chunk_id": f"{d[:-3]}-card", "via": "graph"} for d in ranked]
    if h["answer_type"] == "records" and not scope:
        pre = PREFIX.get(h["target"])
        cards = [c for c in v["_cards"] if not pre or c["id"].startswith(pre)] or v["_cards"]
        run.emit("search", f"Record cards: {len(cards)} {h['target']} record(s)", f"cards for “{query}” (no scope)",
                 layer="cards", scoped=False, hits=len(cards))
        return [{"doc": c["id"], "score": c.get("relevance_score"), "text": L["cards"].get(c["id"], ""),
                 "chunk_id": c.get("chunk_id"), "via": "card"} for c in cards[:20]]
    # document / detail (and soft records): chunk search, the scope's docs first, then the rest
    rows = _content(L, [query, user_q], exclude)
    if scope:
        inside = [r for r in rows if r["id"] in scope]
        if inside:
            run.emit("search", f"Scoped search: {len(inside)} passage(s) in {len({r['id'] for r in inside})} scoped doc(s)",
                     f"chunks for “{query}” + the user's words: inside the scope first, then the rest of the index",
                     layer="chunks", scoped=True, hits=len(inside))
            rows = inside + [r for r in rows if r["id"] not in scope]
        else:
            run.emit("fallback", "No passages in the scope: searching everything",
                     f"{len(scope)} scoped doc(s), none with a passage for “{query}”", layer="chunks")
    else:
        run.emit("search", f"Chunk search: {len(rows)} passages",
                 f"chunks for “{query}” + the user's words (no named records to scope by)", layer="chunks", scoped=False, hits=len(rows))
    return [{"doc": d, "score": r.get("relevance_score"), "text": r.get("content") or r.get("description") or "",
             "chunk_id": r.get("chunk_id"), "via": "chunk"} for d, r in _best_chunk(rows).items()][:20]


_DISTINCT = re.compile(r"[A-Za-z0-9_.\-/]*(?:_|\d|[a-z][A-Z])[A-Za-z0-9_.\-/]*|[A-Za-z]{7,}")


def _excerpt(text: str, queries: List[str]) -> str:
    """The passage window around the first distinctive query word it contains."""
    flat = re.sub(r"\s+", " ", text or "")
    if len(flat) <= EXCERPT_CHARS:
        return flat
    low = flat.lower()
    for w in sorted({m.group(0).strip("`'\".,") for q in queries for m in _DISTINCT.finditer(q or "")},
                    key=lambda w: (-("_" in w or any(c.isdigit() for c in w)), -len(w))):
        at = low.find(w.lower()) if len(w) >= 4 else -1
        if at >= 0:
            start = max(0, at - EXCERPT_CHARS // 3)
            return ("…" if start else "") + flat[start:start + EXCERPT_CHARS] + "…"
    return flat[:EXCERPT_CHARS] + "…"


async def _dive_and_check(run: AgentRun, L: dict, goal: dict, h: dict, docs: List[dict], user_q: str,
                          exclude: set) -> dict:
    """Best passage of each top doc becomes evidence; the LLM checks it against the need."""
    if not docs:
        return {"verdict": "no", "best": 0, "why": "nothing retrieved"}
    queries = [h["query"] or h["claim"], user_q]
    passages = _best_chunk(await asyncio.to_thread(_content, L, queries, exclude))   # cached searches
    ev = []
    for d in docs[:EVIDENCE_DOCS]:
        p = passages.get(d["doc"])
        text = (p.get("content") if p else None) or d["text"]
        ev.append({"doc": d["doc"], "title": _title(L, d["doc"]), "excerpt": _excerpt(text, queries)})
    run.emit("evidence", f"Read the top {len(ev)} doc(s)", "best passage of each, from the chunk index",
             evidence=ev)
    excerpts = "\n\n".join(f"[{i}] {e['title']}\n{e['excerpt']}" for i, e in enumerate(ev, 1))
    v = await _llm_json(_CHECK_PROMPT.format(goal=goal["goal"], need=goal["need"], claim=h["claim"], excerpts=excerpts),
                        "You check whether retrieved evidence meets a stated need. Output JSON only.")
    verdict = v.get("verdict") if v.get("verdict") in ("answers", "partial", "no") else "partial"
    try:
        best = int(v.get("best") or 0)
    except Exception:
        best = 0
    out = {"verdict": verdict, "best": best if 1 <= best <= len(ev) else 0, "why": str(v.get("why") or "")[:300]}
    run.emit("check", {"answers": "Evidence answers the need", "partial": "Evidence partly answers the need",
                       "no": "Evidence doesn't answer the need"}[verdict], out["why"],
             verdict=verdict, best=ev[out["best"] - 1]["doc"] if out["best"] else None)
    return out


# ───────────────────────── the loop ─────────────────────────

async def run_agent(run: AgentRun, require_clear: bool = True, exclude: Optional[set] = None) -> dict:
    exclude = set(exclude or ())
    try:
        L = _layers(run.dataset)
        run.emit("context", "Read the conversation",
                 f"{len(run.turns)} turn(s); graph: {L['graph_ds'].id} ({L['G'].number_of_nodes()} records); "
                 f"chunks: {L['chunk_ds'].id}", turns=len(run.turns))
        from retrieval import vector_provider as vp
        if not vp.embed_available():
            run.emit("fallback", "Embedding host unavailable: keyword (BM25) search only",
                     f"{vp.EMBED_HOST} failed or timed out recently; vector results come back once it answers")
        vocab = _vocabulary(L)
        convo = _convo(run.turns)
        user_q = run.query[-300:]

        g = await _llm_json(_GOAL_PROMPT.format(vocab=vocab, convo=convo, hyp=_HYP_FORMAT),
                            "You work out what a user needs before anything is searched. Output JSON only.")
        goal = {"goal": str(g.get("goal") or "").strip(), "need": str(g.get("need") or "").strip(),
                "clear": bool(g.get("clear", True)), "missing": str(g.get("missing") or "").strip()}
        hyps = _clean_hyps(g.get("hypotheses"))
        # The model only proposes; the code decides (small models misjudge their own readiness,
        # as Ping's gate found): a stated goal and need behind a substantive message is clear.
        if not goal["clear"] and goal["goal"] and goal["need"] and hyps and len(user_q.split()) >= 8:
            goal["clear"], goal["overruled"] = True, goal["missing"] or "model said unclear"
        run.emit("goal", goal["goal"] or "(no goal stated)", goal["need"], **goal)
        if (require_clear and not goal["clear"]) or not hyps:
            why = goal["missing"] or ("no hypothesis about where to look" if not hyps else "goal not clear yet")
            run.emit("waiting", "Waiting for the goal to be clear", why)
            run.status = "waiting"
            run.result = {"goal": goal, "route": None, "docs": [], "answer": None, "waiting": why}
            return run.result

        tried, outcome = [], None
        for round_ in (1, 2):
            checked = []
            for i, h in enumerate(hyps, 1):
                run.emit("hypothesis", f"H{len(tried) + i}: {h['claim']}",
                         f"{h['answer_type']} → {h['target']} · query “{h['query']}”", hypothesis=h, n=len(tried) + i)
                v = await asyncio.to_thread(_validate, L, h, convo, user_q, exclude)
                run.emit("validate", f"H{len(tried) + i} {v['verdict']}", v["why"],
                         n=len(tried) + i, **{k: x for k, x in v.items() if not k.startswith("_")})
                checked.append((v["_h"], v))
            good = [hv for hv in checked if hv[1]["verdict"] == "validated"]
            if good or round_ == 2:
                pool = good or sorted((hv for hv in checked if hv[1]["verdict"] != "rejected"),
                                      key=lambda hv: -hv[1]["score"]) or checked
                break
            evidence = "\n".join(
                f"H{i}: {h['claim']} ({h['answer_type']}, target {h['target']}, scope {json.dumps(h['scope'])}) -> "
                f"{v['verdict']}: {v['why']}; top record cards: {'; '.join(c['title'] for c in v['cards'][:3])}"
                for i, (h, v) in enumerate(checked, 1))
            r = await _llm_json(_REVISE_PROMPT.format(goal=goal["goal"], need=goal["need"], evidence=evidence,
                                                      vocab=vocab, hyp=_HYP_FORMAT),
                                "You revise hypotheses from evidence. Output JSON only.")
            tried += [h for h, _ in checked]
            hyps = _clean_hyps(r.get("hypotheses")) or [h for h, _ in checked]
            run.emit("revise", "Revised the hypotheses", str(r.get("note") or "")[:300], count=len(hyps))

        h0 = {"claim": "the user's own words, without a rewrite", "answer_type": "detail", "target": "any",
              "scope": {k: [] for k in ("users", "teams", "agents", "files", "chats", "terms")}, "query": user_q,
              "baseline": True}
        pool = [hv for hv in pool if hv[1]["verdict"] != "rejected"][:2]
        if not pool:
            run.emit("fallback", "Every hypothesis was rejected: using the user's own words",
                     "H0: search what the user wrote, unscoped")
        attempts = pool + [(h0, None)]
        for attempt, (h, v) in enumerate(attempts, 1):
            if v is None:
                v = await asyncio.to_thread(_validate, L, h0, convo, user_q, exclude)
            run.emit("route", f"Route: {h['answer_type']}" + (" (H0: the user's own words)" if h.get("baseline") else ""),
                     {"records": "answer from the graph's records", "document": "find the record that covers it",
                      "detail": "dive into content, scoped by the graph"}[h["answer_type"]]
                     + (f" · scope {len(v['scope'])} doc(s)" if v["scope"] is not None else " · no scope"),
                     route=h["answer_type"], target=h["target"], claim=h["claim"], attempt=attempt)
            docs = await asyncio.to_thread(_execute, run, L, h, v, user_q, exclude)
            if docs and docs[0]["via"] == "graph":
                chk = {"verdict": "answers", "best": 0, "why": "exact: the records came from the graph"}
                run.emit("check", f"{len(docs)} record(s) from the graph", chk["why"], verdict="answers")
            else:
                chk = await _dive_and_check(run, L, goal, h, docs, user_q, exclude)
                if chk["best"]:   # the excerpt the checker picked goes first
                    pick = docs[chk["best"] - 1]
                    docs = [pick] + [d for d in docs if d is not pick]
            cand = {"h": h, "v": v, "docs": docs, "check": chk}
            if outcome is None or ("answers", "partial", "no").index(chk["verdict"]) < \
                    ("answers", "partial", "no").index(outcome["check"]["verdict"]):
                outcome = cand
            if chk["verdict"] != "no" or attempt == len(attempts):
                break
            nxt = attempts[attempt][0]
            run.emit("retry", "Trying the user's own words (H0)" if nxt.get("baseline") else "Trying the next hypothesis",
                     "the evidence didn't meet the need")

        h, docs = outcome["h"], outcome["docs"]
        answer = None
        if docs and docs[0]["via"] == "graph":
            answer = f"{len(docs)} record(s): " + "; ".join(_title(L, d["doc"]) for d in docs)
        result = {"goal": goal, "route": h["answer_type"], "hypothesis": h, "check": outcome["check"],
                  "answer": answer,
                  "docs": [{"doc": d["doc"], "title": _title(L, d["doc"]), "score": d["score"], "via": d["via"],
                            "chunk_id": d["chunk_id"], "text": d["text"]} for d in docs]}
        run.emit("result", f"{len(docs)} doc(s) via the {h['answer_type']} route",
                 answer or f"check: {outcome['check']['verdict']}",
                 docs=[{"doc": d["doc"], "title": _title(L, d["doc"])} for d in docs[:8]])
        run.status = "done"
        run.result = result
        return result
    except Exception as e:
        run.status, run.error = "error", f"{type(e).__name__}: {e}"[:300]
        run.emit("error", "Agent failed", run.error)
        run.result = {"goal": None, "route": None, "docs": [], "answer": None, "error": run.error}
        return run.result


def subgraph(dataset: str, result: dict) -> Optional[dict]:
    """Touched subgraph for the Explorer: result records and their neighbours."""
    try:
        L = _layers(dataset)
    except HTTPException:
        return None
    from graph_search import _metadata_subgraph
    return _metadata_subgraph(L["graph_ds"].storage, [d["doc"] for d in result.get("docs", [])[:10]])


# ───────────────────────── monitoring: gold matching + traversal graph ─────────────────────────

_GOLD: Dict[str, tuple] = {}     # export id -> (loaded at, {normalised query: gold entry})
GOLD_TTL_S = 60.0
SCOPE_SHOW = 30                  # scope records drawn per hypothesis


def _gold_entry(dataset_id: str, query: str) -> Optional[dict]:
    """The gold entry whose question is exactly this run's question (any gold set)."""
    base = DS.get(dataset_id).id.split("~")[0]
    hit = _GOLD.get(base)
    if not hit or time.time() - hit[0] > GOLD_TTL_S:
        entries = DS.gold_entries(DS.get(base))
        hit = _GOLD[base] = (time.time(), {_norm(g["query"]): g for g in entries if g.get("gold_ids")})
    return hit[1].get(_norm(query))


def _gold_info(run: "AgentRun") -> Optional[dict]:
    """For a run whose question is in a gold set: the expected docs, and (once it has
    finished) whether the top 5 results reached any of them -- the monitoring signal."""
    try:
        g = _gold_entry(run.dataset, run.query)
    except Exception:
        return None
    if not g:
        return None
    got = [d["doc"] for d in (run.result or {}).get("docs", [])][:5]
    done = run.status != "running"
    reached = [x for x in g["gold_ids"] if x in got]
    return {"id": g.get("id"), "type": g.get("query_type") or "real", "gold_ids": g["gold_ids"],
            "hit5": (bool(reached) if done else None), "reached": reached}


_HYP_ROLES = ("pending", "validated", "weak", "rejected", "chosen", "failed")
# strength of a record's role; within a step and across steps the stronger one wins
_ROLE_RANK = {"scope": 1, "hit_out": 2, "term": 3, "missing": 3, "dropped": 3, "hit_in": 4, "seed": 5,
              "evidence": 6, "confirmed": 7, "result": 8}


def _put(roles: dict, n: Optional[str], role: str) -> None:
    if n and _ROLE_RANK.get(role, 0) >= _ROLE_RANK.get(roles.get(n), 0):
        roles[n] = role


def traversal(run: "AgentRun") -> dict:
    """The run as a graph: the question, a node per hypothesis, the records each one
    named (or named wrongly), the scope it marked out, which hits fell inside it,
    what was read and checked, and the result -- with each step's role changes, so
    the page can replay the traversal step by step. Real records keep their graph
    edges; the agent's reasoning is drawn as virtual nodes and edges on top."""
    L = _layers(run.dataset)
    G = L["G"]
    nodes: Dict[str, dict] = {}
    edges: Dict[tuple, dict] = {}
    short = lambda n: n.split(": ", 1)[-1].rsplit(" [", 1)[0]

    def real(nid: Optional[str]) -> Optional[str]:
        if not nid or nid not in G:
            return None
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": short(nid)[:48], "type": G.nodes[nid].get("entity_type", "?"),
                          "doc": G.nodes[nid].get("file_path") or None, "virtual": False}
        return nid

    def doc_node(doc: str) -> Optional[str]:
        doc = L["alias"].get(doc, doc)
        ns = L["doc_nodes"].get(doc) or []
        return real(ns[0]) if ns else None

    def virt(nid: str, label: str, typ: str, **extra) -> str:
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label[:60], "type": typ, "doc": None, "virtual": True, **extra}
        return nid

    def edge(a: Optional[str], b: Optional[str], kind: str, label: str = ""):
        if a and b and a != b:
            edges.setdefault((a, b, kind), {"source": a, "target": b, "kind": kind, "label": label})

    steps, claim_h, current = [], {}, None
    for e in run.events:
        k, d, roles = e["kind"], e.get("data") or {}, {}
        if k == "context":
            roles[virt("Q", run.query[:60] or "question", "Question")] = "query"
        elif k == "goal":
            virt("Q", "question", "Question")
            nodes["Q"]["label"] = (d.get("goal") or run.query)[:60]
            roles["Q"] = "goal"
        elif k == "hypothesis":
            hid = f"H{d.get('n')}"
            h = d.get("hypothesis") or {}
            claim_h[h.get("claim")] = hid
            virt(hid, hid, "Hypothesis", claim=h.get("claim"), answer_type=h.get("answer_type"), target=h.get("target"))
            edge("Q", hid, "hypothesis")
            roles[hid] = "pending"
        elif k == "validate":
            hid = f"H{d.get('n')}"
            roles[hid] = d.get("verdict", "weak")
            res = d.get("resolved") or {}
            seeds = [real(x) for key in ("users", "teams", "agents", "files", "chats", "topics") for x in res.get(key, [])]
            for s in filter(None, seeds):
                _put(roles, s, "seed")
                edge(hid, s, "names")
            for name in d.get("missing") or []:
                roles[virt(f"x:{name}", name, "Missing")] = "missing"
                edge(hid, f"x:{name}", "names")
            for name in d.get("dropped") or []:
                roles[virt(f"x:{name}", name, "Dropped")] = "dropped"
                edge(hid, f"x:{name}", "names")
            terms = res.get("terms") or []
            for t in terms:
                roles[virt(f"t:{t}", t, "Term")] = "term"
                edge(hid, f"t:{t}", "names")
            for doc in (d.get("scope") or [])[:SCOPE_SHOW]:
                n = doc_node(doc)
                if not n:
                    continue
                roles.setdefault(n, "scope")
                linked = False
                for s in filter(None, seeds):
                    if G.has_edge(s, n):
                        edge(s, n, "graph", (G.edges[s, n].get("keywords") or "").split(",")[0])
                        linked = True
                for t in terms:
                    if t.lower() in L["texts"].get(L["alias"].get(doc, doc), ""):
                        edge(f"t:{t}", n, "contains")
                        linked = True
                if not linked:
                    edge(hid, n, "scope")
            for c in d.get("passages") or d.get("cards") or []:
                n = doc_node(c["doc"])
                if n:
                    _put(roles, n, "hit_in" if c.get("in_scope") else "hit_out")
                    edge(hid, n, "found")
        elif k == "route":
            if "H0" in e["title"]:
                current = virt("H0", "H0", "Hypothesis", claim="the user's own words, without a rewrite",
                               answer_type="detail", target="any")
                edge("Q", "H0", "hypothesis")
            else:
                current = claim_h.get(d.get("claim"), current)
            if current:
                roles[current] = "chosen"
        elif k == "evidence":
            for x in d.get("evidence") or []:
                n = doc_node(x["doc"])
                if n:
                    _put(roles, n, "evidence")
                    edge(current, n, "read")
        elif k == "check":
            if d.get("best"):
                n = doc_node(d["best"])
                if n:
                    _put(roles, n, "confirmed")
            if d.get("verdict") == "no" and current:
                roles[current] = "failed"
        elif k == "retry" and current:
            roles[current] = "failed"
        elif k == "result":
            for x in (d.get("docs") or [])[:5]:
                n = doc_node(x["doc"])
                if n:
                    _put(roles, n, "result")
                    edge(current, n, "answer")
        steps.append({"i": e["i"], "kind": k, "roles": roles})

    gold = _gold_info(run)
    if gold:
        gold["nodes"] = [n for n in (doc_node(x) for x in gold["gold_ids"]) if n]
    # graph context: real edges among the drawn records (e.g. user -> chat -> file)
    reals = [n for n, v in nodes.items() if not v["virtual"]]
    for a, b, dd in G.subgraph(reals).edges(data=True):
        if not any((a, b, kk) in edges or (b, a, kk) in edges for kk in ("graph",)):
            edge(a, b, "graph", (dd.get("keywords") or "").split(",")[0])
    return {"id": run.id, "status": run.status, "nodes": list(nodes.values()), "edges": list(edges.values()),
            "steps": steps, "gold": gold}


# ───────────────────────── API ─────────────────────────

class Turn(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    dataset: Optional[str] = None
    turns: List[Turn] = []
    query: Optional[str] = None
    source: str = "explorer"
    require_clear: bool = True
    exclude_docs: List[str] = []
    wait: bool = False


def _turns(req: AgentRequest) -> List[dict]:
    turns = [{"role": t.role, "content": t.content} for t in req.turns if t.content.strip()]
    if req.query and req.query.strip():
        turns.append({"role": "user", "content": req.query.strip()})
    return turns


def _detail(run: AgentRun, after: int = 0) -> dict:
    return {**run.summary(), "turns": run.turns[-CONVO_TURNS:], "error": run.error,
            "events": [e for e in run.events if e["i"] > after],
            "result": run.result if run.status != "running" else None}


@router.post("/run")
async def start(req: AgentRequest):
    turns = _turns(req)
    if not turns:
        raise HTTPException(400, "give a query or at least one turn")
    ds = DS.get(req.dataset)
    run = AgentRun(ds.id, turns, req.source)
    task = run_agent(run, req.require_clear, set(req.exclude_docs))
    if req.wait:
        await task
        return _detail(run)
    asyncio.create_task(task)
    return {"id": run.id}


@router.get("/runs")
def runs(limit: int = Query(30, ge=1, le=MAX_RUNS)):
    return [r.summary() for r in reversed(list(_RUNS.values()))][:limit]


@router.get("/runs/{run_id}/graph")
def run_graph(run_id: str):
    """The run's traversal graph (see traversal()), recomputed from its events so far."""
    run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such agent run (runs are kept in memory and lost on restart)")
    return traversal(run)


@router.get("/runs/{run_id}")
def run_detail(run_id: str, after: int = Query(0, ge=0)):
    run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such agent run (runs are kept in memory and lost on restart)")
    return _detail(run, after)
