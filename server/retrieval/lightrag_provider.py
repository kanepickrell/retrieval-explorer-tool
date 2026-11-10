"""
LightRAG retrieval over the corpus (mode=mix by default) -- ported from the
lightrag-explainer query path. Reuses the SAME storage the vector provider reads
(agenthub_storage), so there is nothing to re-index; it just runs LightRAG's
graph-aware retrieval over it and returns the doc titles LightRAG actually
selected, captured via a tap on operate._build_context_str (same technique as
musique_compare.py).

Needs an Ollama host with BOTH the embed model (nomic) and an LLM (gemma3) --
mix/local/global/hybrid do LLM keyword extraction even for context-only queries.

Config (env), shared with vector_provider where sensible:
  PROTOGRAPH_CORPUS_STORAGE  LightRAG storage dir (default: agenthub_storage)
  PROTOGRAPH_EMBED_HOST      Ollama host (default: http://10.128.241.200:11434)
  PROTOGRAPH_EMBED_MODEL     default: nomic-embed-text
  PROTOGRAPH_LLM             default: gemma3:12b
  PROTOGRAPH_LIGHTRAG_MODE   default: mix
"""
from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import lightrag.operate as op

_DEFAULT_STORAGE = r"C:\repos\lightrag-explainer\agenthub_storage"
STORAGE_DIR = os.getenv("PROTOGRAPH_CORPUS_STORAGE", _DEFAULT_STORAGE)
HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
EMBED = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")
LLM = os.getenv("PROTOGRAPH_LLM", "gemma3:12b")
MODE = os.getenv("PROTOGRAPH_LIGHTRAG_MODE", "mix")
NUM_CTX = 16384

_RAGS: Dict[str, "LightRAG"] = {}          # one LightRAG instance per storage dir
_SESSION: Optional[aiohttp.ClientSession] = None
_LOCK: Optional[asyncio.Lock] = None
# LightRAG's context is captured through a module-level tap (_CUR), so queries
# must not overlap or one query's subgraph/answer leaks into another's result.
_QUERY_LOCK: Optional[asyncio.Lock] = None
_CUR: Dict[str, list] = {"ctx": [], "ents": [], "rels": [], "chunks": []}  # per-query capture
# per-query trace of LightRAG's real pipeline stages (timings, keywords, counts)
_TRACE: Dict = {}
# the subgraph LightRAG touched on the last query, for the retrieval graph UI
LAST_GRAPH: Dict[str, list] = {"nodes": [], "edges": []}
# real pipeline steps of the last query: [{label, detail, stat, ms}]
LAST_STEPS: List[Dict] = []
# error from the last query, if LightRAG raised (surfaced instead of "0 results")
LAST_ERROR: Optional[str] = None
# Keyword cache: LightRAG's keyword extraction is one LLM call per query and
# ~95% of graph-mode latency. Keywords depend only on the query text and the LLM
# (not the corpus or mode), so they are reused across modes, datasets and repeat
# runs. Every run records whether its keywords were cached, so cold and cached
# latencies are never mixed silently.
_KW_CACHE: Dict[tuple, tuple] = {}   # (llm, normalized query) -> (hl, ll)
LAST_KEYWORDS_CACHED: bool = False


def _kw_key(query: str) -> tuple:
    return (LLM, " ".join(query.split()).lower())
# the answer prompt LightRAG built for the last query (system prompt incl. the full
# KG + chunk context), and the chunk texts in that context -- stored on the run so
# the answer is generated over exactly what retrieval returned
LAST_PROMPT: Optional[str] = None
LAST_CONTEXT_CHUNKS: List[str] = []
# wall time of the last LightRAG query, excluding time spent waiting for _QUERY_LOCK
LAST_QUERY_MS: float = 0.0


def _title_of(c: dict) -> str:
    t = c.get("file_path")
    if t and t != "unknown_source":
        return t
    return (c.get("content", "") or "").split("\n", 1)[0].strip()


async def _ollama_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += list(history_messages or [])
    msgs.append({"role": "user", "content": prompt})
    body = {"model": LLM, "messages": msgs, "stream": False,
            "options": {"num_ctx": NUM_CTX, "temperature": 0.0}, "keep_alive": "30m"}
    async with _SESSION.post(f"{HOST}/api/chat", json=body,
                             timeout=aiohttp.ClientTimeout(total=900)) as r:
        d = await r.json()
    return d.get("message", {}).get("content", "")


async def _ollama_embed(texts, **kwargs):
    prefix = "search_query: " if kwargs.get("context") == "query" else "search_document: "
    async with _SESSION.post(f"{HOST}/api/embed",
                             json={"model": EMBED, "input": [prefix + t for t in texts],
                                   "keep_alive": "30m"},
                             timeout=aiohttp.ClientTimeout(total=300)) as r:
        d = await r.json()
    return np.array(d["embeddings"], dtype=np.float32)


def _wrap_stage(name: str, record):
    """Wrap one of LightRAG's module-level pipeline functions so each call is
    timed and its real outputs recorded into _TRACE. kg_query resolves these
    through the operate module's globals, so patching the module is enough.
    Safe only because queries are serialized by _QUERY_LOCK."""
    orig = getattr(op, name)
    if getattr(orig, "_pg_tapped", False):
        return

    async def wrapped(*a, **k):
        t = time.perf_counter()
        res = await orig(*a, **k)
        _TRACE.setdefault("ms", {})[name] = round((time.perf_counter() - t) * 1000, 1)
        try:
            record(res, a, k)
        except Exception as e:  # tracing must never break retrieval
            _TRACE.setdefault("trace_errors", []).append(f"{name}: {e!r}"[:120])
        return res
    wrapped._pg_tapped = True
    setattr(op, name, wrapped)


def _rec_keywords(res, a, k):
    hl, ll = res
    _TRACE["hl"], _TRACE["ll"] = list(hl or []), list(ll or [])


def _rec_search(res, a, k):
    _TRACE["raw"] = {"entities": len(res.get("final_entities") or []),
                     "relations": len(res.get("final_relations") or []),
                     "vector_chunks": len(res.get("vector_chunks") or [])}
    # keep the SAME dict object: in local/global/hybrid it is still empty here and
    # LightRAG fills it later, during _merge_all_chunks (`or {}` would drop that)
    tracking = res.get("chunk_tracking")
    _TRACE["tracking"] = tracking if tracking is not None else {}


def _rec_truncate(res, a, k):
    _TRACE["kept"] = {"entities": len(res.get("filtered_entities") or []),
                      "relations": len(res.get("filtered_relations") or [])}


def _rec_merge(res, a, k):
    _TRACE["merged_chunks"] = len(res or [])


def _rec_context(res, a, k):
    _, raw = res
    data = (raw or {}).get("data", {})
    _CUR["chunks"] = data.get("chunks", []) or []
    _CUR["ctx"] = [_title_of(c) for c in _CUR["chunks"]]
    _CUR["ents"] = data.get("entities", []) or []
    _CUR["rels"] = data.get("relationships", []) or []


def _install_tap():
    _wrap_stage("get_keywords_from_query", _rec_keywords)
    _wrap_stage("_perform_kg_search", _rec_search)
    _wrap_stage("_apply_token_truncation", _rec_truncate)
    _wrap_stage("_merge_all_chunks", _rec_merge)
    _wrap_stage("_build_context_str", _rec_context)


_MODE_SEARCH = {
    "local": "entities matched on low-level keywords, then their relations",
    "global": "relations matched on high-level keywords, then their entities",
    "hybrid": "local (entities) + global (relations), merged",
    "mix": "local + global graph search, plus vector search over chunks",
}


def _build_steps(mode: str, top_k: int, chunk_top_k: int, total_ms: float) -> List[Dict]:
    """Turn the trace of one real LightRAG query into UI steps. Every stat and
    timing here was measured on this query; nothing is reconstructed."""
    ms = _TRACE.get("ms", {})
    hl, ll = _TRACE.get("hl", []), _TRACE.get("ll", [])
    raw = _TRACE.get("raw", {})
    kept = _TRACE.get("kept", {})
    tracked = _TRACE.get("tracking", {})
    by_src = {"E": 0, "R": 0, "C": 0}
    for c in _CUR["chunks"]:
        s = (tracked.get(c.get("chunk_id")) or {}).get("source")
        if s in by_src:
            by_src[s] += 1
    steps = [
        {"label": "Keywords (cached)" if LAST_KEYWORDS_CACHED else "Extract keywords (LLM)",
         "detail": f"high-level: {', '.join(hl) or '(none)'} · low-level: {', '.join(ll) or '(none)'}",
         "stat": f"{len(hl)} hl · {len(ll)} ll", "ms": ms.get("get_keywords_from_query")},
        {"label": "Search graph" + (" + vectors" if mode == "mix" else ""),
         "detail": _MODE_SEARCH.get(mode, mode) + f" (top_k {top_k}, chunk_top_k {chunk_top_k})",
         "stat": f"{raw.get('entities', 0)}E · {raw.get('relations', 0)}R" + (f" · {raw.get('vector_chunks', 0)}C" if mode == "mix" else ""),
         "ms": ms.get("_perform_kg_search")},
        {"label": "Truncate to token budget",
         "detail": "drop lowest-ranked entities/relations that exceed the context token limits",
         "stat": f"kept {kept.get('entities', 0)}E · {kept.get('relations', 0)}R",
         "ms": ms.get("_apply_token_truncation")},
        {"label": "Merge chunks",
         "detail": "round-robin merge of vector, entity and relation chunks, deduplicated",
         "stat": f"{_TRACE.get('merged_chunks', 0)} chunks", "ms": ms.get("_merge_all_chunks")},
        {"label": "Build final context",
         "detail": f"chunks by source: {by_src['E']} from entities, {by_src['R']} from relations, {by_src['C']} from vector search",
         "stat": f"{len(_CUR['ents'])}E · {len(_CUR['rels'])}R · {len(_CUR['chunks'])}C",
         "ms": ms.get("_build_context_str")},
    ]
    staged = sum(v for v in ms.values() if v)
    if total_ms - staged > 50:
        steps.append({"label": "Other", "detail": "LightRAG overhead outside the traced stages",
                      "stat": "", "ms": round(total_ms - staged, 1)})
    return steps


async def _ensure_session():
    global _SESSION, _LOCK, _QUERY_LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
        _QUERY_LOCK = asyncio.Lock()
    if _SESSION is None or _SESSION.closed:
        _SESSION = aiohttp.ClientSession()


async def _ensure_rag(storage_dir: str = None):
    await _ensure_session()
    key = str(storage_dir or STORAGE_DIR)
    async with _LOCK:
        if key in _RAGS:
            return _RAGS[key]
        rag = LightRAG(
            working_dir=key,
            llm_model_func=_ollama_llm, llm_model_name=LLM,
            embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=2048,
                                         func=_ollama_embed, model_name=EMBED),
            chunk_token_size=500, chunk_overlap_token_size=50,
            llm_model_max_async=2, enable_llm_cache=False,
        )
        await rag.initialize_storages()
        _install_tap()
        _RAGS[key] = rag
        return rag


def status() -> Dict:
    return {"storage_dir": STORAGE_DIR, "embed_host": HOST, "embed_model": EMBED,
            "llm": LLM, "mode": MODE, "loaded": sorted(_RAGS)}


async def search(query: str, limit: int = 20, mode: str = None,
                 storage_dir: str = None, top_k: int = None,
                 fresh_keywords: bool = False) -> List[Dict]:
    """Run LightRAG retrieval and return the selected doc titles as unified rows,
    deduped preserving LightRAG's order (rank = order).

    Also captures LightRAG's rendered answer prompt (LAST_PROMPT) and the final
    context chunks (LAST_CONTEXT_CHUNKS) so the answer can be generated later over
    exactly this context. The context tap fires
    either way, so the touched subgraph (LAST_GRAPH) is captured regardless."""
    global LAST_QUERY_MS, LAST_STEPS
    rag = await _ensure_rag(storage_dir)
    mode = mode or MODE
    top_k = top_k or max(20, limit)
    async with _QUERY_LOCK:
        t0 = time.perf_counter()  # timed inside the lock: excludes queue wait behind other queries
        try:
            rows = await _search_locked(rag, query, limit, mode, top_k, fresh_keywords)
        finally:
            LAST_QUERY_MS = round((time.perf_counter() - t0) * 1000, 1)
        LAST_STEPS = _build_steps(mode, top_k, limit, LAST_QUERY_MS)
    # real relevance: cosine(query, chunk) with the same embeddings vector mode
    # uses, so scores are comparable across techniques (computed outside the timing)
    if rows:
        from retrieval import vector_provider
        sims = vector_provider.cosine_for([r["chunk_id"] for r in rows], query, storage_dir)
        for r in rows:
            r["relevance_score"] = sims.get(r["chunk_id"])
    return rows


async def _search_locked(rag, query, limit, mode, top_k, fresh_keywords=False) -> List[Dict]:
    global LAST_GRAPH, LAST_PROMPT, LAST_CONTEXT_CHUNKS, LAST_ERROR, LAST_KEYWORDS_CACHED
    _CUR["ctx"], _CUR["ents"], _CUR["rels"], _CUR["chunks"] = [], [], [], []
    _TRACE.clear()
    LAST_GRAPH = {"nodes": [], "edges": []}
    LAST_PROMPT, LAST_CONTEXT_CHUNKS = None, []
    LAST_ERROR = None
    try:
        # only_need_prompt: build the context AND LightRAG's full answer prompt,
        # without calling the LLM -- same cost as only_need_context
        cached = None if fresh_keywords else _KW_CACHE.get(_kw_key(query))
        LAST_KEYWORDS_CACHED = cached is not None
        # passing hl/ll keywords makes LightRAG skip its extraction LLM call
        ans = await rag.aquery(query, param=QueryParam(
            mode=mode, top_k=top_k, chunk_top_k=limit,
            max_total_tokens=12000, enable_rerank=False,
            only_need_prompt=True,
            hl_keywords=list(cached[0]) if cached else [],
            ll_keywords=list(cached[1]) if cached else []))
        if not cached and (_TRACE.get("hl") or _TRACE.get("ll")):
            _KW_CACHE[_kw_key(query)] = (tuple(_TRACE.get("hl", [])), tuple(_TRACE.get("ll", [])))
        prompt = ans if isinstance(ans, str) else getattr(ans, "content", None)
        if prompt:
            # kg_query renders "<system prompt>\n\n---User Query---\n\n<query>"
            LAST_PROMPT = prompt.rsplit("\n\n---User Query---\n\n", 1)[0]
        LAST_CONTEXT_CHUNKS = [c.get("content") or "" for c in _CUR["chunks"]]
    except Exception as e:
        LAST_ERROR = f"lightrag query failed: {e!r}"[:200]
        return []
    # build the touched subgraph for the retrieval graph UI
    nodes = [{"id": e.get("entity_name", ""), "label": e.get("entity_name", ""),
              "type": e.get("entity_type", "UNKNOWN"),
              "file_path": e.get("file_path", "")}
             for e in _CUR["ents"] if e.get("entity_name")]
    nid = {n["id"] for n in nodes}
    edges = [{"source": r.get("src_id", ""), "target": r.get("tgt_id", ""),
              "label": (r.get("keywords") or r.get("description") or "")[:40]}
             for r in _CUR["rels"]
             if r.get("src_id") in nid and r.get("tgt_id") in nid]
    LAST_GRAPH = {"nodes": nodes, "edges": edges}
    # one row per chunk in LightRAG's final context, in its order (recall@k counts
    # distinct docs among them)
    tracked = _TRACE.get("tracking", {})
    out = []
    for c in _CUR["chunks"]:
        title = _title_of(c)
        if not title:
            continue
        cid = c.get("chunk_id", "")
        t = tracked.get(cid) or {}
        out.append({"id": title, "label": title, "type": "chunk", "chunk_id": cid,
                    "description": (c.get("content") or "")[:200],
                    "rank": len(out) + 1,
                    # E = reached via an entity, R = via a relation, C = vector search
                    "via": t.get("source"),
                    "source": f"lightrag:{mode}"})
        if len(out) >= limit:
            break
    return out


_GEN_SYS = (
    "You are a precise assistant answering questions about a team's chat sessions "
    "and documents. Answer ONLY from the provided context. If the context does not "
    "contain the answer, say so plainly. Be concise and cite the source titles you used."
)


async def llm(prompt: str, system_prompt: str = None) -> str:
    """Public one-shot LLM call on the same model the graph modes use. For the
    metrics judge, so Ragas-style scores use the same gemma3 as generation."""
    await _ensure_session()
    return await _ollama_llm(prompt, system_prompt=system_prompt)


async def embed(texts, as_query: bool = False):
    """Public embedding call (nomic). Returns an (n, dim) float32 np.array."""
    await _ensure_session()
    return await _ollama_embed(list(texts), context="query" if as_query else "document")


async def generate_from_prompt(query: str, system_prompt: str) -> str:
    """Answer exactly as LightRAG's kg_query would: its rendered system prompt
    (template + this run's context) and the raw query as the user message."""
    await _ensure_session()
    return await _ollama_llm(query, system_prompt=system_prompt)


async def generate_over(query: str, context_text: str) -> str:
    """Naive-RAG generation: answer a query grounded in the given context text,
    using the SAME LLM the graph modes use (so answers are comparable across
    techniques). Used for the `vector` strategy, which has no graph context."""
    await _ensure_session()
    if not context_text.strip():
        return ""
    prompt = (f"Context:\n{context_text}\n\n"
              f"Question: {query}\n\n"
              f"Answer from the context above:")
    try:
        return await _ollama_llm(prompt, system_prompt=_GEN_SYS)
    except Exception:
        return ""


async def aclose():
    global _SESSION
    if _SESSION is not None:
        await _SESSION.close()
        _SESSION = None
