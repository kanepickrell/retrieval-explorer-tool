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
_CUR: Dict[str, list] = {"ctx": [], "ents": [], "rels": []}  # per-query capture
# the subgraph LightRAG touched on the last query, for the retrieval graph UI
LAST_GRAPH: Dict[str, list] = {"nodes": [], "edges": []}
# the generated answer from the last query run with generate=True
LAST_ANSWER: str = ""
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


def _install_tap():
    """Wrap _build_context_str to capture the final selected chunk titles."""
    if getattr(op._build_context_str, "_pg_tapped", False):
        return
    orig = op._build_context_str

    async def wrapped(*a, **k):
        res = await orig(*a, **k)
        try:
            _, raw = res
            data = (raw or {}).get("data", {})
            _CUR["ctx"] = [_title_of(c) for c in data.get("chunks", [])]
            _CUR["ents"] = data.get("entities", []) or []
            _CUR["rels"] = data.get("relationships", []) or []
        except Exception:
            pass
        return res
    wrapped._pg_tapped = True
    op._build_context_str = wrapped


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


async def search(query: str, limit: int = 20, mode: str = None, generate: bool = False,
                 storage_dir: str = None, top_k: int = None) -> List[Dict]:
    """Run LightRAG retrieval and return the selected doc titles as unified rows,
    deduped preserving LightRAG's order (rank = order).

    With generate=True, LightRAG also writes a grounded answer (over THIS mode's
    graph+chunk context) into the module global LAST_ANSWER. The context tap fires
    either way, so the touched subgraph (LAST_GRAPH) is captured regardless."""
    global LAST_QUERY_MS
    rag = await _ensure_rag(storage_dir)
    async with _QUERY_LOCK:
        t0 = time.perf_counter()  # timed inside the lock: excludes queue wait behind other queries
        try:
            return await _search_locked(rag, query, limit, mode, generate, top_k)
        finally:
            LAST_QUERY_MS = round((time.perf_counter() - t0) * 1000, 1)


async def _search_locked(rag, query, limit, mode, generate, top_k) -> List[Dict]:
    global LAST_GRAPH, LAST_ANSWER
    _CUR["ctx"], _CUR["ents"], _CUR["rels"] = [], [], []
    LAST_GRAPH = {"nodes": [], "edges": []}
    LAST_ANSWER = ""
    try:
        ans = await rag.aquery(query, param=QueryParam(
            mode=mode or MODE, top_k=top_k or max(20, limit), chunk_top_k=limit,
            max_total_tokens=12000, enable_rerank=False,
            only_need_context=not generate))
        if generate and isinstance(ans, str):
            LAST_ANSWER = ans
    except Exception:
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
    seen, out = set(), []
    for i, title in enumerate(_CUR["ctx"]):
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({"id": title, "label": title, "type": "chunk",
                    "description": "", "relevance_score": round(1.0 - i * 0.01, 4),
                    "source": f"lightrag:{mode or MODE}"})
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
