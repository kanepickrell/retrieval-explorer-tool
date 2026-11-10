"""
Naive dense (vector) retrieval over a document corpus -- ported from the
lightrag-explainer `naive_recall.py` baseline, the green-lit winner for focused
queries. Reads a LightRAG storage dir's chunk vector DB (vdb_chunks.json): the
chunk embeddings are already built and L2-normalized, so there is nothing to
re-embed on the corpus side and NO GraphDB involved.

Only dependency at query time is an Ollama embedding host (to embed the query
with the same nomic model + "search_query:" prefix the index was built with).

Config (env):
  PROTOGRAPH_CORPUS_STORAGE  dir holding vdb_chunks.json
                             (default: the agenthub_storage we built)
  PROTOGRAPH_EMBED_HOST      Ollama host for query embedding
                             (default: http://10.128.241.200:11434 -- built the index)
  PROTOGRAPH_EMBED_MODEL     default: nomic-embed-text
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import httpx

# default to the Agent Hub corpus storage we already built in lightrag-explainer
_DEFAULT_STORAGE = r"C:\repos\lightrag-explainer\agenthub_storage"
STORAGE_DIR = os.getenv("PROTOGRAPH_CORPUS_STORAGE", _DEFAULT_STORAGE)
EMBED_HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
EMBED_MODEL = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")

# lazily-loaded corpus indexes, one per storage dir
# {storage_dir: {"ids":[chunk_id], "titles":[file_path], "contents":[...], "mat":[[float]]}}
_INDEX: Dict[str, Dict] = {}


def _load_index(storage_dir: str = None) -> Dict:
    """Load + decode vdb_chunks.json once. Matrix rows are already normalized."""
    key = str(storage_dir or STORAGE_DIR)
    if key in _INDEX:
        return _INDEX[key]
    sd = Path(key)
    f = sd / "vdb_chunks.json"
    if not f.exists():
        return {"ids": [], "titles": [], "contents": [], "mat": [], "dim": 0,
                  "error": f"vdb_chunks.json not found in {sd}"}
    d = json.loads(f.read_text(encoding="utf-8"))
    data = d["data"]
    dim = int(d["embedding_dim"])
    raw = base64.b64decode(d["matrix"])
    # float32 little-endian, row-major [n, dim], aligned with data order
    import struct
    flat = struct.unpack(f"<{len(raw)//4}f", raw)
    mat = [list(flat[i * dim:(i + 1) * dim]) for i in range(len(data))]
    _INDEX[key] = {
        "ids": [r.get("__id__", "") for r in data],
        "titles": [r.get("file_path", r.get("full_doc_id", "")) for r in data],
        "contents": [r.get("content", "") for r in data],
        "mat": mat, "dim": dim, "error": None,
    }
    return _INDEX[key]


def _embed_query(text: str) -> Optional[List[float]]:
    """Embed the query with the same model + 'search_query:' prefix as the index.
    Returns an L2-normalized vector, or None on any failure (graceful degrade)."""
    try:
        res = httpx.post(f"{EMBED_HOST}/api/embed",
                         json={"model": EMBED_MODEL, "input": ["search_query: " + text]},
                         timeout=30.0)
        if res.status_code != 200:
            return None
        vec = res.json()["embeddings"][0]
    except Exception:
        return None
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return None
    return [x / norm for x in vec]


def status() -> Dict:
    idx = _load_index()
    return {"storage_dir": STORAGE_DIR, "embed_host": EMBED_HOST,
            "embed_model": EMBED_MODEL, "n_chunks": len(idx["ids"]),
            "dim": idx["dim"], "error": idx["error"]}


def cosine_for(chunk_ids: List[str], query: str, storage_dir: str = None) -> Dict[str, float]:
    """cosine(query, chunk) for the given chunk ids, using the same query
    embedding as search(). Lets graph modes report a real relevance score that
    is directly comparable with vector mode's. Missing ids are omitted."""
    idx = _load_index(storage_dir)
    if idx["error"] or not idx["mat"]:
        return {}
    qv = _embed_query(query)
    if qv is None:
        return {}
    row = idx.setdefault("row_of", {cid: i for i, cid in enumerate(idx["ids"])})
    out = {}
    for cid in chunk_ids:
        i = row.get(cid)
        if i is not None:
            out[cid] = round(float(sum(a * b for a, b in zip(qv, idx["mat"][i]))), 4)
    return out


# real pipeline steps of the last search(): [{label, detail, stat, ms}]
LAST_STEPS: List[Dict] = []


def search(query: str, limit: int = 20, storage_dir: str = None) -> List[Dict]:
    """Return the top-k CHUNKS as unified rows (id = the chunk's doc, chunk_id =
    the chunk). Several chunks of one doc can appear, so a long doc's answer isn't
    lost to a single "best" chunk; recall@k counts distinct docs among them."""
    global LAST_STEPS
    import time
    LAST_STEPS = []
    t = time.perf_counter()
    idx = _load_index(storage_dir)
    load_ms = round((time.perf_counter() - t) * 1000, 1)
    if idx["error"] or not idx["mat"]:
        return []
    t = time.perf_counter()
    qv = _embed_query(query)
    embed_ms = round((time.perf_counter() - t) * 1000, 1)
    if qv is None:
        return []
    t = time.perf_counter()
    mat, titles, contents, ids = idx["mat"], idx["titles"], idx["contents"], idx["ids"]
    # rows are normalized and qv is normalized -> cosine == dot
    scored = []
    for i, row in enumerate(mat):
        dot = 0.0
        for a, b in zip(qv, row):
            dot += a * b
        scored.append((dot, i))
    scored.sort(reverse=True)
    out = []
    for dot, i in scored[:limit]:
        title = titles[i] or ids[i]
        out.append({"id": title, "label": title, "type": "chunk",
                    "description": (contents[i] or "")[:200],
                    "content": contents[i] or "",
                    "relevance_score": round(float(dot), 4),
                    "source": "vector", "chunk_id": ids[i]})
    score_ms = round((time.perf_counter() - t) * 1000, 1)
    for rank, r in enumerate(out, 1):
        r["rank"], r["via"] = rank, "C"
    LAST_STEPS = ([{"label": "Load chunk index", "detail": "first query on this dataset decodes vdb_chunks.json",
                    "stat": f"{len(ids)} chunks", "ms": load_ms}] if load_ms > 5 else []) + [
        {"label": "Embed query", "detail": f"{EMBED_MODEL} via Ollama, 'search_query:' prefix",
         "stat": f"{idx['dim']}d", "ms": embed_ms},
        {"label": "Cosine search", "detail": "score every chunk, keep the top-k chunks (a doc can contribute several)",
         "stat": f"{len(ids)} chunks → top {len(out)} from {len({r['id'] for r in out})} docs", "ms": score_ms},
    ]
    return out


def contents_for(titles: List[str], char_cap: int = 1800, storage_dir: str = None) -> List[str]:
    """Full (capped) chunk text for given doc titles, preserving input order.
    Lets the metrics judge see the context a graph mode actually retrieved."""
    idx = _load_index(storage_dir)
    by_title: Dict[str, str] = {}
    for t, c in zip(idx["titles"], idx["contents"]):
        if t and t not in by_title:
            by_title[t] = (c or "")[:char_cap]
    return [by_title[t] for t in titles if t in by_title]


def contexts(query: str, limit: int = 5, storage_dir: str = None, char_cap: int = 1800) -> List[Dict]:
    """Top-k hits with FULL (capped) chunk content, for naive-RAG generation.
    Same ranking/doc-level dedup as search(), but returns usable context text."""
    idx = _load_index(storage_dir)
    if idx["error"] or not idx["mat"]:
        return []
    qv = _embed_query(query)
    if qv is None:
        return []
    mat, titles, contents, ids = idx["mat"], idx["titles"], idx["contents"], idx["ids"]
    best: Dict[str, Dict] = {}
    for i, row in enumerate(mat):
        dot = sum(a * b for a, b in zip(qv, row))
        title = titles[i] or ids[i]
        prev = best.get(title)
        if prev is None or dot > prev["relevance_score"]:
            best[title] = {"title": title, "content": (contents[i] or "")[:char_cap],
                           "relevance_score": round(float(dot), 4), "chunk_id": ids[i]}
    out = sorted(best.values(), key=lambda r: r["relevance_score"], reverse=True)
    return out[:limit]
