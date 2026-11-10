"""
Keyword retrieval (Okapi BM25) over the same chunks the vector index holds, and
BM25 + vector fusion by Reciprocal Rank Fusion.

  bm25         lexical: exact terms, IDs, file names. No LLM, no embeddings.
  bm25_vector  RRF over the BM25 and vector doc rankings (score = sum 1/(60 + rank)),
               the hybrid lexical + semantic setup (cf. Anthropic's contextual
               retrieval, which pairs embeddings with BM25)

Corpus = the chunks in the dataset's vdb_chunks.json (via vector_provider), so all
techniques retrieve from identical chunks. Rows are CHUNKS (several may come from
one doc); recall@k counts distinct docs among them. Fusion ranks chunks too, so a
chunk found by both BM25 and vector is the same chunk, not just the same doc.

Tokenizer: lowercase; compound tokens such as "t1558.003", "obap_3.resource",
"c2library.py" are kept whole AND split into their parts, so both an exact
identifier and its pieces can match.

relevance_score on every row is cosine(query, chunk) with the shared embeddings,
so the score column is comparable across all techniques; the BM25 / RRF scores
that actually ranked the rows are returned alongside (bm25_score / rrf_score).
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from retrieval import vector_provider as vp

RRF_K = 60  # standard RRF constant (Cormack et al. 2009)
_COMPOUND = re.compile(r"[a-z0-9]+(?:[._\-/:][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")

# {storage_dir: {"bm25": BM25Okapi, ...}}
_BM25: Dict[str, Dict] = {}
# real pipeline steps of the last search: [{label, detail, stat, ms}]
LAST_STEPS: List[Dict] = []


def tokenize(text: str) -> List[str]:
    out = []
    for m in _COMPOUND.finditer((text or "").lower()):
        tok = m.group(0)
        out.append(tok)
        parts = _PART.findall(tok)
        if len(parts) > 1:
            out.extend(parts)
    return out


def _index(storage_dir: str = None) -> Tuple[Dict, float]:
    """BM25 over the dataset's chunks, built once per dataset. Returns (index, build_ms)."""
    key = str(storage_dir or vp.STORAGE_DIR)
    if key in _BM25:
        return _BM25[key], 0.0
    t = time.perf_counter()
    idx = vp._load_index(storage_dir)
    tokens = [tokenize(c) for c in idx["contents"]]
    _BM25[key] = {"bm25": BM25Okapi(tokens) if tokens else None, "vec": idx}
    return _BM25[key], round((time.perf_counter() - t) * 1000, 1)


def _bm25_chunks(query: str, storage_dir: str = None) -> Tuple[List[Dict], List[Dict]]:
    """All chunks with a term match, ranked by BM25. Returns (rows, steps)."""
    ix, build_ms = _index(storage_dir)
    idx, bm25 = ix["vec"], ix["bm25"]
    steps = []
    if build_ms:
        steps.append({"label": "Build BM25 index", "detail": "first query on this dataset tokenizes every chunk",
                      "stat": f"{len(idx['ids'])} chunks", "ms": build_ms})
    if bm25 is None:
        return [], steps
    t = time.perf_counter()
    q = tokenize(query)
    scores = bm25.get_scores(q)
    rows = []
    for i, s in enumerate(scores):
        if s > 0:
            title = idx["titles"][i] or idx["ids"][i]
            rows.append({"id": title, "label": title, "type": "chunk", "chunk_id": idx["ids"][i],
                         "description": (idx["contents"][i] or "")[:200],
                         "content": idx["contents"][i] or "",
                         "bm25_score": round(float(s), 4)})
    rows.sort(key=lambda r: r["bm25_score"], reverse=True)
    steps.append({"label": "BM25 keyword search",
                  "detail": f"query terms: {', '.join(dict.fromkeys(q)) or '(none)'}",
                  "stat": f"{len(scores)} chunks → {len(rows)} with a term match",
                  "ms": round((time.perf_counter() - t) * 1000, 1)})
    return rows, steps


def _with_cosine(rows: List[Dict], query: str, storage_dir: str) -> None:
    sims = vp.cosine_for([r["chunk_id"] for r in rows], query, storage_dir)
    for r in rows:
        r["relevance_score"] = sims.get(r["chunk_id"])


def search(query: str, limit: int = 20, storage_dir: str = None) -> List[Dict]:
    global LAST_STEPS
    rows, steps = _bm25_chunks(query, storage_dir)
    out = rows[:limit]
    for rank, r in enumerate(out, 1):
        r.update(rank=rank, via="B", source="bm25")
    LAST_STEPS = steps
    _with_cosine(out, query, storage_dir)  # display score only; outside the timed steps
    return out


def search_fused(query: str, limit: int = 20, storage_dir: str = None, depth: int = 50) -> List[Dict]:
    """Reciprocal Rank Fusion of the BM25 and vector CHUNK rankings (each to `depth`)."""
    global LAST_STEPS
    b_rows, steps = _bm25_chunks(query, storage_dir)
    b_rows = b_rows[:depth]
    v_rows = vp.search(query, depth, storage_dir=storage_dir)
    steps += vp.LAST_STEPS
    t = time.perf_counter()
    fused: Dict[str, Dict] = {}
    for src, lst in (("B", b_rows), ("C", v_rows)):
        for rank, r in enumerate(lst, 1):
            f = fused.setdefault(r["chunk_id"], {**r, "rrf_score": 0.0, "via_set": []})
            f["rrf_score"] += 1.0 / (RRF_K + rank)
            f["via_set"].append(src)
    out = sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)[:limit]
    both = sum(1 for r in out if len(r["via_set"]) == 2)
    for rank, r in enumerate(out, 1):
        srcs = set(r.pop("via_set"))
        r.update(rank=rank, via="B+C" if len(srcs) == 2 else srcs.pop(), source="bm25_vector",
                 rrf_score=round(r["rrf_score"], 5))
    steps.append({"label": "Reciprocal rank fusion",
                  "detail": f"score = Σ 1/({RRF_K} + rank) over the BM25 and vector chunk rankings (top {depth} each)",
                  "stat": f"{both} of top {len(out)} found by both",
                  "ms": round((time.perf_counter() - t) * 1000, 1)})
    LAST_STEPS = steps
    _with_cosine(out, query, storage_dir)
    return out
