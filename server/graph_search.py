"""
Corpus-backed unified search router for ProtoGraph.

Retrieval is RAG over the document corpus, NOT SPARQL over the graph: the vector
strategy reads the LightRAG chunk index (see retrieval/vector_provider.py) and
needs only an Ollama embed host -- GraphDB does not have to be up. SPARQL/graph
strategies (keyword, neural) are PARKED for now and return a clear note.

Mounted by main.py at /api/search/unified. Default strategy is `vector` (the main
retrieval now); force others with ?force_strategy=.

Strategies:
  vector  -> naive dense retrieval over corpus chunks (green-lit for focused queries). ACTIVE.
  lightrag-> LightRAG over the corpus (breadth). NOT YET PORTED (next slice).
  keyword -> SPARQL regex over the graph. PARKED (needs GraphDB).
  neural  -> graph-routing subsystem. PARKED (needs GraphDB + assembly).
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Query

from unified_search import analyze_query_intent  # backend-agnostic intent signals
from retrieval import vector_provider
import datasets as DS

# kept for main.py back-compat; retrieval no longer depends on it
_gdb = None
PARKED = {"keyword", "neural"}           # need GraphDB/SPARQL -- parked for now
NOT_PORTED: set = set()                  # (lightrag now ported)
LIGHTRAG_ALIASES = {"lightrag", "mix", "local", "global", "hybrid"}


def init_graph_search(gdb=None):
    """main.py calls this at startup. gdb is optional and currently unused --
    retrieval is corpus-backed, so search runs even with GraphDB down."""
    global _gdb
    _gdb = gdb


def run_vector(query: str, limit: int, storage_dir: str = None) -> List[Dict]:
    return vector_provider.search(query, limit, storage_dir=storage_dir)


router = APIRouter(prefix="/api/search", tags=["Unified Search (corpus RAG)"])


@router.get("/unified")
async def unified_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, le=100),
    force_strategy: Optional[str] = Query(
        None, description="vector (active) | lightrag/keyword/neural (parked)"),
    dataset: Optional[str] = Query(None, description="dataset id from GET /datasets"),
    top_k: Optional[int] = Query(None, le=200, description="LightRAG entity/relation top_k"),
):
    t0 = time.time()
    ds = DS.get(dataset)
    sd = str(ds.storage)
    intent = analyze_query_intent(q)
    # vector is the main retrieval now; routing signals are advisory only
    strategy = (force_strategy or "vector").lower()

    note, results, graph, lr_ms, steps = None, [], None, None, []
    if strategy == "vector":
        results = run_vector(q, limit, sd)
        steps = vector_provider.LAST_STEPS
        if not results:
            note = ("no vector results -- check the embed host "
                    f"({vector_provider.EMBED_HOST}) and corpus storage")
    elif strategy in LIGHTRAG_ALIASES:
        # lazy import so a missing LightRAG install never breaks the vector path
        try:
            from retrieval import lightrag_provider
            lr_mode = "mix" if strategy == "lightrag" else strategy
            results = await lightrag_provider.search(q, limit, mode=lr_mode, storage_dir=sd, top_k=top_k)
            graph = lightrag_provider.LAST_GRAPH  # subgraph for the retrieval graph UI
            lr_ms = lightrag_provider.LAST_QUERY_MS
            steps = lightrag_provider.LAST_STEPS
            if not results:
                note = lightrag_provider.LAST_ERROR or "lightrag returned no context for this query"
        except Exception as e:
            note = f"lightrag unavailable: {e!r}"[:160]
    elif strategy in PARKED:
        note = f"'{strategy}' is parked (needs GraphDB/SPARQL, on hold)"
    else:
        note = f"unknown strategy '{strategy}', use vector"

    return {
        "query": q,
        "dataset": ds.id,
        "strategy_used": strategy if (results or strategy == "vector") else "none",
        "intent_analysis": {
            "recommended_strategy": intent.strategy.value,
            "confidence": intent.confidence,
            "signals": intent.signals,
            "search_terms": intent.search_terms,
        },
        "total_results": len(results),
        # retrieval time only: for LightRAG modes this excludes waiting behind other queries
        "time_ms": lr_ms if lr_ms is not None else round((time.time() - t0) * 1000, 1),
        "note": note,
        "graph": graph,  # {nodes, edges} for lightrag strategies, else null
        "steps": steps,  # measured pipeline stages: [{label, detail, stat, ms}]
        "results": results,
    }


@router.get("/answer")
async def answer(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(5, le=20, description="chunks/context used for grounding"),
    force_strategy: Optional[str] = Query("vector"),
    dataset: Optional[str] = Query(None, description="dataset id from GET /datasets"),
    top_k: Optional[int] = Query(None, le=200),
):
    """Generate a grounded answer for a query under one strategy, using the SAME
    LLM for every mode so only the retrieval differs. Kept separate from /unified
    so the graph + chunks render instantly while this (slower) call streams in.
      vector  -> naive RAG: generate over the dense-retrieved chunk texts.
      lightrag-> LightRAG generates over that mode's graph+chunk context.
    """
    t0 = time.time()
    strategy = (force_strategy or "vector").lower()
    sd = str(DS.get(dataset).storage)
    answer_text, note, used = "", None, 0
    try:
        if strategy == "vector":
            ctx = vector_provider.contexts(q, limit, storage_dir=sd)
            used = len(ctx)
            if not ctx:
                note = "no vector context to ground an answer"
            else:
                from retrieval import lightrag_provider
                blob = "\n\n".join(f"[{c['title']}]\n{c['content']}" for c in ctx)
                answer_text = await lightrag_provider.generate_over(q, blob)
        elif strategy in LIGHTRAG_ALIASES:
            from retrieval import lightrag_provider
            lr_mode = "mix" if strategy == "lightrag" else strategy
            rows = await lightrag_provider.search(q, limit, mode=lr_mode, generate=True, storage_dir=sd, top_k=top_k)
            used = len(rows)
            answer_text = lightrag_provider.LAST_ANSWER
            if not answer_text:
                note = "lightrag returned no answer -- check LLM host"
        elif strategy in PARKED:
            note = f"'{strategy}' is parked (needs GraphDB/SPARQL)"
        else:
            note = f"unknown strategy '{strategy}'"
    except Exception as e:
        note = f"generation failed: {e!r}"[:160]

    return {
        "query": q,
        "strategy": strategy,
        "answer": answer_text,
        "grounded_chunks": used,
        "time_ms": round((time.time() - t0) * 1000, 1),
        "note": note,
    }


@router.get("/metrics")
async def evaluate(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(5, le=20, description="chunks used for retrieval + grounding"),
    force_strategy: Optional[str] = Query("vector"),
    dataset: Optional[str] = Query(None, description="dataset id from GET /datasets"),
    top_k: Optional[int] = Query(None, le=200),
):
    """Compute real Ragas-style metrics for one query under one strategy. Heavy
    (retrieval + generation + several judge calls), so the UI calls it on demand.
    Reference-free metrics (faithfulness, answer relevancy, context precision) are
    computed for any query; reference-based ones need gold and otherwise report
    pending (value null)."""
    import time as _t
    import metrics as M
    from retrieval import lightrag_provider

    t0 = _t.time()
    strategy = (force_strategy or "vector").lower()
    ds = DS.get(dataset)
    sd = str(ds.storage)
    note = None
    answer_text, contexts, retrieved_ids = "", [], []

    try:
        if strategy == "vector":
            ctx = vector_provider.contexts(q, limit, storage_dir=sd)
            retrieved_ids = [c["title"] for c in ctx]
            contexts = [c["content"] for c in ctx]
            blob = "\n\n".join(f"[{c['title']}]\n{c['content']}" for c in ctx)
            answer_text = await lightrag_provider.generate_over(q, blob) if ctx else ""
        elif strategy in LIGHTRAG_ALIASES:
            lr_mode = "mix" if strategy == "lightrag" else strategy
            rows = await lightrag_provider.search(q, limit, mode=lr_mode, generate=True, storage_dir=sd, top_k=top_k)
            retrieved_ids = [r["id"] for r in rows]
            answer_text = lightrag_provider.LAST_ANSWER
            contexts = vector_provider.contents_for(retrieved_ids, storage_dir=sd)
        else:
            note = f"'{strategy}' is not evaluable (parked)"
    except Exception as e:
        note = f"evaluation setup failed: {e!r}"[:160]

    # reference-free metrics (any query)
    ref_free = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}
    if answer_text and contexts:
        try:
            ref_free = await M.reference_free(q, answer_text, contexts)
        except Exception as e:
            note = (note or "") + f" | metrics failed: {e!r}"[:120]

    # reference-based: recall@k only when the query matches a gold set
    gold = M.match_gold(q, DS.gold_entries(ds))
    recall = M.recall_at_k(retrieved_ids, gold.get("gold_ids", []), limit) if gold else None

    # reference-based LLM metrics: only when the gold entry carries a reference answer
    ref_based = {"context_recall": None, "context_entities_recall": None, "noise_sensitivity": None}
    if gold and gold.get("reference") and contexts:
        try:
            ref_based = await M.reference_based(answer_text, gold["reference"], contexts)
        except Exception as e:
            note = (note or "") + f" | reference metrics failed: {e!r}"[:120]

    def tile(key, label, value, ref):
        return {"key": key, "label": label, "value": value, "reference_based": ref}

    out_metrics = [
        tile("faithfulness", "Faithfulness", ref_free["faithfulness"], False),
        tile("answer_relevancy", "Answer Relevancy", ref_free["answer_relevancy"], False),
        tile("context_precision", "Context Precision", ref_free["context_precision"], False),
        tile("context_recall", "Context Recall", ref_based["context_recall"], True),
        tile("context_entities_recall", "Context Entities Recall", ref_based["context_entities_recall"], True),
        tile("noise_sensitivity", "Noise Sensitivity", ref_based["noise_sensitivity"], True),
        tile("recall_at_k", "recall@k", recall, True),
    ]
    return {
        "query": q, "strategy": strategy, "dataset": ds.id,
        "gold_matched": bool(gold), "gold_id": gold.get("id") if gold else None,
        "grounded_chunks": len(contexts),
        "answer_chars": len(answer_text or ""),
        "time_ms": round((_t.time() - t0) * 1000, 1),
        "note": note or None,
        "metrics": out_metrics,
    }


@router.get("/unified/analyze")
async def analyze(q: str = Query(..., min_length=1)):
    intent = analyze_query_intent(q)
    return {
        "query": q,
        "recommended_strategy": intent.strategy.value,
        "confidence": intent.confidence,
        "signals": intent.signals,
        "active_strategies": ["vector"] + sorted(LIGHTRAG_ALIASES),
        "parked": sorted(PARKED),
    }


@router.get("/unified/status")
async def corpus_status():
    """Corpus / embed-host status for both retrievers."""
    out = {"vector": vector_provider.status()}
    try:
        from retrieval import lightrag_provider
        out["lightrag"] = lightrag_provider.status()
    except Exception as e:
        out["lightrag"] = {"error": f"import failed: {e!r}"[:120]}
    return out


@router.post("/unified/reindex")
async def reindex():
    """Drop the cached corpus index so the next query reloads vdb_chunks.json."""
    vector_provider._INDEX.clear()
    return {"ok": True}
