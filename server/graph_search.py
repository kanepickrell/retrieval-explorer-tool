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

from fastapi import APIRouter, HTTPException, Query

from unified_search import analyze_query_intent  # backend-agnostic intent signals
from retrieval import vector_provider
import datasets as DS
import runs as RUNS

# kept for main.py back-compat; retrieval no longer depends on it
_gdb = None
PARKED = {"keyword", "neural"}           # need GraphDB/SPARQL -- parked for now
NOT_PORTED: set = set()                  # (lightrag now ported)
LIGHTRAG_ALIASES = {"lightrag", "mix", "local", "global", "hybrid"}
LEXICAL = {"bm25", "bm25_vector"}         # keyword retrieval, and keyword + vector fusion


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
        None, description="vector | bm25 | bm25_vector | local | global | hybrid | mix"),
    dataset: Optional[str] = Query(None, description="dataset id from GET /datasets"),
    top_k: Optional[int] = Query(None, le=200, description="LightRAG entity/relation top_k"),
    fresh_keywords: bool = Query(False, description="bypass the keyword cache (cold LLM extraction)"),
):
    """Retrieve under one strategy and store the result as a run. Returns run_id:
    /answer and /metrics take it, so they work on exactly this retrieval."""
    return await run_unified(q, limit, force_strategy, dataset, top_k, fresh_keywords)


async def run_unified(q: str, limit: int = 20, force_strategy: Optional[str] = None,
                      dataset: Optional[str] = None, top_k: Optional[int] = None,
                      fresh_keywords: bool = False, exclude_docs: Optional[set] = None) -> dict:
    """The retrieval behind /unified, callable directly (e.g. by /api/assist).
    exclude_docs: doc ids to drop from the results (a replayed chat must not
    retrieve itself); excluded docs are removed before the run is stored, so
    answers and metrics never see them either."""
    t0 = time.time()
    ds = DS.get(dataset)
    sd = str(ds.storage)
    intent = analyze_query_intent(q)
    strategy = (force_strategy or "vector").lower()

    note, results, graph, lr_ms, steps = None, [], None, None, []
    contexts, context_text, system_prompt = [], "", None
    keywords_cached = None  # None = not applicable (vector has no keyword step)
    if strategy == "vector" or strategy in LEXICAL:
        if strategy == "vector":
            results = run_vector(q, limit, sd)
            steps = vector_provider.LAST_STEPS
        else:
            from retrieval import bm25_provider
            results = (bm25_provider.search(q, limit, sd) if strategy == "bm25"
                       else bm25_provider.search_fused(q, limit, sd))
            steps = bm25_provider.LAST_STEPS
        if exclude_docs:
            results = [r for r in results if r["id"] not in exclude_docs]
            for i, r in enumerate(results, 1):
                r["rank"] = i
        contexts = [r.pop("content", "") for r in results]
        # exactly the context naive RAG generates over
        context_text = "\n\n".join(f"[{r['id']}]\n{c}" for r, c in zip(results, contexts))
        if not results:
            note = ("no keyword matches for this query" if strategy == "bm25" else
                    "no vector results -- check the embed host "
                    f"({vector_provider.EMBED_HOST}) and corpus storage")
    elif strategy in LIGHTRAG_ALIASES:
        # lazy import so a missing LightRAG install never breaks the vector path
        try:
            from retrieval import lightrag_provider
            lr_mode = "mix" if strategy == "lightrag" else strategy
            results = await lightrag_provider.search(q, limit, mode=lr_mode, storage_dir=sd, top_k=top_k,
                                                     fresh_keywords=fresh_keywords)
            keywords_cached = lightrag_provider.LAST_KEYWORDS_CACHED
            graph = lightrag_provider.LAST_GRAPH  # subgraph for the retrieval graph UI
            lr_ms = lightrag_provider.LAST_QUERY_MS
            steps = lightrag_provider.LAST_STEPS
            contexts = lightrag_provider.LAST_CONTEXT_CHUNKS
            system_prompt = lightrag_provider.LAST_PROMPT
            if exclude_docs:
                # drop the excluded docs' chunks from the rows; LightRAG's rendered prompt
                # can't be edited safely, so answers for such runs use naive RAG over the
                # remaining chunks instead (system_prompt=None)
                keep = [i for i, r in enumerate(results) if r["id"] not in exclude_docs]
                if len(keep) != len(results):
                    results = [results[i] for i in keep]
                    for i, r in enumerate(results, 1):
                        r["rank"] = i
                    excluded_ids = {c for c in exclude_docs}
                    contexts = [c for c, ch in zip(contexts, lightrag_provider._CUR["chunks"])
                                if lightrag_provider._title_of(ch) not in excluded_ids]
                    system_prompt = None
            context_text = system_prompt or ""
            if not results:
                note = lightrag_provider.LAST_ERROR or "lightrag returned no context for this query"
        except Exception as e:
            note = f"lightrag unavailable: {e!r}"[:160]
    elif strategy in PARKED:
        note = f"'{strategy}' is parked (needs GraphDB/SPARQL, on hold)"
    else:
        note = f"unknown strategy '{strategy}', use vector"

    # retrieval time only. LightRAG: timed inside its query lock (excludes queue wait).
    # vector/bm25/fusion: the sum of their measured steps, which excludes the extra
    # query embedding used only for the cross-technique display score (cosine).
    if lr_ms is not None:
        retrieval_ms = lr_ms
    elif steps:
        retrieval_ms = round(sum(st.get("ms") or 0 for st in steps), 1)
    else:
        retrieval_ms = round((time.time() - t0) * 1000, 1)
    run = RUNS.new_run(dataset=ds.id, query=q, strategy=strategy, limit=limit, top_k=top_k,
                       retrieved_ids=[r["id"] for r in results], contexts=contexts,
                       context_text=context_text, system_prompt=system_prompt,
                       retrieval_ms=retrieval_ms, keywords_cached=keywords_cached)
    return {
        "run_id": run.id,
        "query": q,
        "dataset": ds.id,
        "strategy_used": strategy if (results or strategy == "vector" or strategy in LEXICAL) else "none",
        "intent_analysis": {
            "recommended_strategy": intent.strategy.value,
            "confidence": intent.confidence,
            "signals": intent.signals,
            "search_terms": intent.search_terms,
        },
        "total_results": len(results),
        "time_ms": retrieval_ms,
        # true = keyword extraction was skipped (cached); compare latencies like with like
        "keywords_cached": keywords_cached,
        # time spent getting keywords (LLM call when cold, ~0 when cached); time_ms
        # minus this is the retrieval itself, which is what differs between modes
        "keyword_ms": next((s.get("ms") for s in steps if s.get("label", "").startswith(("Extract keywords", "Keywords"))), None),
        "note": note,
        "graph": graph,  # {nodes, edges} for lightrag strategies, else null
        "steps": steps,  # measured pipeline stages: [{label, detail, stat, ms}]
        "results": results,
    }


def _get_run(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        # never silently re-retrieve: the caller must re-run retrieval so what is
        # evaluated is what is shown
        raise HTTPException(404, f"run '{run_id}' not found (backend restarted or run expired) -- re-run the search")
    return run


async def _ensure_answer(run) -> None:
    """Generate the run's answer once, over the run's stored context, and cache it."""
    if run.answer is not None:
        return
    from retrieval import lightrag_provider
    t0 = time.time()
    try:
        if not run.retrieved_ids:
            run.answer, run.answer_note = "", "nothing was retrieved, so there is nothing to ground an answer in"
        elif run.system_prompt:
            # LightRAG modes: the exact prompt LightRAG built for this context
            run.answer = await lightrag_provider.generate_from_prompt(run.query, run.system_prompt)
        else:
            # vector: naive RAG over the retrieved chunks
            run.answer = await lightrag_provider.generate_over(run.query, run.context_text)
        if not run.answer and not run.answer_note:
            run.answer_note = "LLM returned an empty answer -- check the LLM host"
    except Exception as e:
        run.answer, run.answer_note = "", f"generation failed: {e!r}"[:160]
    run.answer_ms = round((time.time() - t0) * 1000, 1)


@router.get("/answer")
async def answer(run_id: str = Query(..., description="run_id from /unified")):
    """Generate (once) and return the answer for a stored retrieval run."""
    run = _get_run(run_id)
    await _ensure_answer(run)
    return {
        "run_id": run.id, "query": run.query, "strategy": run.strategy,
        "answer": run.answer,
        "grounded_chunks": len(run.contexts),
        "time_ms": run.answer_ms,
        "note": run.answer_note,
    }


@router.get("/metrics")
async def evaluate(run_id: str = Query(..., description="run_id from /unified")):
    """Ragas-style metrics for a stored run: its retrieved contexts and its answer
    (the same answer /answer returns). Reference-free metrics are computed for any
    query; reference-based ones need the query in the dataset's gold set and
    otherwise report pending (value null)."""
    import metrics as M

    run = _get_run(run_id)
    t0 = time.time()
    await _ensure_answer(run)
    ds = DS.get(run.dataset)
    note = run.answer_note
    answer_text, contexts = run.answer or "", run.contexts

    ref_free = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}
    if answer_text and contexts:
        try:
            ref_free = await M.reference_free(run.query, answer_text, contexts, grounding=run.context_text or None)
        except Exception as e:
            note = (note or "") + f" | metrics failed: {e!r}"[:120]

    gold = M.match_gold(run.query, DS.gold_entries(ds))
    recall = M.recall_at_k(run.retrieved_ids, gold.get("gold_ids", []), run.limit) if gold else None
    precision = M.precision_at_k(run.retrieved_ids, gold.get("gold_ids", []), run.limit) if gold else None

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
        tile("precision_at_k", "precision@k", precision, True),
    ]
    return {
        "run_id": run.id, "query": run.query, "strategy": run.strategy, "dataset": ds.id,
        "gold_matched": bool(gold), "gold_id": gold.get("id") if gold else None,
        "grounded_chunks": len(contexts),
        "answer": answer_text,
        "time_ms": round((time.time() - t0) * 1000, 1),
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
