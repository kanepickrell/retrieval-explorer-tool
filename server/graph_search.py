"""
Retrieval over a dataset's chunk index, as one endpoint for every technique:

  vector       dense retrieval over the chunk embeddings (retrieval/vector_provider.py)
  bm25         keyword retrieval (retrieval/bm25_provider.py)
  bm25_vector  BM25 + vector, Reciprocal Rank Fusion (the default for Ping and the agent)
  agent        the routing agent (agent_loop.py): goal -> hypotheses -> metadata graph -> scoped search

  GET /api/search/unified?q=&limit=&force_strategy=&dataset=   -> a stored run (run_id)
  GET /api/search/answer?run_id=    the answer generated over exactly that run's context
  GET /api/search/metrics?run_id=   Ragas-style metrics + recall@k for that run

LightRAG's graph modes (local / global / hybrid / mix) were removed on 2026-11-13: on the
72 real+generated gold questions they scored hit@5 0.75-0.83 against 0.972 for
bm25_vector, and each cold query needed an LLM call. LightRAG still builds the index.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from retrieval import vector_provider
import datasets as DS
import runs as RUNS

LEXICAL = {"bm25", "bm25_vector"}         # keyword retrieval, and keyword + vector fusion
STRATEGIES = ("vector", "bm25", "bm25_vector", "agent")


def run_vector(query: str, limit: int, storage_dir: str = None) -> List[Dict]:
    return vector_provider.search(query, limit, storage_dir=storage_dir)


router = APIRouter(prefix="/api/search", tags=["Unified Search (corpus RAG)"])


@router.get("/unified")
async def unified_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, le=100),
    force_strategy: Optional[str] = Query(
        None, description="vector | bm25 | bm25_vector | agent"),
    dataset: Optional[str] = Query(None, description="dataset id from GET /datasets"),
):
    """Retrieve under one strategy and store the result as a run. Returns run_id:
    /answer and /metrics take it, so they work on exactly this retrieval."""
    return await run_unified(q, limit, force_strategy, dataset)


_META_GRAPHS: dict = {}   # graphml path -> (mtime, nx graph)


def _metadata_subgraph(storage, doc_ids: List[str]) -> Optional[dict]:
    """Touched subgraph for a metadata-graph dataset (ingest/build_metadata_graph.py):
    the nodes whose card was retrieved plus their direct neighbours (user, team, topics,
    files, agents), so the graph view shows where each hit sits."""
    import networkx as nx
    f = storage / "graph_chunk_entity_relation.graphml"
    if not f.exists():
        return None
    mtime = f.stat().st_mtime
    hit = _META_GRAPHS.get(str(f))
    if not hit or hit[0] != mtime:
        hit = _META_GRAPHS[str(f)] = (mtime, nx.read_graphml(f))
    G = hit[1]
    wanted = set(doc_ids)
    seeds = [n for n, d in G.nodes(data=True) if d.get("file_path") in wanted]
    keep = set(seeds)
    for n in seeds:
        keep.update(G.neighbors(n))
    nodes = [{"id": n, "label": n, "type": G.nodes[n].get("entity_type", "UNKNOWN"),
              "file_path": G.nodes[n].get("file_path", "")} for n in keep]
    edges = [{"source": u, "target": v, "label": (d.get("keywords") or "")[:40]}
             for u, v, d in G.subgraph(keep).edges(data=True)]
    return {"nodes": nodes, "edges": edges}


async def run_unified(q: str, limit: int = 20, force_strategy: Optional[str] = None,
                      dataset: Optional[str] = None, exclude_docs: Optional[set] = None) -> dict:
    """The retrieval behind /unified, callable directly (e.g. by /api/assist).
    exclude_docs: doc ids to drop from the results (a replayed chat must not
    retrieve itself); excluded docs are removed before the run is stored, so
    answers and metrics never see them either."""
    t0 = time.time()
    ds = DS.get(dataset)
    sd = str(ds.storage)
    strategy = (force_strategy or "vector").lower()

    note, results, graph, agent_ms, steps = None, [], None, None, []
    contexts, context_text = [], ""
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
        if ds.storage.name == "storage-metadata":
            graph = _metadata_subgraph(ds.storage, [r["id"] for r in results])
        if not results:
            note = ("no keyword matches for this query" if strategy == "bm25" else
                    "no vector results -- check the embed host "
                    f"({vector_provider.EMBED_HOST}) and corpus storage")
    elif strategy == "agent":
        # routing agent (agent_loop.py): goal -> hypotheses -> validate on the metadata
        # graph -> scoped search. Its events also stream to the Agent page.
        import agent_loop
        arun = agent_loop.AgentRun(ds.id, [{"role": "user", "content": q}], "search")
        out = await agent_loop.run_agent(arun, require_clear=False, exclude=exclude_docs)
        via = {"chunk": "C", "card": "B+C", "graph": "E"}
        results = [{"id": d["doc"], "label": d["doc"], "type": "chunk", "chunk_id": d["chunk_id"] or d["doc"],
                    "description": (d["text"] or "")[:200], "content": d["text"] or "",
                    "relevance_score": d["score"], "rank": i, "via": via.get(d["via"]), "source": "agent"}
                   for i, d in enumerate(out.get("docs", [])[:limit], 1)]
        steps = [{"label": e["title"][:90], "detail": e["detail"][:240], "stat": e["kind"], "ms": e["took_ms"]}
                 for e in arun.events]
        graph = agent_loop.subgraph(ds.id, out)
        contexts = [r.pop("content", "") for r in results]
        context_text = "\n\n".join(f"[{r['id']}]\n{c}" for r, c in zip(results, contexts))
        note = out.get("error") or out.get("waiting") or (None if results else "the agent found nothing")
        agent_ms = round(arun.events[-1]["ms"], 1) if arun.events else None
    else:
        note = f"unknown strategy '{strategy}': use one of {', '.join(STRATEGIES)}"

    # retrieval time only. vector/bm25/fusion: the sum of their measured steps, which
    # excludes the extra query embedding used only for the cross-technique display score.
    if agent_ms is not None:
        retrieval_ms = agent_ms
    elif steps:
        retrieval_ms = round(sum(st.get("ms") or 0 for st in steps), 1)
    else:
        retrieval_ms = round((time.time() - t0) * 1000, 1)
    run = RUNS.new_run(dataset=ds.id, query=q, strategy=strategy, limit=limit,
                       retrieved_ids=[r["id"] for r in results], contexts=contexts,
                       context_text=context_text, retrieval_ms=retrieval_ms)
    return {
        "run_id": run.id,
        "query": q,
        "dataset": ds.id,
        "strategy_used": strategy if strategy in STRATEGIES else "none",
        "total_results": len(results),
        "time_ms": retrieval_ms,
        "note": note,
        "graph": graph,  # {nodes, edges}: the agent and metadata-graph datasets, else null
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
    from retrieval import ollama_client as ollama
    t0 = time.time()
    try:
        if not run.retrieved_ids:
            run.answer, run.answer_note = "", "nothing was retrieved, so there is nothing to ground an answer in"
        else:
            # RAG over the retrieved chunks, the same prompt and model for every technique
            run.answer = await ollama.generate_over(run.query, run.context_text)
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


@router.get("/unified/status")
async def corpus_status():
    """Vector index / embed-host status."""
    return {"vector": vector_provider.status()}


@router.post("/unified/reindex")
async def reindex():
    """Drop the cached corpus index so the next query reloads vdb_chunks.json."""
    vector_provider._INDEX.clear()
    return {"ok": True}
