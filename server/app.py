"""
Retrieval Explorer backend (corpus RAG, GraphDB-free).

Serves exactly what the frontend's src/lib/api.ts consumes:
  GET /graph                      -> corpus knowledge graph {nodes, edges}
  GET /api/search/unified?q=...   -> retrieval over the corpus (vector + LightRAG)
     &limit=..&force_strategy=vector|local|global|hybrid|mix

It reads the synthetic Agent Hub corpus only (LightRAG storage on disk + an
Ollama embed/chat host). It never touches GraphDB / the pb-kane graph, so no
real user data is exposed.

Self-contained: the retrieval providers (graph_search / unified_search /
retrieval/*) live beside this file -- nothing is imported from another repo.
Config via env:

  PROTOGRAPH_CORPUS_STORAGE  LightRAG storage dir   (default C:\\repos\\lightrag-explainer\\agenthub_storage)
  PROTOGRAPH_EMBED_HOST      Ollama embed/chat host (default http://10.128.241.200:11434)
  PROTOGRAPH_LLM             LightRAG generation model (default gemma3:12b)

Run (see requirements.txt for deps):
  uvicorn app:app --host 0.0.0.0 --port 8000
or just:  python app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# make this dir importable no matter where the process is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from graph_search import init_graph_search, router as search_router  # noqa: E402
import datasets as DS  # noqa: E402

app = FastAPI(title="Retrieval Explorer backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
init_graph_search(None)          # retrieval is corpus-backed; no GraphDB handle needed
app.include_router(search_router)  # mounts /api/search/unified and friends
from assist import router as assist_router  # noqa: E402
app.include_router(assist_router)  # /api/assist (chime-in)
from live import router as live_router  # noqa: E402
app.include_router(live_router)  # /api/live (feed of chime-in calls for the Live page)
from agent_loop import router as agent_router  # noqa: E402
app.include_router(agent_router)  # /api/agent (routing agent: runs + live events)
from ping_cycle import router as ping_router  # noqa: E402
app.include_router(ping_router)  # /api/ping (Ping cycle: profile, probes, gate, per-turn stages)

_GRAPH_CACHE: dict = {}   # graphml path -> (mtime, graph)


def _load_corpus_graph(dataset: str | None = None) -> dict:
    """Read a dataset's LightRAG graphml into the frontend's {nodes, edges} shape.
    Cached per file, reloaded when the file changes (e.g. after a re-index)."""
    graphml = DS.get(dataset).storage / "graph_chunk_entity_relation.graphml"
    if not graphml.exists():
        raise HTTPException(404, f"no graph for dataset '{dataset}' (index not built yet)")
    mtime = graphml.stat().st_mtime
    hit = _GRAPH_CACHE.get(str(graphml))
    if hit and hit[0] == mtime:
        return hit[1]
    import networkx as nx

    G = nx.read_graphml(graphml)
    deg = dict(G.degree())
    nodes = []
    for n, d in G.nodes(data=True):
        etype = (d.get("entity_type") or "UNKNOWN").strip().strip('"') or "UNKNOWN"
        nodes.append({
            "id": str(n),
            "label": (lambda s: s if len(s) <= 60 else s[:59] + "…")(str(n).strip('"')),
            "cluster": etype.lower(),
            "size": 40 + min(deg.get(n, 0), 25) * 4,  # degree -> radius variety
            "type": etype,
            "description": (d.get("description") or "")[:300],
        })
    edges = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        try:
            w = float(d.get("weight", 1.0))
        except Exception:
            w = 1.0
        edges.append({
            "id": f"e{i}:{u}|{v}",
            "source": str(u), "target": str(v),
            "weight": w, "type": "RELATED",
            "label": (d.get("keywords") or d.get("description") or "")[:60],
        })
    g = {"nodes": nodes, "edges": edges}
    _GRAPH_CACHE[str(graphml)] = (mtime, g)
    return g


@app.get("/datasets")
def datasets():
    """Every corpus the backend can serve, with index status and gold-set info."""
    return DS.list_all()


@app.get("/documents")
def documents(dataset: str | None = Query(None)):
    return DS.documents(DS.get(dataset))


@app.get("/documents/chunks")
def document_chunks(doc: str = Query(..., description="document id (file name) from /documents"),
                    dataset: str | None = Query(None)):
    """One document's chunks in order, with the entities extracted from each."""
    return DS.chunks_of(DS.get(dataset), doc)


@app.get("/documents/source")
def document_source(doc: str = Query(..., description="document id (file name), e.g. chat-<id>.md"),
                    dataset: str | None = Query(None)):
    """One document's full source markdown, as it was indexed. This is what Ping links to:
    the readable copy of a surfaced chat, file, agent or step."""
    import re as _re
    ds = DS.get(dataset)
    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.md", doc or "") or not ds.docs_dir:
        raise HTTPException(404, "no such document")
    f = ds.docs_dir / doc
    if not f.is_file():
        raise HTTPException(404, f"'{doc}' is not in dataset '{ds.id}'")
    md = f.read_text(encoding="utf-8", errors="replace")
    from assist import doc_title
    return {"dataset": ds.id, "doc": doc, "kind": doc.split("-", 1)[0],
            "title": doc_title(ds, doc, md), "markdown": md}


@app.get("/graph")
def graph(dataset: str | None = Query(None)):
    return _load_corpus_graph(dataset)


@app.get("/api/llm/stats")
def llm_stats_(last: int = Query(30, ge=0, le=500)):
    """Every LLM / query-embedding call to Ollama: per-task latency, failures, reloads."""
    from retrieval import llm_stats
    return llm_stats.summary(last)


@app.get("/api/graph/stats")
def graph_stats(dataset: str | None = Query(None)):
    g = _load_corpus_graph(dataset)
    return {"dataset": DS.get(dataset).id, "nodes": len(g["nodes"]), "edges": len(g["edges"])}


_STATIC = Path(os.getenv("STATIC_DIR", "")) if os.getenv("STATIC_DIR") else None


@app.get("/")
def root():
    if _STATIC and (_STATIC / "index.html").exists():
        from fastapi.responses import FileResponse
        return FileResponse(_STATIC / "index.html")
    return {
        "ok": True,
        "datasets": "/datasets",
        "graph": "/graph?dataset=<id>",
        "search": "/api/search/unified?q=...&force_strategy=vector&dataset=<id>",
    }


if _STATIC and (_STATIC / "index.html").exists():
    # Built frontend served from the same origin (the Docker image sets STATIC_DIR).
    # Registered last so every API route above wins; unknown paths get the SPA.
    from fastapi.responses import FileResponse

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = (_STATIC / path).resolve()
        if f.is_file() and f.is_relative_to(_STATIC.resolve()):
            return FileResponse(f)
        return FileResponse(_STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
