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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from graph_search import init_graph_search, router as search_router  # noqa: E402

STORAGE = Path(os.getenv("PROTOGRAPH_CORPUS_STORAGE", r"C:\repos\lightrag-explainer\agenthub_storage"))
GRAPHML = STORAGE / "graph_chunk_entity_relation.graphml"

app = FastAPI(title="Retrieval Explorer backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
init_graph_search(None)          # retrieval is corpus-backed; no GraphDB handle needed
app.include_router(search_router)  # mounts /api/search/unified and friends

_GRAPH_CACHE: dict | None = None


def _load_corpus_graph() -> dict:
    """Read the LightRAG graphml into the frontend's {nodes, edges} shape."""
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE
    import networkx as nx

    G = nx.read_graphml(GRAPHML)
    deg = dict(G.degree())
    nodes = []
    for n, d in G.nodes(data=True):
        etype = (d.get("entity_type") or "UNKNOWN").strip().strip('"') or "UNKNOWN"
        nodes.append({
            "id": str(n),
            "label": str(n).strip('"'),
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
    _GRAPH_CACHE = {"nodes": nodes, "edges": edges}
    return _GRAPH_CACHE


@app.get("/graph")
def graph():
    return _load_corpus_graph()


@app.get("/api/graph/stats")
def graph_stats():
    g = _load_corpus_graph()
    return {"nodes": len(g["nodes"]), "edges": len(g["edges"]), "source": "agenthub corpus (LightRAG)"}


@app.get("/")
def root():
    return {
        "ok": True,
        "graph": "/graph",
        "search": "/api/search/unified?q=...&force_strategy=vector",
        "storage": str(STORAGE),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
