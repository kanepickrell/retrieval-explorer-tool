# Retrieval Explorer backend

Corpus-RAG backend for the frontend. Serves the two endpoints `src/lib/api.ts`
calls, over the **synthetic Agent Hub corpus only** — it never touches GraphDB
or any real-user graph.

| endpoint | returns |
|---|---|
| `GET /graph` | corpus knowledge graph `{nodes, edges}` (from the LightRAG graphml) |
| `GET /api/search/unified?q=&limit=&force_strategy=` | retrieval: `vector` (dense) or `local`/`global`/`hybrid`/`mix` (LightRAG), with the touched subgraph + ranked chunks |

## Run

Self-contained — the retrieval providers live in this folder, nothing is
imported from another repo. Needs the deps in `requirements.txt` plus a
reachable Ollama host for embeddings (and generation for the LightRAG modes).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python app.py          # -> http://localhost:8000
```

Config (env):

| var | default | meaning |
|---|---|---|
| `PROTOGRAPH_CORPUS_STORAGE` | `C:\repos\lightrag-explainer\agenthub_storage` | LightRAG storage (graphml + chunk vdb) |
| `PROTOGRAPH_EMBED_HOST` | `http://10.128.241.200:11434` | Ollama embed/chat host |
| `PROTOGRAPH_LLM` | `gemma3:12b` | LightRAG generation model |
| `PORT` | `8000` | listen port |

Point the frontend at it with `VITE_API_URL` (see `../.env`).

## Status (phase 1)

Graph, retrieval, touched subgraph, ranked chunks and the step counts are
**real**. Still to come (phase 2):

- generated answer (LLM over the retrieved context) — currently the Answer tab
  summarises the retrieved chunks and says so.
- Ragas metrics (Faithfulness, Answer Relevancy, Context Precision/Recall, …)
  and deterministic recall@k — currently rendered as `pending` tiles.

## Files

| file | role |
|---|---|
| `app.py` | FastAPI app: `/graph` + mounts the search router |
| `graph_search.py` | `/api/search/unified` (retrieval) + `/api/search/answer` (generation) |
| `unified_search.py` | query-intent analysis (`analyze_query_intent`) |
| `retrieval/vector_provider.py` | naive dense retrieval over the chunk vdb |
| `retrieval/lightrag_provider.py` | LightRAG local/global/hybrid/mix + answer generation |
