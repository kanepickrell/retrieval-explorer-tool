# Retrieval Explorer backend

Corpus-RAG backend for the frontend. Serves the datasets under `datasets/`: the
synthetic Agent Hub corpus and any pseudonymized Agent Hub / Open WebUI export in
`datasets/incoming/` (gitignored, never committed). It never touches GraphDB.

| endpoint | returns |
|---|---|
| `GET /datasets`, `GET /documents[/chunks\|/source]` | the datasets and their docs, chunks and source files |
| `GET /graph`, `GET /api/graph/stats` | a dataset's knowledge graph `{nodes, edges}` (LightRAG or metadata graphml) |
| `GET /api/search/unified?q=&limit=&force_strategy=` | retrieval: `vector` (dense), `bm25`, `bm25_vector` (fusion, the default), or `agent` (routing agent), with ranked chunks (+ the touched subgraph for the agent and metadata datasets) |
| `GET /api/search/answer`, `GET /api/search/metrics` | the generated answer for a run, and its Ragas-style metrics + recall@k |
| `POST /api/assist`, `GET /api/assist/chats` | one chime-in decision for a conversation; replayable export chats |
| `GET /api/live` | feed of chime-in calls (the Live page) |
| `GET /api/llm/stats` | every LLM / query-embedding call to Ollama: per-task latency, failures, empty replies, model reloads |
| `POST /api/agent/run`, `GET /api/agent/runs[/{id}[/graph]]` | routing agent runs and their step-by-step events |
| `/api/ping/chats`, `/api/ping/sessions[/{id}[/next\|/say]]` | Ping cycle sessions: replay a real chat or type to the chat model; per-turn stages (the Agent page) |

## Run

Self-contained — the retrieval providers live in this folder, nothing is
imported from another repo. Needs the deps in `requirements.txt` plus a
reachable Ollama host for embeddings and generation.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python app.py          # -> http://localhost:8000
```

Config (env):

| var | default | meaning |
|---|---|---|
| `PROTOGRAPH_CORPUS_STORAGE` | `C:\repos\lightrag-explainer\agenthub_storage` | storage of the synthetic corpus (graphml + chunk vdb) |
| `PROTOGRAPH_EMBED_HOST` | `http://10.128.241.200:11434` | Ollama embed/chat host |
| `PROTOGRAPH_LLM` | `gemma3:12b` | generation model: answers, Ping's profile and follow-ups, the chat model, index builds |
| `PROTOGRAPH_JUDGE_LLM` | `gpt-oss:20b` | the evidence check (agent and Ping) |
| `PORT` | `8000` | listen port |

Point the frontend at it with `VITE_API_URL` (see `../.env`).

## Files

| file | role |
|---|---|
| `app.py` | FastAPI app: datasets, documents, `/graph`; mounts the routers below |
| `datasets.py` | dataset registry: each export's storage dirs, docs and gold sets |
| `graph_search.py` | `/api/search/unified` (retrieval) + `/api/search/answer` (generation) |
| `retrieval/vector_provider.py` | naive dense retrieval over the chunk vdb |
| `retrieval/bm25_provider.py` | BM25 over chunks, and BM25+vector Reciprocal Rank Fusion (`search_fused`) |
| `retrieval/ollama_client.py` | the one Ollama client: chat (`llm`, answer generation) and embeddings, on a shared session |
| `retrieval/llm_stats.py` | records every LLM / query-embedding call for `/api/llm/stats` |
| `runs.py` / `metrics.py` | retrieval runs kept for the Answer tab; Ragas-style metrics with the local judge |
| `assist.py` / `live.py` | chime-in decisions for a conversation, and their live feed |
| `agent_loop.py` | routing agent: goal → hypotheses → validate on the metadata graph → route (records / document / detail) → scoped dive → check, with H0 (the user's own words) as the fallback |
| `ping_cycle.py` | Ping: a profile of what the user is after across the conversation (hypotheses kept as artifact / insight / idea), checked on the metadata graph each turn, tested through the follow-ups, and acted on only past a gate; plus the chat model for typed sessions |
| `ingest/build_metadata_graph.py` | metadata graph from the export tables (no LLM): one card per record, written as `storage-metadata/` → dataset `<export>~metadata`; teams from `<export>/teams.json` (pseudonyms only) |
| `ingest/plant.py` + `ingest/packs/` | copy an export to `<export>-planted` and add a fictional test scenario with its answer key (`range_reset_beacons`) |
| `ingest/build_structural_gold.py` | `eval_set_structural.json`: questions about how records relate, answered exactly from the metadata graph |
