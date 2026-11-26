# Running retrieval-explorer as a service in the Agent Hub stack

This folder packages retrieval-explorer's retrieval **engine** as a container so the
official Agent Hub / Open WebUI stack can use it as the backend for **chime-in (Ping)** —
and, later, as an in-stack admin dashboard for monitoring retrieval performance and data
scaling. Everything about retrieval-explorer stays the same; Open WebUI just becomes the
chat UI in front of it.

## How the pieces fit

- Open WebUI ships a chime-in **bridge** (`routers/chime_in.py`) that holds no retrieval
  logic. It POSTs the conversation to retrieval-explorer `POST /api/assist` and renders the
  `suggestions` as chime cards. The request/response contract was verified against this
  engine on 2026-11-13.
- `../Dockerfile` (already in the repo) builds a single image with the React UI **and** the
  FastAPI backend on one port (8000), serving the UI from `STATIC_DIR`. Datasets are mounted,
  not baked in.
- `agenthub-stack.yaml` is a compose **override** that adds a `retrieval-explorer` service
  to the stack and sets the chime-in env on the existing `open-webui` service.

```
Open WebUI  --POST /api/assist-->  retrieval-explorer  --->  Ollama (10.128.241.200)
 (chat UI)      (compose net)        (UI + engine :8000)       (remote GPU box)
                                            ^
                      browser --/doc, dashboard--/ (host :5180)
```

## Prerequisites

- Docker (not yet installed on the current machine — install Docker Desktop or run an
  engine in WSL first).
- The two repos as **siblings**: `…/agenthub-chime-in` and `…/retrieval-explorer`.
- The corpus datasets present on the host under `retrieval-explorer/server/datasets`
  (gitignored; built by `ingest/`). They are mounted into the container read side.

## Launch

```bash
# 1. set chime-in + engine env (see retrieval-explorer.env.example)
cp ../retrieval-explorer/deploy/retrieval-explorer.env.example .env.add
cat .env.add >> .env          # into the agenthub-chime-in .env

# 2. bring up the stack plus the engine (run from the agenthub-chime-in repo)
docker compose -f docker-compose.yaml -f ../retrieval-explorer/deploy/agenthub-stack.yaml up --build
```

Open WebUI comes up on its usual port (3000); chime-in calls the engine at
`http://retrieval-explorer:8000` over the compose network. Change `CHIME_DATASET` to point
Ping at a different corpus.

## Known gap — `/api/assist` is not (yet) the full Ping cycle

The chime-in bridge calls **`/api/assist`**: a rewrite → retrieve → score → LLM-judge path.
It is **not** the hypothesis-driven **Ping cycle** in `server/ping_cycle.py` (profile,
hypotheses, convergence gate, the async decouple). So as wired here, "Ping in the official
tool" is the assist engine — good manual-strike results and solid precision, but without the
cross-turn profile. To put the real Ping behaviour into Open WebUI, either expose the Ping
cycle behind an assist-compatible endpoint or fold its logic into `/api/assist`. Decide that
before investing further in the in-stack dashboard.

## Not done yet

- The image already bundles the UI + API, so retrieval-explorer's own pages (doc viewer,
  the `/api/llm/stats` monitoring view) come up at the published host port — the start of the
  in-stack **admin dashboard**. Embedding it as an admin pane *inside* Open WebUI is a later step.
- Nothing here has been built or run — Docker is unavailable on the current machine. These
  files are the ready-to-build groundwork; paths/ports may need a tweak on first real build.
