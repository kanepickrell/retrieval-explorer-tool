/**
 * Data seam for the Retrieval Explorer.
 *
 * Every piece of data the UI shows comes through the exported functions in this
 * file. They are mock implementations today; a real backend plugs in by
 * replacing the bodies below (keep the types + signatures) — nothing else in
 * the app needs to change.
 *
 * Additions beyond the original contract (all optional, so a backend can
 * omit them and the UI degrades gracefully):
 *   - Chunk.entities   : node ids mentioned by a chunk (drives "click chunk →
 *                        highlight on graph" and the vector/final-context colors)
 *   - Dataset.labeled  : whether gold answers exist (drives "needs gold" tiles)
 *   - listDocuments()  : documents inside a dataset, for the Prospector
 */

// ───────────────────────── Types (the contract) ─────────────────────────

export type Technique = "vector" | "local" | "global" | "hybrid" | "mix";
export type GraphNode = { id: string; label: string; cluster: string; type: string; size: number; description?: string };
export type GraphEdge = { id: string; source: string; target: string; weight: number; label?: string };
export type Step = { label: string; detail: string; stat?: string; ms?: number };
export type Chunk = { id: string; title: string; source: string; score: number; snippet?: string; entities?: string[] };
export type Metric = { key: string; label: string; value: number; referenceBased: boolean; pending?: boolean };
export type RetrievalResult = {
  strategy: Technique; timeMs: number;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }; // touched subgraph; empty for vector
  chunks: Chunk[]; steps: Step[]; answer: string; metrics: Metric[];
};
export type Dataset = { id: string; name: string; kind: "chat" | "document" | "user"; rows: number; updated: string; labeled?: boolean };
export type DocumentSummary = { id: string; title: string; chunks: number; entities: string[] };

export const TECHNIQUES: Technique[] = ["vector", "local", "global", "hybrid", "mix"];

// ───────────────────────── Backend seam ─────────────────────────
// The two network functions below hit the ProtoGraph corpus backend
// (server/app.py). Everything else (datasets, documents, the mock world) is
// still local and acts as an offline fallback when the backend is unreachable.
//   getGraph      -> GET  /graph                         (corpus knowledge graph)
//   runRetrieval  -> GET  /api/search/unified?q&limit&force_strategy
// Set VITE_API_URL to point at the backend (defaults to localhost:8000).

const API_BASE = ((import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** Undirected, deterministic edge id so a retrieval subgraph's edges match the
 *  full graph's edges (GraphCanvas highlights relations by edge id). */
const pairKey = (a: string, b: string) => (a < b ? `${a}|||${b}` : `${b}|||${a}`);

// ───────────────────────── Public API ─────────────────────────

export async function getGraph(datasetId: string): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> {
  try {
    const r = await fetch(`${API_BASE}/graph`);
    if (!r.ok) throw new Error(`/graph ${r.status}`);
    const d = await r.json();
    const nodes: GraphNode[] = (d.nodes ?? []).map((n: any) => ({
      id: String(n.id),
      label: String(n.label ?? n.id),
      cluster: String(n.cluster ?? n.type ?? "unknown").toLowerCase(),
      type: String(n.type ?? n.cluster ?? "unknown"),
      size: Number(n.size ?? 50),
      description: n.description ? String(n.description) : undefined,
    }));
    const seen = new Set<string>();
    const edges: GraphEdge[] = [];
    for (const e of d.edges ?? []) {
      const s = String(e.source), t = String(e.target);
      const id = pairKey(s, t);
      if (seen.has(id)) continue; // dedupe parallel/undirected edges -> unique, matchable ids
      seen.add(id);
      edges.push({ id, source: s, target: t, weight: Number(e.weight ?? 1), label: e.label ? String(e.label) : undefined });
    }
    return { nodes, edges };
  } catch (err) {
    console.warn("[api] getGraph: backend unreachable, using mock —", err);
    await sleep(80);
    return graphFor(datasetId);
  }
}

export async function runRetrieval(
  datasetId: string,
  query: string,
  technique: Technique,
  topK: number,
  chunkTopK: number,
): Promise<RetrievalResult> {
  try {
    const limit = Math.max(topK, chunkTopK, 5);
    const url = `${API_BASE}/api/search/unified?q=${encodeURIComponent(query)}&limit=${limit}&force_strategy=${technique}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`/api/search/unified ${r.status}`);
    return mapResult(await r.json(), technique, topK, chunkTopK);
  } catch (err) {
    console.warn("[api] runRetrieval: backend unreachable, using mock —", err);
    const result = mockRetrieve(datasetId, query, technique, topK, chunkTopK);
    await sleep(Math.min(900, 220 + result.timeMs / 4));
    return result;
  }
}

/** Map the backend's unified-search response onto RetrievalResult.
 *  The graph, chunks and counts are real; steps are derived from the real
 *  counts + extracted keywords; the answer summarises retrieved context; the
 *  Ragas metrics are marked `pending` (computed by the backend in phase 2). */
function mapResult(d: any, technique: Technique, topK: number, chunkTopK: number): RetrievalResult {
  const g = d?.graph ?? { nodes: [], edges: [] };
  const nodes: GraphNode[] = (g.nodes ?? []).map((n: any) => ({
    id: String(n.id),
    label: String(n.label ?? n.id),
    cluster: String(n.type ?? n.cluster ?? "unknown").toLowerCase(),
    type: String(n.type ?? n.cluster ?? "unknown"),
    size: 50,
  }));
  const seen = new Set<string>();
  const edges: GraphEdge[] = [];
  for (const e of g.edges ?? []) {
    const s = String(e.source), t = String(e.target);
    const id = pairKey(s, t);
    if (seen.has(id)) continue;
    seen.add(id);
    edges.push({ id, source: s, target: t, weight: Number(e.weight ?? 1), label: e.label ? String(e.label) : undefined });
  }
  const chunks: Chunk[] = (d?.results ?? []).slice(0, chunkTopK).map((x: any) => ({
    id: String(x.chunk_id ?? x.id),
    title: String(x.label ?? x.id),
    source: technique,
    score: Number(x.relevance_score ?? 0),
    snippet: (x.description ? String(x.description).trim() : "") || undefined,
  }));
  const keywords: string[] = d?.intent_analysis?.search_terms ?? [];
  const timeMs = Math.round(Number(d?.time_ms ?? 0));
  const steps = buildSteps(technique, keywords, nodes.length, edges.length, chunks.length, topK, chunkTopK, timeMs);
  // answer is generated lazily by generateAnswer() when the Answer tab is opened
  return { strategy: technique, timeMs, graph: { nodes, edges }, chunks, steps, answer: "", metrics: pendingMetrics() };
}

function buildSteps(
  tech: Technique, keywords: string[], nE: number, nR: number, nC: number,
  topK: number, chunkTopK: number, timeMs: number,
): Step[] {
  const kw: Step = { label: "Extract keywords", detail: keywords.join(", ") || "(none)", stat: `${keywords.length} kw` };
  if (tech === "vector") {
    return [kw, { label: "Vector chunks", detail: "cosine similarity over chunk embeddings", stat: `${nC} chunks @chunk_top_k ${chunkTopK}`, ms: timeMs }];
  }
  const steps: Step[] = [kw];
  if (tech !== "global") steps.push({ label: "Entity search", detail: "match query entities, expand neighbourhood", stat: `${nE} entities @top_k ${topK}` });
  if (tech !== "local") steps.push({ label: "Relation search", detail: "rank the strongest relationships", stat: `${nR} relations` });
  if (tech === "mix") steps.push({ label: "Vector chunks", detail: "cosine similarity over chunk embeddings", stat: `${nC} chunks @chunk_top_k ${chunkTopK}` });
  steps.push({ label: "Merge → final context", detail: "dedupe, truncate to token budget", stat: `${nE}E · ${nR}R · ${nC}C`, ms: timeMs });
  return steps;
}

const RAGAS: [string, string, boolean][] = [
  ["context_precision", "Context Precision", false],
  ["faithfulness", "Faithfulness", false],
  ["answer_relevancy", "Answer Relevancy", false],
  ["noise_sensitivity", "Noise Sensitivity", false],
  ["context_recall", "Context Recall", true],
  ["context_entities_recall", "Context Entities Recall", true],
  ["recall_at_k", "recall@k", true],
];
const pendingMetrics = (): Metric[] =>
  RAGAS.map(([key, label, referenceBased]) => ({ key, label, referenceBased, value: 0, pending: true }));

/** Generate a grounded answer for the current query under one technique. Kept
 *  separate from runRetrieval so the graph + chunks render instantly and this
 *  (slower, LLM) call is only made when the Answer tab is viewed. Same generator
 *  for every technique on the backend, so only the retrieval differs. */
export async function generateAnswer(
  _datasetId: string, query: string, technique: Technique, limit = 5,
): Promise<{ answer: string; timeMs: number; groundedChunks: number; note?: string }> {
  try {
    const url = `${API_BASE}/api/search/answer?q=${encodeURIComponent(query)}&limit=${limit}&force_strategy=${technique}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`/api/search/answer ${r.status}`);
    const d = await r.json();
    return {
      answer: String(d.answer ?? ""),
      timeMs: Math.round(Number(d.time_ms ?? 0)),
      groundedChunks: Number(d.grounded_chunks ?? 0),
      note: d.note ? String(d.note) : undefined,
    };
  } catch (err) {
    console.warn("[api] generateAnswer: backend unreachable —", err);
    return { answer: "", timeMs: 0, groundedChunks: 0, note: "answer generation unavailable" };
  }
}

/** Compute real Ragas-style metrics for the current run under one technique.
 *  Heavy (retrieval + generation + judge calls), so it's triggered on demand
 *  from the metrics strip. A metric with no computable value comes back pending
 *  (reference-based metrics without gold), rendered as "—"/"needs gold". */
export async function evaluate(
  _datasetId: string, query: string, technique: Technique, topK: number, chunkTopK: number,
): Promise<{ metrics: Metric[]; goldMatched: boolean; timeMs: number; note?: string }> {
  try {
    const limit = Math.max(topK, chunkTopK, 5);
    const url = `${API_BASE}/api/search/metrics?q=${encodeURIComponent(query)}&limit=${limit}&force_strategy=${technique}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`/api/search/metrics ${r.status}`);
    const d = await r.json();
    const metrics: Metric[] = (d.metrics ?? []).map((m: any) => ({
      key: String(m.key),
      label: String(m.label),
      referenceBased: !!m.reference_based,
      value: typeof m.value === "number" ? m.value : 0,
      pending: m.value === null || m.value === undefined,
    }));
    return { metrics, goldMatched: !!d.gold_matched, timeMs: Math.round(Number(d.time_ms ?? 0)), note: d.note ? String(d.note) : undefined };
  } catch (err) {
    console.warn("[api] evaluate: backend unreachable —", err);
    return { metrics: [], goldMatched: false, timeMs: 0, note: "metrics unavailable" };
  }
}

export async function listDatasets(): Promise<Dataset[]> {
  await sleep(80);
  return DATASETS;
}

export async function listDocuments(datasetId: string): Promise<DocumentSummary[]> {
  await sleep(80);
  return corpusFor(datasetId).map((c) => ({ id: c.id, title: c.title, chunks: c.chunks, entities: c.entities }));
}

// ───────────────────────── Mock world ─────────────────────────
// A fictional ML / data-platform org's knowledge base: chat sessions,
// runbooks and design docs. Neutral placeholder content — swap via backend.

type MockNode = { id: string; label: string; cluster: string; description: string; aliases?: string[] };

const N = (cluster: string) => (id: string, label: string, description: string, aliases: string[] = []): MockNode => ({
  id, label, cluster, description, aliases,
});
const person = N("person"), org = N("organization"), tool = N("tool"), tech = N("technique"), method = N("method"), model = N("model");

const NODES: MockNode[] = [
  person("ada", "Ada Brooks", "Search engineer; owns lexical retrieval and fusion experiments.", ["ada"]),
  person("ravi", "Ravi Menon", "Search engineer focused on dense retrieval and vector indexes.", ["ravi"]),
  person("lena", "Lena Ortiz", "Data engineer; maintains the nightly pipelines.", ["lena"]),
  person("tomas", "Tomás Silva", "Analytics engineer; dbt models and warehouse schema.", ["tomas"]),
  person("mei", "Mei Chen", "Applied ML researcher; graph extraction and chunking.", ["mei"]),
  person("jonah", "Jonah Reed", "Platform engineer; streaming and release tooling.", ["jonah"]),
  person("priya", "Priya Nair", "Infra lead; IaC, capacity and incident reviews.", ["priya"]),
  person("sam", "Sam Okafor", "Applied ML engineer; local LLM evaluation.", ["sam"]),

  org("search", "Search team", "Owns query understanding, ranking and the search API.", ["search"]),
  org("data", "Data team", "Owns ingestion, the warehouse and scheduled pipelines.", ["data"]),
  org("appliedml", "Applied ML team", "Prototypes model-backed features and evaluations.", ["applied", "ml"]),
  org("platform", "Platform team", "Owns messaging, deploys and developer tooling.", ["platform"]),
  org("infra", "Infra team", "Owns cloud infrastructure, observability and capacity.", ["infra", "infrastructure"]),

  tool("airflow", "Airflow", "Workflow scheduler for the nightly batch pipelines.", ["airflow", "dag", "scheduler"]),
  tool("dbt", "dbt", "SQL transformation framework for warehouse models.", ["dbt"]),
  tool("postgres", "Postgres", "Primary relational database and warehouse staging.", ["postgres", "postgresql", "database"]),
  tool("pgvector", "pgvector", "Postgres extension storing embedding vectors.", ["pgvector", "index"]),
  tool("kafka", "Kafka", "Event streaming backbone.", ["kafka", "streaming", "consumer"]),
  tool("grafana", "Grafana", "Dashboards and alerting.", ["grafana", "dashboard", "dashboards", "alerting"]),
  tool("terraform", "Terraform", "Infrastructure-as-code for cloud resources.", ["terraform", "iac"]),
  tool("ollama", "Ollama", "Local model server hosting embedding and chat models.", ["ollama", "host"]),

  tech("bm25", "BM25", "Lexical keyword ranking function.", ["bm25", "lexical", "keyword"]),
  tech("dense", "Dense retrieval", "Embedding-similarity passage retrieval.", ["dense", "vector", "semantic"]),
  tech("rerank", "Cross-encoder reranking", "Second-stage reranking of candidate passages.", ["rerank", "reranking", "reranker"]),
  tech("qexp", "Query expansion", "LLM-generated query rewrites to widen recall.", ["expansion", "rewrite", "rewrites"]),
  tech("chunking", "Semantic chunking", "Splitting documents on topic boundaries.", ["chunking", "chunk", "chunks"]),
  tech("hyde", "HyDE", "Hypothetical-document embeddings for retrieval.", ["hyde", "hypothetical"]),
  tech("kgx", "Graph extraction", "LLM extraction of entities and relations into a graph.", ["graph", "extraction", "entities"]),
  tech("rrf", "Reciprocal rank fusion", "Merging ranked lists from multiple retrievers.", ["rrf", "fusion", "hybrid"]),

  method("backfill", "Backfill", "Re-running pipelines over historical partitions.", ["backfill", "backfills", "timeout", "timeouts"]),
  method("abtest", "A/B testing", "Online controlled experiments.", ["a/b", "ab", "experiment", "experiments"]),
  method("canary", "Canary deploy", "Gradual rollout to a slice of traffic.", ["canary", "rollout", "deploy"]),
  method("migration", "Schema migration", "Versioned changes to table schemas.", ["migration", "schema"]),
  method("loadtest", "Load testing", "Synthetic traffic to find capacity limits.", ["load", "capacity"]),
  method("postmortem", "Postmortem", "Blameless incident review.", ["postmortem", "incident", "outage"]),

  model("nomic", "nomic-embed-text", "Text embedding model.", ["embedding", "embeddings", "embed", "nomic"]),
  model("bge", "bge-reranker", "Cross-encoder reranker model.", ["bge"]),
  model("gemma", "gemma3:12b", "Mid-size chat model used for extraction.", ["gemma"]),
  model("llama", "Llama 3.1 8B", "Small chat model used for rewrites.", ["llama"]),
  model("mistral", "Mistral 7B", "Small chat model under evaluation.", ["mistral"]),
  model("gptoss", "gpt-oss-20b", "Open-weight model used as an eval judge.", ["judge", "gpt-oss"]),
];

// [source, target, label, weight]
const EDGE_ROWS: [string, string, string, number][] = [
  ["ada", "search", "member of", 1], ["ravi", "search", "member of", 1], ["lena", "data", "member of", 1],
  ["tomas", "data", "member of", 1], ["mei", "appliedml", "member of", 1], ["jonah", "platform", "member of", 1],
  ["priya", "infra", "member of", 1], ["sam", "appliedml", "member of", 1],
  ["ada", "bm25", "works on", 0.9], ["ada", "rrf", "works on", 0.85], ["ada", "rerank", "works on", 0.8], ["ada", "abtest", "ran", 0.6],
  ["ravi", "dense", "works on", 0.9], ["ravi", "nomic", "uses", 0.7], ["ravi", "pgvector", "tuned", 0.8], ["ravi", "hyde", "tried", 0.6],
  ["lena", "airflow", "maintains", 0.9], ["lena", "backfill", "solved", 0.95], ["lena", "dbt", "uses", 0.5],
  ["tomas", "dbt", "maintains", 0.9], ["tomas", "postgres", "uses", 0.6], ["tomas", "migration", "led", 0.85], ["tomas", "airflow", "uses", 0.5],
  ["mei", "kgx", "works on", 0.9], ["mei", "gemma", "uses", 0.7], ["mei", "chunking", "works on", 0.8], ["mei", "bge", "evaluated", 0.5],
  ["jonah", "kafka", "maintains", 0.9], ["jonah", "canary", "built", 0.85], ["jonah", "grafana", "uses", 0.5],
  ["priya", "terraform", "maintains", 0.9], ["priya", "loadtest", "ran", 0.75], ["priya", "ollama", "operates", 0.7], ["priya", "postmortem", "wrote", 0.8],
  ["sam", "llama", "evaluated", 0.8], ["sam", "mistral", "evaluated", 0.7], ["sam", "gptoss", "uses", 0.6], ["sam", "qexp", "works on", 0.85],
  ["search", "pgvector", "uses", 0.7], ["search", "bm25", "uses", 0.8], ["data", "airflow", "uses", 0.9], ["data", "postgres", "uses", 0.8],
  ["appliedml", "ollama", "uses", 0.7], ["infra", "grafana", "uses", 0.7], ["platform", "kafka", "uses", 0.9], ["infra", "terraform", "uses", 0.8],
  ["dense", "nomic", "embeds with", 0.8], ["rerank", "bge", "implemented by", 0.9], ["hyde", "gemma", "generates with", 0.6],
  ["kgx", "gemma", "extracts with", 0.8], ["qexp", "llama", "generates with", 0.7], ["rrf", "bm25", "fuses", 0.7], ["rrf", "dense", "fuses", 0.7],
  ["chunking", "dense", "feeds", 0.6], ["pgvector", "dense", "stores", 0.7], ["postgres", "pgvector", "extends", 0.6],
  ["ollama", "gemma", "serves", 0.6], ["ollama", "nomic", "serves", 0.6], ["backfill", "airflow", "runs on", 0.8],
  ["loadtest", "grafana", "measured in", 0.5], ["canary", "grafana", "monitored in", 0.5], ["postmortem", "ollama", "reviewed", 0.6],
  ["migration", "postgres", "applies to", 0.7],
];

type MockChunk = { id: string; title: string; kind: "chat" | "document"; entities: string[]; snippet: string; chunks: number };

const CORPUS: MockChunk[] = [
  { id: "c01", kind: "chat", title: "Session: nightly backfill keeps timing out", entities: ["lena", "airflow", "backfill", "data"], chunks: 6,
    snippet: "Lena: the backfill died at the 2h Airflow task timeout. Splitting it per partition and raising the pool slots fixed it — runs in 40 min now." },
  { id: "c02", kind: "document", title: "Runbook: Airflow backfills", entities: ["airflow", "backfill", "postgres", "tomas"], chunks: 9,
    snippet: "Always backfill by partition with max_active_runs=2; long single-run backfills hit the task timeout and hold Postgres locks." },
  { id: "c03", kind: "chat", title: "Session: reranker rollout", entities: ["ada", "rerank", "bge", "abtest", "search"], chunks: 5,
    snippet: "Ada: bge-reranker on the top 50 lifted click-through 4% in the A/B test; latency budget is the open question." },
  { id: "c04", kind: "document", title: "Design doc: hybrid search v2", entities: ["bm25", "dense", "rrf", "pgvector", "search", "ada", "ravi"], chunks: 14,
    snippet: "BM25 and dense candidates are fused with reciprocal rank fusion (k=60) before reranking; vectors live in pgvector." },
  { id: "c05", kind: "chat", title: "Session: HyDE experiment", entities: ["ravi", "hyde", "gemma", "dense"], chunks: 4,
    snippet: "Ravi: HyDE with gemma3 helped vague questions but hurt exact-identifier lookups, so it stays behind a flag." },
  { id: "c06", kind: "document", title: "Notes: graph extraction pilot", entities: ["mei", "kgx", "gemma", "chunking", "appliedml"], chunks: 8,
    snippet: "Entity/relation extraction with gemma3:12b over 150 chunks; relation precision was fine, entity merging needs work." },
  { id: "c07", kind: "chat", title: "Session: chunk sizes for long PDFs", entities: ["mei", "chunking", "nomic", "dense"], chunks: 5,
    snippet: "Mei: semantic chunking at ~600 tokens beat fixed 1k chunks for dense retrieval with nomic-embed-text." },
  { id: "c08", kind: "document", title: "Postmortem: embedding host outage", entities: ["priya", "ollama", "nomic", "postmortem", "infra"], chunks: 7,
    snippet: "The Ollama host ran out of disk; embedding calls failed for 3h. Action items: disk alerts and a warm standby." },
  { id: "c09", kind: "chat", title: "Session: Kafka consumer lag", entities: ["jonah", "kafka", "grafana", "platform"], chunks: 4,
    snippet: "Jonah: lag spikes came from a rebalance storm; static membership plus a Grafana lag alert settled it." },
  { id: "c10", kind: "document", title: "Runbook: canary deploys", entities: ["jonah", "canary", "grafana", "platform"], chunks: 6,
    snippet: "Promote a canary after 30 minutes at 5% traffic if error rate and p95 stay inside the Grafana SLO band." },
  { id: "c11", kind: "chat", title: "Session: dbt model migration", entities: ["tomas", "dbt", "migration", "postgres", "data"], chunks: 5,
    snippet: "Tomás: we moved the orders models to incremental dbt and versioned the schema migration in the same PR." },
  { id: "c12", kind: "document", title: "Load test report: search API", entities: ["priya", "loadtest", "grafana", "search"], chunks: 6,
    snippet: "Search API holds 400 rps at p95 180 ms; the reranker stage is the first to saturate." },
  { id: "c13", kind: "chat", title: "Session: choosing a local LLM", entities: ["sam", "llama", "mistral", "gptoss", "ollama", "appliedml"], chunks: 6,
    snippet: "Sam: Llama 3.1 8B for rewrites, gpt-oss-20b as the eval judge; Mistral 7B was close but less consistent." },
  { id: "c14", kind: "chat", title: "Session: query expansion prompts", entities: ["sam", "qexp", "llama", "search"], chunks: 3,
    snippet: "Sam: three rewrites per query raised recall on broad questions; we cap it so focused lookups are not diluted." },
  { id: "c15", kind: "document", title: "Terraform module notes", entities: ["priya", "terraform", "infra"], chunks: 4,
    snippet: "Modules are pinned per environment; plan output is posted to the PR before any apply." },
  { id: "c16", kind: "chat", title: "Session: pgvector index tuning", entities: ["ravi", "pgvector", "postgres", "search"], chunks: 4,
    snippet: "Ravi: switching to HNSW with ef_search=80 cut vector query time in half at the same recall@10." },
  { id: "c17", kind: "document", title: "Eval notes: reranking vs RRF", entities: ["ada", "rrf", "rerank", "bge", "bm25"], chunks: 5,
    snippet: "RRF alone is cheap and robust; adding the reranker gives the largest gain on long, multi-clause questions." },
  { id: "c18", kind: "document", title: "Team charter: Applied ML", entities: ["appliedml", "mei", "sam", "gemma"], chunks: 3,
    snippet: "Applied ML prototypes model-backed features and owns the shared evaluation harness." },
  { id: "c19", kind: "chat", title: "Session: backfill after a schema change", entities: ["tomas", "backfill", "migration", "airflow"], chunks: 4,
    snippet: "Tomás: run the migration first, then backfill only the affected partitions — never the whole table." },
  { id: "c20", kind: "document", title: "Onboarding: platform tooling", entities: ["platform", "infra", "terraform", "kafka", "jonah"], chunks: 5,
    snippet: "New engineers get Kafka topic access via Terraform; Platform and Infra share the on-call rotation." },
];

const DATASETS: Dataset[] = [
  { id: "ds-chat", name: "Team chat sessions", kind: "chat", rows: 1284, updated: "2026-11-08", labeled: true },
  { id: "ds-docs", name: "Runbooks & design docs", kind: "document", rows: 212, updated: "2026-11-02", labeled: false },
  { id: "ds-users", name: "User directory metadata", kind: "user", rows: 48, updated: "2026-10-27", labeled: false },
  { id: "ds-gold", name: "Focused eval set (gold)", kind: "chat", rows: 6, updated: "2026-11-09", labeled: true },
];

export const SAMPLE_QUERIES = [
  "Has anyone solved the Airflow backfill timeout problem?",
  "Who has worked on reranking, and which teams were they on?",
  "Which retrieval techniques show up across the chat sessions?",
  "Which teams have never used an embedding model?",
];

// ───────────────────────── Mock internals ─────────────────────────

const BY_ID = new Map(NODES.map((n) => [n.id, n]));
const ALL_EDGES: GraphEdge[] = EDGE_ROWS.map(([s, t, label, w], i) => ({ id: `e${i}`, source: s, target: t, weight: w, label }));
const DEGREE = new Map<string, number>();
for (const e of ALL_EDGES) {
  DEGREE.set(e.source, (DEGREE.get(e.source) ?? 0) + 1);
  DEGREE.set(e.target, (DEGREE.get(e.target) ?? 0) + 1);
}

function toGraphNode(n: MockNode): GraphNode {
  return { id: n.id, label: n.label, cluster: n.cluster, type: n.cluster, size: DEGREE.get(n.id) ?? 1, description: n.description };
}

function graphFor(datasetId: string) {
  const keep = datasetId === "ds-users"
    ? NODES.filter((n) => n.cluster === "person" || n.cluster === "organization")
    : NODES;
  const ids = new Set(keep.map((n) => n.id));
  return {
    nodes: keep.map(toGraphNode),
    edges: ALL_EDGES.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ ...e })),
  };
}

function corpusFor(datasetId: string): MockChunk[] {
  const ds = DATASETS.find((d) => d.id === datasetId);
  if (!ds || ds.kind === "user") return CORPUS.filter((c) => c.id === "c18" || c.id === "c20");
  return CORPUS.filter((c) => c.kind === ds.kind);
}

const STOP = new Set("a an the and or of on in to for has have had anyone who what which were was is are with across show up never been their they there any how do does did i we you it this that".split(" "));

const CLUSTER_HINTS: Record<string, string> = {
  who: "person", people: "person", team: "organization", teams: "organization",
  techniques: "technique", technique: "technique", retrieval: "technique",
  tools: "tool", tool: "tool", models: "model", model: "model", methods: "method",
};

function tokenize(q: string) {
  return q.toLowerCase().replace(/[^a-z0-9/\- ]/g, " ").split(/\s+/).filter(Boolean);
}

function hash(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
/** deterministic 0..1 noise */
function noise(seed: string) { return (hash(seed) % 10000) / 10000; }
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const round = (v: number, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
function sleep(ms: number) { return new Promise((r) => setTimeout(r, ms)); }

function mockRetrieve(datasetId: string, query: string, technique: Technique, topK: number, chunkTopK: number): RetrievalResult {
  const graph = graphFor(datasetId);
  const nodeIds = new Set(graph.nodes.map((n) => n.id));
  const tokens = tokenize(query);
  const content = tokens.filter((t) => !STOP.has(t));
  const hintClusters = new Set(tokens.map((t) => CLUSTER_HINTS[t]).filter(Boolean));

  // 1 · keyword extraction → seed entities
  const seedScore = new Map<string, number>();
  for (const n of NODES) {
    if (!nodeIds.has(n.id)) continue;
    const words = new Set([...tokenize(n.label), ...(n.aliases ?? [])]);
    let s = 0;
    for (const t of content) if (words.has(t)) s += 1;
    if (s > 0) seedScore.set(n.id, s);
  }
  if (seedScore.size === 0) {
    [...graph.nodes].sort((a, b) => b.size - a.size).slice(0, 3).forEach((n) => seedScore.set(n.id, 0.5));
  }
  const seeds = [...seedScore.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
  const keywords = [...new Set([...content.filter((t) => t.length > 2), ...[...hintClusters].map((c) => `#${c}`)])].slice(0, 8);
  const broad = hintClusters.size > 0 || /\b(across|which teams|who)\b/i.test(query);

  const adj = new Map<string, GraphEdge[]>();
  for (const e of graph.edges) {
    (adj.get(e.source) ?? adj.set(e.source, []).get(e.source)!).push(e);
    (adj.get(e.target) ?? adj.set(e.target, []).get(e.target)!).push(e);
  }
  const other = (e: GraphEdge, id: string) => (e.source === id ? e.target : e.source);

  // 2 · local: seeds + neighbours (entity-centric), with cluster-hint boost
  const localEntities = (): string[] => {
    const score = new Map<string, number>();
    seeds.forEach((id, i) => score.set(id, 10 - i * 0.1));
    for (const id of seeds) for (const e of adj.get(id) ?? []) {
      const o = other(e, id);
      const boost = hintClusters.has(BY_ID.get(o)!.cluster) ? 1.5 : 0;
      score.set(o, Math.max(score.get(o) ?? 0, e.weight * 3 + boost));
    }
    return [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK).map(([id]) => id);
  };

  // 3 · global: relation-centric — strongest edges touching seeds or hinted clusters
  const globalRelations = (): GraphEdge[] => {
    const seedSet = new Set(seeds);
    const scored = graph.edges.map((e) => {
      let s = 0;
      if (seedSet.has(e.source) || seedSet.has(e.target)) s += 2;
      if (hintClusters.has(BY_ID.get(e.source)!.cluster) || hintClusters.has(BY_ID.get(e.target)!.cluster)) s += 1;
      return { e, s: s * e.weight };
    }).filter((x) => x.s > 0);
    return scored.sort((a, b) => b.s - a.s).slice(0, topK).map((x) => x.e);
  };

  let entityIds: string[] = [];
  let relations: GraphEdge[] = [];
  if (technique === "local") {
    entityIds = localEntities();
    const set = new Set(entityIds);
    relations = graph.edges.filter((e) => set.has(e.source) && set.has(e.target));
  } else if (technique === "global") {
    relations = globalRelations();
    entityIds = [...new Set(relations.flatMap((e) => [e.source, e.target]))];
  } else if (technique === "hybrid" || technique === "mix") {
    const loc = localEntities();
    const glob = globalRelations();
    entityIds = [...new Set([...loc, ...glob.flatMap((e) => [e.source, e.target])])];
    const set = new Set(entityIds);
    const relIds = new Set(glob.map((e) => e.id));
    relations = graph.edges.filter((e) => relIds.has(e.id) || (set.has(e.source) && set.has(e.target)));
  }

  // 4 · chunks
  const corpus = corpusFor(datasetId);
  const entitySet = new Set(entityIds);
  const lexical = (c: MockChunk) => {
    const words = new Set(tokenize(`${c.title} ${c.snippet}`));
    const hits = content.filter((t) => words.has(t)).length;
    const seedHits = c.entities.filter((id) => seedScore.has(id)).length;
    return hits * 1.0 + seedHits * 0.8 + noise(c.id + query) * 0.3;
  };
  const graphScore = (c: MockChunk) => c.entities.filter((id) => entitySet.has(id)).length + noise(c.id + technique) * 0.4;

  // graph-derived chunks need real entity overlap; vector search always returns its nearest k
  const rank = (fn: (c: MockChunk) => number, source: string, minScore: number) =>
    corpus.map((c) => ({ c, s: fn(c) })).filter((x) => x.s > minScore).sort((a, b) => b.s - a.s).map((x) => ({ ...x, source }));

  let ranked: { c: MockChunk; s: number; source: string }[] = [];
  if (technique === "vector") ranked = rank(lexical, "vector", 0);
  else if (technique === "mix") {
    // interleave graph-derived and vector chunks (round-robin, deduped), like LightRAG's mix merge
    const g = rank(graphScore, "graph", 0.5), v = rank(lexical, "vector", 0);
    const seen = new Set<string>();
    for (let i = 0; i < Math.max(g.length, v.length); i++) {
      for (const x of [g[i], v[i]]) if (x && !seen.has(x.c.id)) { seen.add(x.c.id); ranked.push(x); }
    }
    const top = Math.max(...ranked.map((x) => x.s), 1);
    ranked = ranked.map((x, i) => ({ ...x, s: top * (1 - i * 0.06) }));
  } else ranked = rank(graphScore, technique, 0.5);
  ranked = ranked.slice(0, chunkTopK);
  const max = ranked[0]?.s ?? 1;
  const chunks: Chunk[] = ranked.map((x, i) => ({
    id: x.c.id, title: x.c.title, source: x.source, snippet: x.c.snippet, entities: x.c.entities.filter((id) => nodeIds.has(id)),
    score: round(clamp01((x.s / max) * (0.96 - i * 0.025))),
  }));

  // 5 · steps
  const ms = (k: string, base: number, spread: number) => Math.round(base + noise(k + query + technique) * spread);
  const steps: Step[] = [
    { label: "Extract keywords", detail: keywords.join(", ") || "(none)", stat: `${keywords.length} kw`, ms: ms("kw", 180, 160) },
  ];
  if (technique !== "vector") {
    const finalE = entityIds.length, finalR = relations.length;
    if (technique !== "global") steps.push({ label: "Entity search", detail: `${seeds.slice(0, 4).map((id) => BY_ID.get(id)!.label).join(", ")}…`, stat: `${finalE} entities @top_k ${topK}`, ms: ms("ent", 60, 70) });
    if (technique !== "local") steps.push({ label: "Relation search", detail: relations.slice(0, 3).map((e) => `${BY_ID.get(e.source)!.label} → ${BY_ID.get(e.target)!.label}`).join("; "), stat: `${finalR} relations`, ms: ms("rel", 70, 80) });
  }
  if (technique === "vector" || technique === "mix") {
    steps.push({ label: "Vector chunks", detail: "cosine similarity over chunk embeddings", stat: `${chunks.filter((c) => c.source === "vector").length} chunks @chunk_top_k ${chunkTopK}`, ms: ms("vec", 40, 50) });
  }
  if (technique !== "vector") {
    steps.push({ label: "Merge → final context", detail: "dedupe, truncate to token budget", stat: `${entityIds.length}E · ${relations.length}R · ${chunks.length}C`, ms: ms("merge", 8, 20) });
  }
  const timeMs = steps.reduce((a, s) => a + (s.ms ?? 0), 0) + ms("llm", 900, 700);

  // answer
  const answer = buildAnswer(query, technique, chunks, entityIds);

  // metrics: focused queries favour vector, broad aggregation favours graph modes
  const graphy = technique !== "vector";
  const fit = broad ? (graphy ? 0.12 : -0.14) : (graphy ? -0.08 : 0.1);
  const mixBonus = technique === "mix" ? 0.04 : 0;
  const m = (key: string, base: number) => round(clamp01(base + fit + mixBonus + (noise(key + query + technique + topK + chunkTopK) - 0.5) * 0.14));
  const metrics: Metric[] = [
    { key: "context_precision", label: "Context Precision", value: m("cp", 0.72 - (graphy ? entityIds.length * 0.004 : 0)), referenceBased: false },
    { key: "faithfulness", label: "Faithfulness", value: m("f", 0.8), referenceBased: false },
    { key: "answer_relevancy", label: "Answer Relevancy", value: m("ar", 0.76), referenceBased: false },
    { key: "noise_sensitivity", label: "Noise Sensitivity", value: round(clamp01(0.22 - fit + (graphy ? 0.05 : 0) + (noise("ns" + query + technique) - 0.5) * 0.1)), referenceBased: false },
    { key: "context_recall", label: "Context Recall", value: m("cr", 0.66), referenceBased: true },
    { key: "context_entities_recall", label: "Context Entities Recall", value: m("cer", graphy ? 0.7 : 0.48), referenceBased: true },
    { key: "recall_at_k", label: "recall@k", value: m("rk", 0.62), referenceBased: true },
  ];

  const resultNodes = entityIds.map((id) => toGraphNode(BY_ID.get(id)!));
  return { strategy: technique, timeMs, graph: { nodes: resultNodes, edges: relations.map((e) => ({ ...e })) }, chunks, steps, answer, metrics };
}

function buildAnswer(query: string, technique: Technique, chunks: Chunk[], entityIds: string[]): string {
  if (chunks.length === 0) return `I couldn't find anything in this dataset that answers **"${query}"**.`;
  const top = chunks.slice(0, 3);
  const lines = [
    `The most relevant thread is **${top[0].title}**.`,
    "",
    ...top.map((c) => `- ${c.snippet}`),
    "",
  ];
  if (technique === "vector") {
    lines.push("_Vector mode retrieves passages only — no graph context was used._");
  } else {
    const names = entityIds.slice(0, 6).map((id) => BY_ID.get(id)!.label);
    lines.push(`**Key entities:** ${names.join(", ")}.`);
  }
  return lines.join("\n");
}
