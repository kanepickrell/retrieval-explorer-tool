/**
 * Data seam for the Retrieval Explorer.
 *
 * Every piece of data the UI shows comes through the exported functions in this
 * file, and every one of them is served by the backend (server/app.py). There is
 * no mock fallback: if the backend is unreachable or errors, the call throws an
 * ApiError and the UI shows it, so a number on screen is always a real result.
 *
 *   listDatasets   -> GET /datasets                  corpora + index status + gold info
 *   listDocuments  -> GET /documents?dataset         indexed documents of a corpus
 *   getGraph       -> GET /graph?dataset             corpus knowledge graph
 *   runRetrieval   -> GET /api/search/unified        retrieval under one technique
 *   generateAnswer -> GET /api/search/answer         grounded answer (LLM)
 *   evaluate       -> GET /api/search/metrics        Ragas-style metrics (LLM judge)
 *
 * Set VITE_API_URL to point at the backend (defaults to localhost:8000).
 */

// ───────────────────────── Types (the contract) ─────────────────────────

export type Technique = "vector" | "local" | "global" | "hybrid" | "mix";
export type GraphNode = { id: string; label: string; cluster: string; type: string; size: number; description?: string };
export type GraphEdge = { id: string; source: string; target: string; weight: number; label?: string };
export type Step = { label: string; detail: string; stat?: string; ms?: number };
/** One retrieved document (its best/first chunk). score = cosine(query, chunk) with the
 *  same embeddings for every technique (null if the chunk has no embedding); `via` is how
 *  LightRAG reached it: E = entity, R = relation, C = vector search. */
export type Chunk = {
  id: string; title: string; source: string; score: number | null; rank: number;
  via?: "E" | "R" | "C"; snippet?: string; entities?: string[];
};
export type Metric = { key: string; label: string; value: number; referenceBased: boolean; pending?: boolean };
export type RetrievalResult = {
  /** backend run id: answer + metrics are computed on exactly this retrieval */
  runId: string;
  strategy: Technique; timeMs: number;
  /** graph modes: true if keyword extraction was skipped (cache hit), false if it ran
   *  (cold LLM call); null for vector, which has no keyword step */
  keywordsCached: boolean | null;
  /** time spent obtaining keywords (null for vector); timeMs - keywordMs = the search itself */
  keywordMs: number | null;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] }; // touched subgraph; empty for vector
  chunks: Chunk[]; steps: Step[]; answer: string; metrics: Metric[];
  note?: string;
};
export type DatasetStatus = "ready" | "indexing" | "missing";
export type Dataset = {
  id: string; name: string; kind: "chat" | "document" | "user";
  /** documents indexed so far */
  rows: number; expectedRows: number; failedRows: number;
  /** source docs whose content was already indexed under another id (indexed once) */
  duplicateRows: number;
  updated: string | null;
  /** has a gold eval set (enables reference-based metrics) */
  labeled: boolean; goldQueries: number; hasReferenceAnswers: boolean;
  status: DatasetStatus;
  /** short gold queries, offered as "try" chips */
  samples: string[];
};
export type DocumentSummary = { id: string; title: string; chunks: number; entities: string[] };

export const TECHNIQUES: Technique[] = ["vector", "local", "global", "hybrid", "mix"];

export class ApiError extends Error {}

const API_BASE = ((import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

async function getJson(path: string, params: Record<string, string | number | undefined> = {}): Promise<any> {
  const qs = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");
  const url = `${API_BASE}${path}${qs ? `?${qs}` : ""}`;
  let r: Response;
  try {
    r = await fetch(url);
  } catch {
    throw new ApiError(`backend unreachable at ${API_BASE}`);
  }
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json())?.detail ?? ""; } catch { /* non-JSON error body */ }
    throw new ApiError(`${path} → ${r.status}${detail ? `: ${detail}` : ""}`);
  }
  return r.json();
}

/** Undirected, deterministic edge id so a retrieval subgraph's edges match the
 *  full graph's edges (GraphCanvas highlights relations by edge id). */
const pairKey = (a: string, b: string) => (a < b ? `${a}|||${b}` : `${b}|||${a}`);

function mapEdges(raw: any[]): GraphEdge[] {
  const seen = new Set<string>();
  const edges: GraphEdge[] = [];
  for (const e of raw ?? []) {
    const s = String(e.source), t = String(e.target);
    const id = pairKey(s, t);
    if (seen.has(id)) continue; // dedupe parallel/undirected edges -> unique, matchable ids
    seen.add(id);
    edges.push({ id, source: s, target: t, weight: Number(e.weight ?? 1), label: e.label ? String(e.label) : undefined });
  }
  return edges;
}

// ───────────────────────── Public API ─────────────────────────

export async function listDatasets(): Promise<Dataset[]> {
  const d = await getJson("/datasets");
  return (d ?? []).map((x: any) => ({
    id: String(x.id), name: String(x.name), kind: (x.kind ?? "chat") as Dataset["kind"],
    rows: Number(x.rows ?? 0), expectedRows: Number(x.expected_rows ?? 0), failedRows: Number(x.failed_rows ?? 0),
    duplicateRows: Number(x.duplicate_rows ?? 0),
    updated: x.updated ?? null,
    labeled: !!x.labeled, goldQueries: Number(x.gold_queries ?? 0), hasReferenceAnswers: !!x.has_reference_answers,
    status: (x.status ?? "missing") as DatasetStatus,
    samples: (x.samples ?? []).map(String),
  }));
}

export async function listDocuments(datasetId: string): Promise<DocumentSummary[]> {
  const d = await getJson("/documents", { dataset: datasetId });
  return (d ?? []).map((x: any) => ({
    id: String(x.id), title: String(x.title), chunks: Number(x.chunks ?? 0), entities: (x.entities ?? []).map(String),
  }));
}

export async function getGraph(datasetId: string): Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }> {
  const d = await getJson("/graph", { dataset: datasetId });
  const nodes: GraphNode[] = (d.nodes ?? []).map((n: any) => ({
    id: String(n.id),
    label: String(n.label ?? n.id),
    cluster: String(n.cluster ?? n.type ?? "unknown").toLowerCase(),
    type: String(n.type ?? n.cluster ?? "unknown"),
    size: Number(n.size ?? 50),
    description: n.description ? String(n.description) : undefined,
  }));
  return { nodes, edges: mapEdges(d.edges) };
}

export async function runRetrieval(
  datasetId: string,
  query: string,
  technique: Technique,
  topK: number,
  chunkTopK: number,
): Promise<RetrievalResult> {
  const d = await getJson("/api/search/unified", {
    q: query, limit: chunkTopK, top_k: topK, force_strategy: technique, dataset: datasetId,
  });
  return mapResult(d, technique, topK, chunkTopK);
}

/** Map the backend's unified-search response onto RetrievalResult.
 *  Graph, chunks, scores, steps and timings are all measured by the backend. Quality metrics start
 *  `pending`; they are computed on demand by evaluate(). */
function mapResult(d: any, technique: Technique, _topK: number, chunkTopK: number): RetrievalResult {
  const g = d?.graph ?? { nodes: [], edges: [] };
  const nodes: GraphNode[] = (g.nodes ?? []).map((n: any) => ({
    id: String(n.id),
    label: String(n.label ?? n.id),
    cluster: String(n.type ?? n.cluster ?? "unknown").toLowerCase(),
    type: String(n.type ?? n.cluster ?? "unknown"),
    size: 50,
  }));
  const edges = mapEdges(g.edges);
  const chunks: Chunk[] = (d?.results ?? []).slice(0, chunkTopK).map((x: any, i: number) => ({
    id: String(x.chunk_id ?? x.id),
    title: String(x.label ?? x.id),
    source: technique,
    score: typeof x.relevance_score === "number" ? x.relevance_score : null,
    rank: Number(x.rank ?? i + 1),
    via: x.via ?? undefined,
    snippet: (x.description ? String(x.description).trim() : "") || undefined,
  }));
  const timeMs = Math.round(Number(d?.time_ms ?? 0));
  // measured by the backend on this exact query (stage timings, real keywords and counts)
  const steps: Step[] = (d?.steps ?? []).map((st: any) => ({
    label: String(st.label), detail: String(st.detail ?? ""),
    stat: st.stat ? String(st.stat) : undefined,
    ms: typeof st.ms === "number" ? Math.round(st.ms) : undefined,
  }));
  return {
    runId: String(d?.run_id ?? ""),
    keywordsCached: typeof d?.keywords_cached === "boolean" ? d.keywords_cached : null,
    keywordMs: typeof d?.keyword_ms === "number" ? Math.round(d.keyword_ms) : null,
    strategy: technique, timeMs, graph: { nodes, edges }, chunks, steps, answer: "", metrics: pendingMetrics(),
    note: d?.note ? String(d.note) : undefined,
  };
}

const RAGAS: [string, string, boolean][] = [
  ["context_precision", "Context Precision", false],
  ["faithfulness", "Faithfulness", false],
  ["answer_relevancy", "Answer Relevancy", false],
  ["noise_sensitivity", "Noise Sensitivity", true],
  ["context_recall", "Context Recall", true],
  ["context_entities_recall", "Context Entities Recall", true],
  ["recall_at_k", "recall@k", true],
  ["precision_at_k", "precision@k", true],
];
const pendingMetrics = (): Metric[] =>
  RAGAS.map(([key, label, referenceBased]) => ({ key, label, referenceBased, value: 0, pending: true }));

/** Generate (once, cached by the backend) the grounded answer for a retrieval run,
 *  over exactly that run's context. Kept separate from runRetrieval so the graph +
 *  chunks render instantly and this slow LLM call only happens when needed. */
export async function generateAnswer(
  runId: string,
): Promise<{ answer: string; timeMs: number; groundedChunks: number; note?: string }> {
  try {
    const d = await getJson("/api/search/answer", { run_id: runId });
    return {
      answer: String(d.answer ?? ""),
      timeMs: Math.round(Number(d.time_ms ?? 0)),
      groundedChunks: Number(d.grounded_chunks ?? 0),
      note: d.note ? String(d.note) : undefined,
    };
  } catch (err) {
    return { answer: "", timeMs: 0, groundedChunks: 0, note: `answer generation failed: ${(err as Error).message}` };
  }
}

/** Ragas-style metrics for a retrieval run: its retrieved contexts and its answer
 *  (the same one generateAnswer returns). Heavy (several LLM judge calls), so it's
 *  triggered on demand. A metric with no computable value comes back pending (e.g.
 *  reference-based metrics for a query that isn't in the dataset's gold set). */
export async function evaluate(
  runId: string,
): Promise<{ metrics: Metric[]; answer: string; goldMatched: boolean; timeMs: number; note?: string }> {
  try {
    const d = await getJson("/api/search/metrics", { run_id: runId });
    const metrics: Metric[] = (d.metrics ?? []).map((m: any) => ({
      key: String(m.key),
      label: String(m.label),
      referenceBased: !!m.reference_based,
      value: typeof m.value === "number" ? m.value : 0,
      pending: m.value === null || m.value === undefined,
    }));
    return {
      metrics, answer: String(d.answer ?? ""), goldMatched: !!d.gold_matched,
      timeMs: Math.round(Number(d.time_ms ?? 0)), note: d.note ? String(d.note) : undefined,
    };
  } catch (err) {
    return { metrics: [], answer: "", goldMatched: false, timeMs: 0, note: `metrics failed: ${(err as Error).message}` };
  }
}
