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

export type Technique = "vector" | "bm25" | "bm25_vector" | "local" | "global" | "hybrid" | "mix" | "agent";
export type GraphNode = { id: string; label: string; cluster: string; type: string; size: number; description?: string };
export type GraphEdge = { id: string; source: string; target: string; weight: number; label?: string };
export type Step = { label: string; detail: string; stat?: string; ms?: number };
/** One retrieved document (its best/first chunk). score = cosine(query, chunk) with the
 *  same embeddings for every technique (null if the chunk has no embedding); `via` is how
 *  it was reached: E = entity, R = relation (LightRAG), C = vector search, B = BM25
 *  keyword match, B+C = found by both BM25 and vector (fusion). */
export type Chunk = {
  id: string; title: string; source: string; score: number | null; rank: number;
  via?: "E" | "R" | "C" | "B" | "B+C"; snippet?: string; entities?: string[];
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
export type GoldRef = { id: string; query: string; type: string };
export type DocumentSummary = {
  id: string; title: string; chunks: number; entities: string[];
  /** browse hierarchy: category › subcategory › document › chunks */
  category: string; subcategory: string;
  /** gold eval queries whose answer should come from this doc */
  gold: GoldRef[];
};
export type DocChunk = { id: string; order: number; tokens: number; content: string; entities: string[] };

export const TECHNIQUES: Technique[] = ["vector", "bm25", "bm25_vector", "local", "global", "hybrid", "mix", "agent"];
/** display names (ids are URL-safe) */
export const TECHNIQUE_LABEL: Record<Technique, string> = {
  vector: "vector", bm25: "bm25", bm25_vector: "bm25+vector",
  local: "local", global: "global", hybrid: "hybrid", mix: "mix", agent: "agent",
};

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
    category: String(x.category ?? "Other"), subcategory: String(x.subcategory ?? "Other"),
    gold: (x.gold ?? []).map((g: any) => ({ id: String(g.id), query: String(g.query), type: String(g.type ?? "") })),
  }));
}

export async function listDocChunks(datasetId: string, docId: string): Promise<DocChunk[]> {
  const d = await getJson("/documents/chunks", { dataset: datasetId, doc: docId });
  return (d ?? []).map((c: any) => ({
    id: String(c.id), order: Number(c.order ?? 0), tokens: Number(c.tokens ?? 0),
    content: String(c.content ?? ""), entities: (c.entities ?? []).map(String),
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

// ───────────────────────── Assist (chime-in) ─────────────────────────

export type ChatTurn = { role: "user" | "assistant"; content: string };
export type QueryStrategy = "last" | "window" | "rewrite";
export type ReplayChat = { chatId: string; doc: string; title: string; author: string | null; userTurns: number; turns: ChatTurn[] };
export type AssistSuggestion = {
  doc: string; chunkId: string | null; title: string; snippet: string;
  score: number | null; rank: number | null; via?: string;
  /** LLM usefulness verdict (undefined when the LLM check is off) */
  useful?: boolean | null; why?: string;
};
export type AssistStage = { label: string; detail: string; stat: string; ms: number };
export type AssistResult = {
  query: string | null; stayQuiet: boolean; reason: string | null;
  suggestions: AssistSuggestion[]; candidates: AssistSuggestion[];
  stages: AssistStage[]; totalMs: number;
  /** the retrieval behind this chime-in, as a normal run (opens in the Explorer panels) */
  retrieval: RetrievalResult | null;
};

export async function listReplayChats(datasetId: string): Promise<ReplayChat[]> {
  const d = await getJson("/api/assist/chats", { dataset: datasetId });
  return (d ?? []).map((c: any) => ({
    chatId: String(c.chat_id), doc: String(c.doc), title: String(c.title), author: c.author ?? null,
    userTurns: Number(c.user_turns ?? 0), turns: (c.turns ?? []) as ChatTurn[],
  }));
}

const mapSuggestion = (x: any): AssistSuggestion => ({
  doc: String(x.doc), chunkId: x.chunk_id ?? null, title: String(x.title ?? x.doc), snippet: String(x.snippet ?? ""),
  score: typeof x.score === "number" ? x.score : null, rank: typeof x.rank === "number" ? x.rank : null,
  via: x.via ?? undefined, useful: x.useful, why: x.why ? String(x.why) : undefined,
});

export async function assist(opts: {
  datasetId: string; turns: ChatTurn[]; technique: Technique; queryStrategy: QueryStrategy;
  limit: number; topK: number; llmFilter: boolean; minScore: number;
  excludeDocs: string[]; alreadySuggested: string[];
}): Promise<AssistResult> {
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/assist`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: opts.datasetId, turns: opts.turns, technique: opts.technique, query_strategy: opts.queryStrategy,
        limit: opts.limit, top_k: opts.topK, llm_filter: opts.llmFilter, min_score: opts.minScore,
        exclude_docs: opts.excludeDocs, already_suggested: opts.alreadySuggested,
      }),
    });
  } catch {
    throw new ApiError(`backend unreachable at ${API_BASE}`);
  }
  if (!r.ok) throw new ApiError(`/api/assist → ${r.status}`);
  return mapAssist(await r.json(), opts.technique, opts.topK, opts.limit);
}

function mapAssist(d: any, technique: Technique, topK: number, limit: number): AssistResult {
  return {
    query: d.query ?? null, stayQuiet: !!d.stay_quiet, reason: d.reason ?? null,
    suggestions: (d.suggestions ?? []).map(mapSuggestion), candidates: (d.candidates ?? []).map(mapSuggestion),
    stages: d.stages ?? [], totalMs: Math.round(Number(d.total_ms ?? 0)),
    retrieval: d.retrieval ? mapResult(d.retrieval, technique, topK, limit) : null,
  };
}

/** One /api/assist call as the backend logged it (see server/live.py). */
export type LiveEvent = {
  seq: number; at: number;
  /** "open-webui" = a real Chime In from Open WebUI; "explorer" = this app's chat replay */
  source: "open-webui" | "explorer" | "other";
  dataset: string | null; technique: Technique; queryStrategy: QueryStrategy;
  llmFilter: boolean; minScore: number;
  turns: ChatTurn[]; error: string | null; result: AssistResult | null;
};

export async function listLive(after: number): Promise<{ latest: number; events: LiveEvent[] }> {
  const d = await getJson("/api/live", { after });
  return {
    latest: Number(d.latest ?? 0),
    events: (d.events ?? []).map((e: any): LiveEvent => ({
      seq: Number(e.seq), at: Number(e.at), source: e.source ?? "other", dataset: e.dataset ?? null,
      technique: e.technique, queryStrategy: e.query_strategy, llmFilter: !!e.llm_filter, minScore: Number(e.min_score),
      turns: e.turns ?? [], error: e.error ?? null,
      result: e.error ? null : mapAssist(e.response ?? {}, e.technique, Number(e.top_k ?? 0), Number(e.limit ?? 5)),
    })),
  };
}

/** One document's full source markdown (what Ping links to). */
export type DocSource = { dataset: string; doc: string; kind: string; title: string; markdown: string };
export async function getDocSource(datasetId: string, doc: string): Promise<DocSource> {
  return getJson("/documents/source", { dataset: datasetId, doc });
}


// ───────────────────────── Routing agent (server/agent_loop.py) ─────────────────────────

/** One step of the agent loop, as it happened: goal, hypothesis, validate, revise, route,
 *  search, fallback, evidence, check, retry, result, waiting, error. */
export type AgentEvent = {
  i: number; ms: number; tookMs: number; kind: string; title: string; detail: string; data: any;
};
export type AgentStatus = "running" | "done" | "waiting" | "error";
export type AgentRunSummary = {
  id: string; at: number; source: string; dataset: string; status: AgentStatus;
  query: string; goal: string | null; route: string | null; steps: number; ms: number;
  /** set when the run's question is in a gold set: the expected docs and, once finished,
   *  whether the top 5 reached any of them */
  gold: AgentGold | null;
};
export type AgentGold = { id: string; type: string; goldIds: string[]; hit5: boolean | null; reached: string[]; nodes?: string[] };
const mapGold = (g: any): AgentGold | null => (g ? {
  id: String(g.id ?? ""), type: String(g.type ?? ""), goldIds: g.gold_ids ?? [], hit5: g.hit5 ?? null,
  reached: g.reached ?? [], nodes: g.nodes ?? undefined,
} : null);
export type AgentDoc = { doc: string; title: string; score: number | null; via: string; chunkId: string | null; text: string };
export type AgentRunDetail = AgentRunSummary & {
  turns: ChatTurn[]; error: string | null; events: AgentEvent[];
  result: { route: string | null; answer: string | null; docs: AgentDoc[]; check?: any; waiting?: string; error?: string } | null;
};

const mapRunSummary = (r: any): AgentRunSummary => ({
  id: String(r.id), at: Number(r.at), source: String(r.source ?? "other"), dataset: String(r.dataset ?? ""),
  status: r.status, query: String(r.query ?? ""), goal: r.goal ?? null, route: r.route ?? null,
  steps: Number(r.events ?? 0), ms: Number(r.ms ?? 0), gold: mapGold(r.gold),
});

export async function startAgent(opts: {
  datasetId: string; turns?: ChatTurn[]; query?: string; requireClear?: boolean; excludeDocs?: string[];
}): Promise<string> {
  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/agent/run`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: opts.datasetId, turns: opts.turns ?? [], query: opts.query ?? null, source: "explorer",
        require_clear: opts.requireClear ?? true, exclude_docs: opts.excludeDocs ?? [],
      }),
    });
  } catch {
    throw new ApiError(`backend unreachable at ${API_BASE}`);
  }
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json())?.detail ?? ""; } catch { /* non-JSON error body */ }
    throw new ApiError(`/api/agent/run → ${r.status}${detail ? `: ${detail}` : ""}`);
  }
  return String((await r.json()).id);
}

export async function listAgentRuns(): Promise<AgentRunSummary[]> {
  return ((await getJson("/api/agent/runs")) ?? []).map(mapRunSummary);
}

export async function getAgentRun(id: string, after = 0): Promise<AgentRunDetail> {
  const d = await getJson(`/api/agent/runs/${encodeURIComponent(id)}`, { after });
  return {
    ...mapRunSummary(d), turns: d.turns ?? [], error: d.error ?? null,
    events: (d.events ?? []).map((e: any): AgentEvent => ({
      i: Number(e.i), ms: Number(e.ms), tookMs: Number(e.took_ms ?? 0), kind: String(e.kind),
      title: String(e.title ?? ""), detail: String(e.detail ?? ""), data: e.data ?? {},
    })),
    result: d.result ? {
      route: d.result.route ?? null, answer: d.result.answer ?? null, check: d.result.check,
      waiting: d.result.waiting, error: d.result.error,
      docs: (d.result.docs ?? []).map((x: any): AgentDoc => ({
        doc: String(x.doc), title: String(x.title ?? x.doc), score: typeof x.score === "number" ? x.score : null,
        via: String(x.via ?? ""), chunkId: x.chunk_id ?? null, text: String(x.text ?? ""),
      })),
    } : null,
  };
}

/** A run as a traversal graph: real records (with their graph edges) plus the agent's
 *  reasoning as virtual nodes (question, hypotheses, terms, missing/dropped names), and
 *  each step's role changes so the page can replay it. See agent_loop.traversal(). */
export type TraversalNode = {
  id: string; label: string; type: string; doc: string | null; virtual: boolean;
  claim?: string; answer_type?: string; target?: string;
};
export type TraversalEdge = { source: string; target: string; kind: string; label: string };
export type TraversalStep = { i: number; kind: string; roles: Record<string, string> };
export type AgentTraversal = {
  id: string; status: AgentStatus; nodes: TraversalNode[]; edges: TraversalEdge[];
  steps: TraversalStep[]; gold: AgentGold | null;
};

export async function getAgentGraph(id: string): Promise<AgentTraversal> {
  const d = await getJson(`/api/agent/runs/${encodeURIComponent(id)}/graph`);
  return { id: String(d.id), status: d.status, nodes: d.nodes ?? [], edges: d.edges ?? [], steps: d.steps ?? [],
           gold: mapGold(d.gold) };
}

// ───────────────────────── Ping cycle (server/ping_cycle.py) ─────────────────────────

export const PING_STAGES = ["observe", "update", "check", "probe", "gate", "dive", "suggest", "learn"] as const;
export type PingStageName = (typeof PING_STAGES)[number];
export type PingStage = { status: "pending" | "done" | "skip"; title?: string; detail?: string; data?: any; ms?: number };
export type PingHypothesis = {
  id: string; kind: "artifact" | "insight" | "idea"; claim: string; answer_type: string; target: string;
  scope: Record<string, string[]>; query: string; confidence: number; status: "active" | "retired"; born: number;
  confirmed: boolean; l1: { verdict: string; why: string; scope: number | null } | null;
  history: { turn: number; delta: number; source: string; why: string; conf: number }[];
};
export type PingProbe = { id: string; text: string; tests: string | null; turn: number; used: boolean | null; sim: number | null };
export type PingFocus = {
  top?: string | null;
  hyps?: Record<string, { conf: number; seeds: string[]; scope: string[]; hits: string[]; verdict: string; status: string }>;
  evidence?: string[]; suggestion?: string[];
};
export type PingSuggestion = { kind: string; doc: string; title: string; why: string; hypothesis: string; verdict: string };
export type PingCycle = {
  turn: number; user: string; reply: string | null; stages: Record<PingStageName, PingStage>;
  focus: PingFocus; snapshot?: { id: string; kind: string; claim: string; conf: number; status: string; l1: string | null }[];
  suggestion?: PingSuggestion | null; error?: string;
  /** replay ground truth, filled in when the next real user message arrives */
  eval: { next_user?: string; top_supported?: boolean; top_effect?: string; probe_used?: string | null };
};
export type PingSessionSummary = {
  id: string; at: number; source: string; dataset: string; status: "idle" | "running" | "done" | "error";
  title: string; author: string | null; turn: number; turns_total: number | null;
  probes_offered: number; probes_used: number; top_supported: number; top_scored: number; suggestions: number;
};
export type PingSession = PingSessionSummary & {
  error: string | null; chat: { chat_id: string; doc: string; title: string; author: string | null } | null;
  turns: ChatTurn[]; cycles: PingCycle[];
  profile: { goal: string; stage: string; facts: { who: string; what: string; why: string };
             hypotheses: PingHypothesis[]; probes: PingProbe[]; suggested: string[] };
};
export type PingChat = { chat_id: string; doc: string; title: string; author: string | null; user_turns: number };

async function sendJson(method: string, path: string, body?: any): Promise<any> {
  let r: Response;
  try {
    r = await fetch(`${API_BASE}${path}`, {
      method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body),
    });
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

export const listPingChats = (datasetId: string): Promise<PingChat[]> => getJson("/api/ping/chats", { dataset: datasetId });
export const listPingSessions = (): Promise<PingSessionSummary[]> => getJson("/api/ping/sessions");
export const getPingSession = (id: string): Promise<PingSession> => getJson(`/api/ping/sessions/${encodeURIComponent(id)}`);
export const newPingSession = (datasetId: string, chatId?: string): Promise<PingSession> =>
  sendJson("POST", "/api/ping/sessions", { dataset: datasetId, chat_id: chatId ?? null });
export const pingNext = (id: string) => sendJson("POST", `/api/ping/sessions/${encodeURIComponent(id)}/next`);
export const pingSay = (id: string, message: string) => sendJson("POST", `/api/ping/sessions/${encodeURIComponent(id)}/say`, { message });
export const resetPing = () => sendJson("DELETE", "/api/ping/sessions");
