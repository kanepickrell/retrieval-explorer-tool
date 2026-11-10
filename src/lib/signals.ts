import type { RetrievalResult } from "./api";

/**
 * How a node/edge participated in a retrieval, derived from a RetrievalResult:
 *  - entity   : node in the touched subgraph (entity search hit)
 *  - relation : edge in the touched subgraph (relation search hit)
 *  - vector   : node mentioned by a chunk that came from vector search
 *  - context  : node that survived into the final context — in the touched
 *               subgraph AND grounded by a retrieved chunk (graph modes only;
 *               vector mode has no graph, so its nodes stay "vector")
 */
export type Signals = {
  entity: Set<string>;
  relation: Set<string>;
  vector: Set<string>;
  context: Set<string>;
};

export function deriveSignals(r: RetrievalResult | null): Signals | null {
  if (!r) return null;
  const entity = new Set(r.graph.nodes.map((n) => n.id));
  const relation = new Set(r.graph.edges.map((e) => e.id));
  const vector = new Set<string>();
  const chunkEntities = new Set<string>();
  for (const c of r.chunks) for (const id of c.entities ?? []) {
    chunkEntities.add(id);
    if (c.source === "vector") vector.add(id);
  }
  const context = new Set<string>();
  if (r.strategy !== "vector") chunkEntities.forEach((id) => entity.has(id) && context.add(id));
  return { entity, relation, vector, context };
}

export const SOURCE_COLOR: Record<string, string> = {
  vector: "var(--vector)",
  bm25: "var(--lexical)",
  bm25_vector: "var(--lexical)",
  local: "var(--local)",
  graph: "var(--local)",
  global: "var(--global)",
  hybrid: "var(--global)",
};

export const CLUSTER_COLOR: Record<string, string> = {
  person: "#35c0be",
  organization: "#e0a73a",
  tool: "#6fa8dc",
  technique: "#a58ff1",
  method: "#5cc27a",
  model: "#e8736b",
};
export const clusterColor = (c: string) => CLUSTER_COLOR[c] ?? "#93a0b0";
