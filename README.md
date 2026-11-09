# Retrieval Explorer

A dark, card-based lab for running a query through different RAG retrieval
techniques (`vector · local · global · hybrid · mix`) and watching what each one
retrieves over a knowledge graph: the touched subgraph, the chunks, the step
trace, the streamed answer and Ragas-style metrics.

Standalone prototype for ProtoGraph. It runs entirely on mock data for now.

```bash
npm install
npm run dev      # http://localhost:5180
npm run build    # typecheck + production build
```

## Routes
- `/`: Explorer (question, mode, graph, Steps / Context / Answer, metrics)
- `/prospector`: data-source browser. Selecting a dataset sets what the Explorer
  queries, and "show on graph" highlights a document's entities.

## The backend seam
**`src/lib/api.ts` is the only data source.** To swap in a real backend, replace
the bodies of `getGraph`, `runRetrieval`, `listDatasets` and `listDocuments`.
Keep the types and signatures as they are. Nothing else needs to change.

Optional fields beyond the original contract (the UI degrades gracefully without them):

| field | used for |
|---|---|
| `Chunk.entities` | click a chunk → highlight its entities; vector / final-context node colours |
| `Dataset.labeled` | reference-based metrics show a value only on labeled datasets, otherwise "needs gold" |
| `listDocuments()` | Prospector document list |

Signal colours are derived from a `RetrievalResult` in `src/lib/signals.ts`:
- teal = entity hit (in the touched subgraph)
- amber edge = relation hit
- violet = entity of a vector-retrieved chunk
- white = in final context (subgraph ∩ chunk-grounded)

## Graph
- **Nebula:** one glowing orb per cluster, drawn as 2D SVG with radial gradients and a blurred bloom.
- **Detail:** clicking an orb springs its members out into a d3-force layout. Running a query expands every cluster.
- **Find a node:** isolates the node's 1-hop neighbourhood.
- **Semantic zoom:** level-of-detail is a TODO in `GraphCanvas.tsx`.

## Stack
React 18, Vite 5, TypeScript, Tailwind 3, Radix (tabs and slider), d3-force and
d3-zoom, react-markdown.
