import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGraph, runRetrieval, type Chunk, type GraphEdge, type GraphNode, type RetrievalResult, type Technique } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import { deriveSignals } from "@/lib/signals";
import QuestionCard from "@/components/QuestionCard";
import ModeCard from "@/components/ModeCard";
import GraphCanvas from "@/components/GraphCanvas";
import InspectorTabs from "@/components/InspectorTabs";

const EMPTY = { nodes: [] as GraphNode[], edges: [] as GraphEdge[] };

export default function Explorer() {
  const { datasetId, dataset, highlight, setHighlight } = useExplorer();

  const [graph, setGraph] = useState(EMPTY);
  const [input, setInput] = useState("");
  const [query, setQuery] = useState("");
  const [runNonce, setRunNonce] = useState(0);
  const [technique, setTechnique] = useState<Technique>("mix");
  const [topK, setTopK] = useState(10);
  const [chunkTopK, setChunkTopK] = useState(5);
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [resultKey, setResultKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("steps");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeChunk, setActiveChunk] = useState<Chunk | null>(null);
  const reqId = useRef(0);

  useEffect(() => {
    let live = true;
    setResult(null);
    getGraph(datasetId).then((g) => live && setGraph(g));
    return () => { live = false; };
  }, [datasetId]);

  // Re-run whenever the query, technique or knobs change (sliders debounced).
  useEffect(() => {
    if (!query) return;
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      setLoading(true);
      runRetrieval(datasetId, query, technique, topK, chunkTopK).then((r) => {
        if (id !== reqId.current) return; // a newer request superseded this one
        setResult(r);
        setResultKey((k) => k + 1);
        setActiveChunk(null);
        setLoading(false);
      });
    }, 200);
    return () => window.clearTimeout(t);
  }, [datasetId, query, runNonce, technique, topK, chunkTopK]);

  const run = useCallback((q: string) => {
    setHighlight(null);
    setQuery(q);
    setRunNonce((n) => n + 1); // re-submitting the same text still re-runs
  }, [setHighlight]);

  const signals = useMemo(() => deriveSignals(result), [result]);

  const chunkHighlight = useMemo(
    () => (activeChunk ? { label: activeChunk.title, ids: activeChunk.entities ?? [] } : null),
    [activeChunk],
  );

  return (
    <main className="mt-5 flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <QuestionCard value={input} onChange={setInput} onRun={run} loading={loading} />
        <ModeCard technique={technique} onTechnique={setTechnique} topK={topK} onTopK={setTopK} chunkTopK={chunkTopK} onChunkTopK={setChunkTopK} />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:h-[700px] lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="h-[560px] lg:h-full">
          <GraphCanvas
            graph={graph}
            signals={signals}
            resultKey={resultKey}
            highlight={chunkHighlight ?? highlight}
            onClearHighlight={() => { setActiveChunk(null); setHighlight(null); }}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>
        <div className="min-h-[560px] lg:h-full lg:min-h-0">
          <InspectorTabs
            result={result}
            loading={loading}
            labeled={!!dataset?.labeled}
            tab={tab}
            onTab={setTab}
            activeChunkId={activeChunk?.id ?? null}
            onChunk={setActiveChunk}
          />
        </div>
      </div>
    </main>
  );
}
