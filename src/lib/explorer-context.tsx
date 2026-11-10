import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { listDatasets, runRetrieval, type Dataset, type RetrievalResult, type Technique } from "./api";

/** A set of graph nodes another view asked the Explorer to highlight (e.g. a Prospector document). */
export type Highlight = { label: string; ids: string[] } | null;

type Ctx = {
  datasets: Dataset[];
  datasetId: string;
  dataset: Dataset | undefined;
  setDatasetId: (id: string) => void;
  highlight: Highlight;
  setHighlight: (h: Highlight) => void;

  // the current retrieval — app-wide so it survives route changes and Compare can reuse it
  query: string;
  run: (q: string) => void;
  runNonce: number;
  technique: Technique;
  setTechnique: (t: Technique) => void;
  topK: number;
  setTopK: (v: number) => void;
  chunkTopK: number;
  setChunkTopK: (v: number) => void;
  result: RetrievalResult | null;
  /** bumps every time a new result lands */
  resultKey: number;
  loading: boolean;
};

const ExplorerCtx = createContext<Ctx | null>(null);

export function ExplorerProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("ds-chat");
  const [highlight, setHighlight] = useState<Highlight>(null);

  const [query, setQuery] = useState("");
  const [runNonce, setRunNonce] = useState(0);
  const [technique, setTechnique] = useState<Technique>("mix");
  const [topK, setTopK] = useState(10);
  const [chunkTopK, setChunkTopK] = useState(5);
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [resultKey, setResultKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const reqId = useRef(0);

  useEffect(() => { listDatasets().then(setDatasets); }, []);
  useEffect(() => { setResult(null); }, [datasetId]);

  // Re-run whenever the query, technique, knobs or dataset change (sliders debounced).
  useEffect(() => {
    if (!query) return;
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      setLoading(true);
      runRetrieval(datasetId, query, technique, topK, chunkTopK).then((r) => {
        if (id !== reqId.current) return; // superseded by a newer request
        setResult(r);
        setResultKey((k) => k + 1);
        setLoading(false);
      });
    }, 200);
    return () => window.clearTimeout(t);
  }, [datasetId, query, runNonce, technique, topK, chunkTopK]);

  const run = useCallback((q: string) => {
    setHighlight(null);
    setQuery(q);
    setRunNonce((n) => n + 1); // re-submitting the same text still re-runs
  }, []);

  const value = useMemo<Ctx>(() => ({
    datasets, datasetId, setDatasetId, highlight, setHighlight,
    dataset: datasets.find((d) => d.id === datasetId),
    query, run, runNonce, technique, setTechnique, topK, setTopK, chunkTopK, setChunkTopK,
    result, resultKey, loading,
  }), [datasets, datasetId, highlight, query, run, runNonce, technique, topK, chunkTopK, result, resultKey, loading]);

  return <ExplorerCtx.Provider value={value}>{children}</ExplorerCtx.Provider>;
}

export function useExplorer() {
  const ctx = useContext(ExplorerCtx);
  if (!ctx) throw new Error("useExplorer must be used inside ExplorerProvider");
  return ctx;
}
