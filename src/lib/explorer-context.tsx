import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { listDatasets, runRetrieval, type Dataset, type RetrievalResult, type Technique } from "./api";

/** A set of graph nodes another view asked the Explorer to highlight (e.g. a Prospector document). */
export type Highlight = { label: string; ids: string[] } | null;

type Ctx = {
  datasets: Dataset[];
  datasetId: string;
  dataset: Dataset | undefined;
  setDatasetId: (id: string) => void;
  /** re-fetch /datasets (index status changes while a build runs) */
  refreshDatasets: () => void;
  /** last backend error (unreachable, 4xx/5xx); null when the last call succeeded */
  error: string | null;
  reportError: (msg: string | null) => void;
  highlight: Highlight;
  setHighlight: (h: Highlight) => void;

  // the current retrieval — app-wide so it survives route changes and Compare can reuse it
  query: string;
  run: (q: string) => void;
  runNonce: number;
  technique: Technique;
  setTechnique: (t: Technique) => void;
  chunkTopK: number;
  setChunkTopK: (v: number) => void;
  result: RetrievalResult | null;
  /** show a retrieval produced elsewhere (e.g. a chat chime-in) in the Explorer panels,
   *  without re-running anything; cancels any in-flight one-off retrieval */
  showResult: (r: RetrievalResult) => void;
  /** bumps every time a new result lands */
  resultKey: number;
  loading: boolean;
};

const ExplorerCtx = createContext<Ctx | null>(null);

export function ExplorerProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<Highlight>(null);

  const [query, setQuery] = useState("");
  const [runNonce, setRunNonce] = useState(0);
  const [technique, setTechnique] = useState<Technique>("bm25_vector");
  const [chunkTopK, setChunkTopK] = useState(5);
  const [result, setResult] = useState<RetrievalResult | null>(null);
  const [resultKey, setResultKey] = useState(0);
  const [loading, setLoading] = useState(false);
  const reqId = useRef(0);

  const refreshDatasets = useCallback(() => {
    listDatasets()
      .then((ds) => {
        setDatasets(ds);
        // keep the user's pick if it still exists; otherwise default to the most recently
        // updated ready dataset (a freshly built export, not the old synthetic corpus)
        const ready = ds.filter((d) => d.status === "ready").sort((a, b) => (b.updated ?? "").localeCompare(a.updated ?? ""));
        setDatasetId((cur) => (cur && ds.some((d) => d.id === cur) ? cur : (ready[0] ?? ds[0])?.id ?? ""));
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);
  useEffect(() => { refreshDatasets(); }, [refreshDatasets]);
  useEffect(() => { setResult(null); }, [datasetId]);

  // Re-run whenever the query, technique, knobs or dataset change (sliders debounced).
  useEffect(() => {
    if (!query || !datasetId) return;
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      setLoading(true);
      runRetrieval(datasetId, query, technique, chunkTopK)
        .then((r) => {
          if (id !== reqId.current) return; // superseded by a newer request
          setResult(r);
          setResultKey((k) => k + 1);
          setError(null);
        })
        .catch((e) => {
          if (id !== reqId.current) return;
          setResult(null);
          setError(String(e.message ?? e));
        })
        .finally(() => { if (id === reqId.current) setLoading(false); });
    }, 200);
    return () => window.clearTimeout(t);
  }, [datasetId, query, runNonce, technique, chunkTopK]);

  const showResult = useCallback((r: RetrievalResult) => {
    reqId.current++;            // supersede any in-flight one-off retrieval
    setLoading(false);
    setResult(r);
    setResultKey((k) => k + 1);
  }, []);

  const run = useCallback((q: string) => {
    setHighlight(null);
    setQuery(q);
    setRunNonce((n) => n + 1); // re-submitting the same text still re-runs
  }, []);

  const value = useMemo<Ctx>(() => ({
    datasets, datasetId, setDatasetId, refreshDatasets, error, reportError: setError, highlight, setHighlight,
    dataset: datasets.find((d) => d.id === datasetId),
    query, run, runNonce, technique, setTechnique, chunkTopK, setChunkTopK,
    result, showResult, resultKey, loading,
  }), [datasets, datasetId, refreshDatasets, error, highlight, query, run, runNonce, technique, chunkTopK, result, resultKey, loading]);

  return <ExplorerCtx.Provider value={value}>{children}</ExplorerCtx.Provider>;
}

export function useExplorer() {
  const ctx = useContext(ExplorerCtx);
  if (!ctx) throw new Error("useExplorer must be used inside ExplorerProvider");
  return ctx;
}
