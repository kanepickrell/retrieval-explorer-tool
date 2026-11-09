import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { listDatasets, type Dataset } from "./api";

/** A set of graph nodes another view asked the Explorer to highlight (e.g. a Prospector document). */
export type Highlight = { label: string; ids: string[] } | null;

type Ctx = {
  datasets: Dataset[];
  datasetId: string;
  dataset: Dataset | undefined;
  setDatasetId: (id: string) => void;
  highlight: Highlight;
  setHighlight: (h: Highlight) => void;
};

const ExplorerCtx = createContext<Ctx | null>(null);

export function ExplorerProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("ds-chat");
  const [highlight, setHighlight] = useState<Highlight>(null);

  useEffect(() => { listDatasets().then(setDatasets); }, []);

  const value = useMemo<Ctx>(() => ({
    datasets, datasetId, setDatasetId, highlight, setHighlight,
    dataset: datasets.find((d) => d.id === datasetId),
  }), [datasets, datasetId, highlight]);

  return <ExplorerCtx.Provider value={value}>{children}</ExplorerCtx.Provider>;
}

export function useExplorer() {
  const ctx = useContext(ExplorerCtx);
  if (!ctx) throw new Error("useExplorer must be used inside ExplorerProvider");
  return ctx;
}
