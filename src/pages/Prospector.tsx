import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, FileText } from "lucide-react";
import { listDocuments, type DocumentSummary } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import ProspectorTable from "@/components/ProspectorTable";
import FlowStepper from "@/components/FlowStepper";


export default function Prospector() {
  const { datasets, datasetId, dataset, setDatasetId, setHighlight, reportError } = useExplorer();
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    let live = true;
    if (!datasetId) return;
    listDocuments(datasetId)
      .then((d) => live && setDocs(d))
      .catch((e) => { if (live) { setDocs([]); reportError(String(e.message ?? e)); } });
    return () => { live = false; };
  }, [datasetId]);

  const showOnGraph = (d: DocumentSummary) => {
    setHighlight({ label: d.title, ids: d.entities });
    navigate("/");
  };

  return (
    <main className="mt-5 flex flex-col gap-4">
      <FlowStepper active={0} />

      <section className="card">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h2 className="caption">Data sources</h2>
          <Link to="/" className="btn">Query {dataset?.name ?? "dataset"} in Explorer <ArrowRight size={12} /></Link>
        </div>
        <ProspectorTable datasets={datasets} activeId={datasetId}
          onSelect={(id) => { if (datasets.find((d) => d.id === id)?.status === "ready") setDatasetId(id); }} />
      </section>

      <section className="card">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="caption">Documents · {dataset?.name ?? "…"}</h2>
          <span className="font-mono text-[11px] text-muted">{docs.length} documents · click to highlight on the graph</span>
        </div>
        <ul className="grid gap-1.5 md:grid-cols-2">
          {docs.map((d) => (
            <li key={d.id}>
              <button onClick={() => showOnGraph(d)}
                className="group flex w-full items-center gap-3 rounded-md border border-line bg-surface-2/50 px-3 py-2 text-left transition-colors hover:border-local/50 hover:bg-surface-2">
                <FileText size={15} className="shrink-0 text-muted" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm">{d.title}</span>
                  <span className="font-mono text-[11px] text-muted">{d.chunks} chunks · {d.entities.length} entities</span>
                </span>
                <ArrowRight size={14} className="shrink-0 text-local opacity-0 transition-opacity group-hover:opacity-100" />
              </button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
