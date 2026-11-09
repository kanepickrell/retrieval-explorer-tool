import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { listDocuments, type DocumentSummary } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import ProspectorTable from "@/components/ProspectorTable";

export default function Prospector() {
  const { datasets, datasetId, dataset, setDatasetId, setHighlight } = useExplorer();
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    let live = true;
    listDocuments(datasetId).then((d) => live && setDocs(d));
    return () => { live = false; };
  }, [datasetId]);

  const showOnGraph = (d: DocumentSummary) => {
    setHighlight({ label: d.title, ids: d.entities });
    navigate("/");
  };

  return (
    <main className="mt-5 flex flex-col gap-4">
      <p className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
        <span className="text-ink">Prospector (data)</span><ArrowRight size={12} />
        <span>Explorer (retrieval)</span><ArrowRight size={12} />
        <span>Metrics (which technique wins)</span>
      </p>
      <section className="card">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="caption">Data sources</h2>
          <span className="font-mono text-[11px] text-muted">select one to set what the Explorer queries</span>
        </div>
        <ProspectorTable datasets={datasets} activeId={datasetId} onSelect={setDatasetId} />
      </section>
      <section className="card">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="caption">Documents in {dataset?.name ?? "…"}</h2>
          <span className="font-mono text-[11px] text-muted">{docs.length} documents</span>
        </div>
        <ul className="grid gap-1.5 md:grid-cols-2">
          {docs.map((d) => (
            <li key={d.id}>
              <button onClick={() => showOnGraph(d)}
                className="flex w-full items-center gap-3 rounded-md border border-line bg-surface-2/60 px-3 py-2 text-left transition-colors hover:border-local/60">
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm">{d.title}</span>
                  <span className="font-mono text-[11px] text-muted">{d.chunks} chunks · {d.entities.length} entities</span>
                </span>
                <span className="shrink-0 font-mono text-[11px] text-local">show on graph →</span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
