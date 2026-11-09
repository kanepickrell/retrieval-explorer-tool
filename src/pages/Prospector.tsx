import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, FileText } from "lucide-react";
import clsx from "clsx";
import { listDocuments, type DocumentSummary } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import ProspectorTable from "@/components/ProspectorTable";

const FLOW = [
  ["Prospector", "data"],
  ["Explorer", "retrieval"],
  ["Metrics", "which technique wins"],
] as const;

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
      <ol className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
        {FLOW.map(([name, role], i) => (
          <li key={name} className="flex items-center gap-2">
            <span className={clsx("rounded-full border px-2.5 py-0.5", i === 0 ? "border-local/60 text-ink" : "border-line text-muted")}>
              {name} <span className="text-muted">({role})</span>
            </span>
            {i < FLOW.length - 1 && <ArrowRight size={12} className="text-muted/60" />}
          </li>
        ))}
      </ol>

      <section className="card">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h2 className="caption">Data sources</h2>
          <Link to="/" className="btn">Query {dataset?.name ?? "dataset"} in Explorer <ArrowRight size={12} /></Link>
        </div>
        <ProspectorTable datasets={datasets} activeId={datasetId} onSelect={setDatasetId} />
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
