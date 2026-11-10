import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { listDocuments, type DocumentSummary } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import ProspectorTable from "@/components/ProspectorTable";
import FlowStepper from "@/components/FlowStepper";
import DocTree from "@/components/DocTree";


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
          <span className="font-mono text-[11px] text-muted">category › subcategory › document › chunks</span>
        </div>
        <DocTree datasetId={datasetId} docs={docs}
          onShow={(label, ids) => { setHighlight({ label, ids }); navigate("/"); }} />
      </section>
    </main>
  );
}
