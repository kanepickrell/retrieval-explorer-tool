import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import { ArrowLeft, FileText } from "lucide-react";
import { getDocSource, type DocSource } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";

const KIND: Record<string, string> = {
  chat: "Past chat", file: "Uploaded file", agent: "Agent Hub agent", step: "Workflow step",
  tool: "Agent Hub tool", catalog: "Catalog", person: "User activity",
};

/**
 * /doc?doc=<id>&dataset=<id>: one indexed document, readable. Ping's cards link here,
 * so a surfaced chat opens as the exported copy instead of a chat the viewer can't access.
 */
export default function Doc() {
  const [params] = useSearchParams();
  const doc = params.get("doc") ?? "";
  const { datasetId } = useExplorer();
  const dataset = params.get("dataset") || datasetId;
  const [src, setSrc] = useState<DocSource | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!doc || !dataset) return;
    setSrc(null); setErr(null);
    getDocSource(dataset, doc).then(setSrc).catch((e) => setErr(String(e.message ?? e)));
  }, [doc, dataset]);

  useEffect(() => { if (src) document.title = `${src.title} · Retrieval Explorer`; }, [src]);

  return (
    <main className="mx-auto mt-5 max-w-[900px]">
      <Link to="/live" className="mb-3 inline-flex items-center gap-1 font-mono text-[11px] text-muted hover:text-ink">
        <ArrowLeft size={12} /> Live chime-ins
      </Link>
      {!doc && <div className="card text-sm text-muted">No document given.</div>}
      {err && <div className="card text-sm text-red-400">Couldn't load {doc}: {err}</div>}
      {!src && !err && doc && <div className="card text-sm text-muted">Loading {doc}…</div>}
      {src && (
        <article className="card">
          <div className="mb-1 flex items-center gap-2 caption">
            <FileText size={12} /> {KIND[src.kind] ?? "Document"} · {src.dataset}
          </div>
          <h2 className="font-title text-xl font-semibold">{src.title}</h2>
          <div className="mb-4 font-mono text-[11px] text-muted">{src.doc}</div>
          <div className="answer-md doc-md text-sm">
            <ReactMarkdown>{src.markdown.replace(/^#\s+.*\n/, "").replace(/\$\\(?:right|long)?arrow\$|\$\\to\$/gi, "→")}</ReactMarkdown>
          </div>
        </article>
      )}
    </main>
  );
}
