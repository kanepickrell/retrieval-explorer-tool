import { useEffect, useState } from "react";
import { Database, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { AnswerMeta } from "@/lib/api";

type Props = { answer: string; meta?: AnswerMeta; loading?: boolean; chunkCount: number; onShowContext: () => void };

/** Streams the answer in progressively (typewriter) once it arrives. While the
 *  backend is generating, shows a generating state. */
export default function AnswerPanel({ answer, meta, loading, chunkCount, onShowContext }: Props) {
  const [shown, setShown] = useState(0);
  const [showLookup, setShowLookup] = useState(false);
  useEffect(() => {
    setShown(0);
    if (!answer) return;
    const id = window.setInterval(() => {
      setShown((n) => {
        if (n >= answer.length) { window.clearInterval(id); return n; }
        return Math.min(answer.length, n + 3);
      });
    }, 16);
    return () => window.clearInterval(id);
  }, [answer]);

  if (loading && !answer) {
    return (
      <div className="flex items-center gap-2 py-6 font-mono text-xs text-muted">
        <Loader2 size={14} className="animate-spin text-local" />
        Checking the graph, then answering…
      </div>
    );
  }
  if (!answer) {
    return <p className="py-6 text-sm text-muted">No answer was generated for this query.</p>;
  }

  const done = shown >= answer.length;
  const g = meta?.route === "graph" ? meta.graph : null;
  return (
    <div>
      {g && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded border border-good/50 bg-good/10 px-1.5 py-px font-mono text-[10.5px] text-good"
            title="Answered from an exact lookup on the metadata graph, not from the retrieved text">
            <Database size={11} /> from the graph · exact
          </span>
          <span className="font-mono text-[11px] text-muted">{g.query}</span>
        </div>
      )}
      <div className={done ? "answer-md text-sm" : "answer-md caret text-sm"}>
        <ReactMarkdown>{answer.slice(0, shown)}</ReactMarkdown>
      </div>
      {done && g && (
        <div className="mt-2">
          <button onClick={() => setShowLookup(!showLookup)}
            className="font-mono text-[11px] text-muted/80 underline decoration-line underline-offset-2 hover:text-ink">
            {showLookup ? "hide" : "show"} the lookup result the answer was written from →
          </button>
          {showLookup && (
            <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap rounded-md border border-line bg-bg p-2 font-mono text-[11px] text-ink">
              {g.text}
            </pre>
          )}
        </div>
      )}
      {done && !g && (
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <button onClick={onShowContext} className="font-mono text-[11px] text-muted/80 underline decoration-line underline-offset-2 hover:text-ink">
            grounded in {chunkCount} chunk{chunkCount === 1 ? "" : "s"} →
          </button>
          {meta?.graph?.skipped && (
            <span className="font-mono text-[11px] text-muted/70" title="The graph lookup was tried first">
              graph lookup skipped: {meta.graph.skipped}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
