import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

type Props = { answer: string; loading?: boolean; chunkCount: number; onShowContext: () => void };

/** Streams the answer in progressively (typewriter) once it arrives. While the
 *  backend is generating, shows a generating state. */
export default function AnswerPanel({ answer, loading, chunkCount, onShowContext }: Props) {
  const [shown, setShown] = useState(0);
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
        Generating answer from the retrieved context…
      </div>
    );
  }
  if (!answer) {
    return <p className="py-6 text-sm text-muted">No answer was generated for this query.</p>;
  }

  const done = shown >= answer.length;
  return (
    <div>
      <div className={done ? "answer-md text-sm" : "answer-md caret text-sm"}>
        <ReactMarkdown>{answer.slice(0, shown)}</ReactMarkdown>
      </div>
      {done && (
        <button onClick={onShowContext} className="mt-2 font-mono text-[11px] text-muted/80 underline decoration-line underline-offset-2 hover:text-ink">
          grounded in {chunkCount} chunk{chunkCount === 1 ? "" : "s"} →
        </button>
      )}
    </div>
  );
}
