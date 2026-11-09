import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

type Props = { answer: string; chunkCount: number; onShowContext: () => void };

/** Streams the answer in progressively (typewriter). A real backend can feed tokens into the same state. */
export default function AnswerPanel({ answer, chunkCount, onShowContext }: Props) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    setShown(0);
    const id = window.setInterval(() => {
      setShown((n) => {
        if (n >= answer.length) { window.clearInterval(id); return n; }
        return Math.min(answer.length, n + 3);
      });
    }, 16);
    return () => window.clearInterval(id);
  }, [answer]);

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
