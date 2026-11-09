import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

/** Streams the answer in progressively (typewriter). A real backend can feed tokens into the same state. */
export default function AnswerPanel({ answer, chunkCount }: { answer: string; chunkCount: number }) {
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
      {done && <p className="mt-3 font-mono text-[11px] text-muted/70">grounded in {chunkCount} chunk{chunkCount === 1 ? "" : "s"}</p>}
    </div>
  );
}
