import type { RefObject } from "react";
import { Loader2, Search } from "lucide-react";
import clsx from "clsx";

type Props = {
  value: string;
  onChange: (v: string) => void;
  onRun: (q: string) => void;
  loading: boolean;
  /** the query that produced the current result, to mark its chip */
  activeQuery: string;
  inputRef?: RefObject<HTMLInputElement>;
  /** example queries from the dataset's gold set */
  samples?: string[];
};

export default function QuestionCard({ value, onChange, onRun, loading, activeQuery, inputRef, samples = [] }: Props) {
  return (
    <section className="card flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <h2 className="caption">Question</h2>
        <kbd className="font-mono text-[10px] text-muted/60">press /</kbd>
      </div>
      <form className="relative" onSubmit={(e) => { e.preventDefault(); if (value.trim()) onRun(value.trim()); }}>
        <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
        <input ref={inputRef} className="input h-11 pl-9 pr-28 text-[15px]" placeholder="Ask anything about the documents"
          value={value} onChange={(e) => onChange(e.target.value)} aria-label="Question" />
        <button type="submit" className="btn-primary absolute right-1.5 top-1/2 h-8 -translate-y-1/2 px-3" disabled={loading || !value.trim()}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : null} Search
        </button>
      </form>
      {samples.length > 0 && <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-0.5 font-mono text-[11px] text-muted/70" title="queries from this dataset's gold eval set">try (gold)</span>
        {samples.map((q) => (
          <button key={q} onClick={() => { onChange(q); onRun(q); }}
            className={clsx("rounded-full border px-2.5 py-0.5 text-left text-xs transition-colors",
              q === activeQuery ? "border-local/60 bg-local/10 text-ink" : "border-line bg-surface-2 text-muted hover:border-muted/50 hover:text-ink")}>
            {q}
          </button>
        ))}
      </div>}
    </section>
  );
}
