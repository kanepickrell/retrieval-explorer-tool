import { Link } from "react-router-dom";
import { Loader2, Search } from "lucide-react";
import { SAMPLE_QUERIES } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";

type Props = {
  value: string;
  onChange: (v: string) => void;
  onRun: (q: string) => void;
  loading: boolean;
};

export default function QuestionCard({ value, onChange, onRun, loading }: Props) {
  const { dataset } = useExplorer();
  return (
    <section className="card flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="caption">Question</h2>
        <span className="font-mono text-[11px] text-muted">
          querying <span className="text-ink">{dataset?.name ?? "…"}</span> ·{" "}
          <Link to="/prospector" className="underline decoration-line underline-offset-2 hover:text-ink">change</Link>
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {SAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            onClick={() => { onChange(q); onRun(q); }}
            className="rounded-full border border-line bg-surface-2 px-2.5 py-1 text-left text-xs text-muted transition-colors hover:border-local/60 hover:text-ink"
          >
            {q}
          </button>
        ))}
      </div>
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (value.trim()) onRun(value.trim()); }}>
        <input className="input" placeholder="Ask anything about the documents" value={value} onChange={(e) => onChange(e.target.value)} />
        <button type="submit" className="btn-primary shrink-0" disabled={loading || !value.trim()}>
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />} Search
        </button>
      </form>
    </section>
  );
}
