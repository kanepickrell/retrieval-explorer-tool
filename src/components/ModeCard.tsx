import * as Slider from "@radix-ui/react-slider";
import clsx from "clsx";
import { Link } from "react-router-dom";
import { BarChart3 } from "lucide-react";
import { TECHNIQUE_LABEL, TECHNIQUES, type Technique } from "@/lib/api";

const HINTS: Record<Technique, string> = {
  vector: "Plain dense retrieval over chunk embeddings — no graph.",
  bm25: "Keyword search (Okapi BM25): exact terms, IDs and file names. No LLM, no embeddings.",
  bm25_vector: "BM25 and vector rankings merged by reciprocal rank fusion — lexical + semantic.",
  local: "Entity-centric: match entities, then pull their neighbourhood.",
  global: "Relation-centric: rank the strongest relationships for the query.",
  hybrid: "Local entities plus global relations, merged.",
  mix: "Hybrid plus chunk vector search — the default mode.",
  agent: "Routing agent: states the goal, hypothesises where the answer lives, checks it on the metadata graph, then searches scoped. Watch it on the Agent page.",
};

type Props = {
  technique: Technique; onTechnique: (t: Technique) => void;
  topK: number; onTopK: (v: number) => void;
  chunkTopK: number; onChunkTopK: (v: number) => void;
};

function Range({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex justify-between font-mono text-[11px] text-muted">
        <span>{label}</span><span className="tabular text-ink">{value}</span>
      </span>
      <Slider.Root className="relative flex h-4 w-full touch-none select-none items-center" min={min} max={max} step={1}
        value={[value]} onValueChange={([v]) => onChange(v)} aria-label={label}>
        <Slider.Track className="relative h-[3px] grow rounded-full bg-line">
          <Slider.Range className="absolute h-full rounded-full bg-local" />
        </Slider.Track>
        <Slider.Thumb className="block h-3.5 w-3.5 rounded-full border border-local bg-bg outline-none focus-visible:ring-2 focus-visible:ring-local/50" />
      </Slider.Root>
    </label>
  );
}

export default function ModeCard(p: Props) {
  return (
    <section className="card flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <h2 className="caption">Mode</h2>
        <Link to="/compare" className="flex items-center gap-1 font-mono text-[11px] text-muted hover:text-ink">
          <BarChart3 size={12} />compare all
        </Link>
      </div>
      <div role="radiogroup" className="flex flex-wrap gap-0.5 rounded-md border border-line bg-bg p-0.5">
        {TECHNIQUES.map((t, i) => (
          <button key={t} role="radio" aria-checked={p.technique === t} onClick={() => p.onTechnique(t)} title={`${TECHNIQUE_LABEL[t]} (press ${i + 1})`}
            className={clsx("flex-1 whitespace-nowrap rounded px-2 py-1.5 font-mono text-xs transition-colors",
              p.technique === t ? "bg-surface-2 text-ink shadow-[inset_0_0_0_1px_var(--line)]" : "text-muted hover:text-ink")}>
            {TECHNIQUE_LABEL[t]}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <Range label="top_k" value={p.topK} min={1} max={40} onChange={p.onTopK} />
        <Range label="chunk_top_k" value={p.chunkTopK} min={1} max={20} onChange={p.onChunkTopK} />
      </div>
      <p className="text-xs text-muted">{HINTS[p.technique]}</p>
    </section>
  );
}
