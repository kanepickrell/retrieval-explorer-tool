import clsx from "clsx";
import type { Metric, RetrievalResult } from "@/lib/api";

const LOWER_IS_BETTER = new Set(["noise_sensitivity"]);
const DOT = { free: "var(--good)", ref: "var(--vector)" };

const PLACEHOLDER: Metric[] = [
  { key: "context_precision", label: "Context Precision", referenceBased: false, value: 0 },
  { key: "faithfulness", label: "Faithfulness", referenceBased: false, value: 0 },
  { key: "answer_relevancy", label: "Answer Relevancy", referenceBased: false, value: 0 },
  { key: "noise_sensitivity", label: "Noise Sensitivity", referenceBased: false, value: 0 },
  { key: "context_recall", label: "Context Recall", referenceBased: true, value: 0 },
  { key: "context_entities_recall", label: "Context Entities Recall", referenceBased: true, value: 0 },
  { key: "recall_at_k", label: "recall@k", referenceBased: true, value: 0 },
];

type TileProps = { label: string; value: string; bar?: number; dot?: string; muted?: boolean; hint?: string };

function Tile({ label, value, bar, dot, muted, hint }: TileProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-md border border-line bg-surface-2/60 px-2.5 py-2" title={hint ?? label}>
      <span className="flex min-h-[26px] items-start gap-1.5 font-mono text-[10px] leading-tight text-muted">
        {dot && <span className="mt-[3px] h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: dot }} />}
        <span className="line-clamp-2">{label}</span>
      </span>
      <span className={clsx("font-mono tabular", muted ? "text-[11px] italic text-muted/70" : "text-sm text-ink")}>{value}</span>
      <span className={clsx("h-[3px] rounded-full", muted ? "border-t border-dashed border-line" : "bg-line")}>
        {bar != null && !muted && <span className="block h-full rounded-full bg-ink/40" style={{ width: `${bar * 100}%` }} />}
      </span>
    </div>
  );
}

export default function MetricsStrip({ result, labeled }: { result: RetrievalResult | null; labeled: boolean }) {
  const dash = "—";
  const r = result;
  return (
    <div className="flex flex-col gap-2 border-t border-line pt-3">
      <div className="grid grid-cols-4 gap-1.5">
        <Tile label="latency" value={r ? `${r.timeMs} ms` : dash} />
        <Tile label="entities" value={r ? String(r.graph.nodes.length) : dash} />
        <Tile label="relations" value={r ? String(r.graph.edges.length) : dash} />
        <Tile label="chunks" value={r ? String(r.chunks.length) : dash} />
      </div>
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {(r?.metrics ?? PLACEHOLDER).map((m) => {
          const needsGold = m.referenceBased && !labeled;
          const lower = LOWER_IS_BETTER.has(m.key);
          return (
            <Tile key={m.key}
              label={lower ? `${m.label} ↓` : m.label}
              hint={lower ? `${m.label} — lower is better` : m.label}
              dot={m.referenceBased ? DOT.ref : DOT.free}
              muted={needsGold}
              value={!r ? dash : needsGold ? "needs gold" : m.value.toFixed(2)}
              bar={r && !needsGold ? m.value : undefined} />
          );
        })}
      </div>
      <p className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted">
        <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full" style={{ background: DOT.free }} />reference-free · any query</span>
        <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full" style={{ background: DOT.ref }} />reference-based · needs a labeled dataset</span>
      </p>
    </div>
  );
}
