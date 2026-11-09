import type { Metric, RetrievalResult } from "@/lib/api";

const LOWER_IS_BETTER = new Set(["noise_sensitivity"]);
const SHORT: Record<string, string> = {
  context_precision: "Context precision",
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer relevancy",
  noise_sensitivity: "Noise sensitivity",
  context_recall: "Context recall",
  context_entities_recall: "Entities recall",
  recall_at_k: "recall@k",
};

const PLACEHOLDER: Metric[] = [
  { key: "context_precision", label: "Context Precision", referenceBased: false, value: 0 },
  { key: "faithfulness", label: "Faithfulness", referenceBased: false, value: 0 },
  { key: "answer_relevancy", label: "Answer Relevancy", referenceBased: false, value: 0 },
  { key: "noise_sensitivity", label: "Noise Sensitivity", referenceBased: false, value: 0 },
  { key: "context_recall", label: "Context Recall", referenceBased: true, value: 0 },
  { key: "context_entities_recall", label: "Context Entities Recall", referenceBased: true, value: 0 },
  { key: "recall_at_k", label: "recall@k", referenceBased: true, value: 0 },
];

function Row({ m, show }: { m: Metric; show: boolean }) {
  const lower = LOWER_IS_BETTER.has(m.key);
  // colour the bar by quality; for lower-is-better metrics invert the scale
  const q = lower ? 1 - m.value : m.value;
  const tone = q >= 0.75 ? "var(--good)" : q >= 0.55 ? "var(--warn)" : "var(--bad)";
  return (
    <div className="flex items-center gap-2 py-[3px]" title={`${m.label}${lower ? " — lower is better" : ""}`}>
      <span className="min-w-0 flex-1 truncate text-xs text-muted">
        {SHORT[m.key] ?? m.label}{lower && <span className="ml-1 text-muted/60">↓</span>}
      </span>
      <span className="h-[3px] w-14 shrink-0 overflow-hidden rounded-full bg-line">
        {show && <span className="block h-full rounded-full transition-[width] duration-500" style={{ width: `${m.value * 100}%`, background: tone }} />}
      </span>
      <span className="w-8 shrink-0 text-right font-mono text-xs tabular text-ink">{show ? m.value.toFixed(2) : "—"}</span>
    </div>
  );
}

function Group({ title, dot, note, metrics, show }: { title: string; dot: string; note?: string; metrics: Metric[]; show: boolean }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] text-muted">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: dot }} />
        <span className="uppercase tracking-wider">{title}</span>
        {note && <span className="ml-auto italic text-muted/70">{note}</span>}
      </div>
      {metrics.map((m) => <Row key={m.key} m={m} show={show} />)}
    </div>
  );
}

/** Ragas-style evaluation, split into reference-free and reference-based (needs gold) groups. */
export default function MetricsStrip({ result, labeled }: { result: RetrievalResult | null; labeled: boolean }) {
  const metrics = result?.metrics ?? PLACEHOLDER;
  const free = metrics.filter((m) => !m.referenceBased);
  const ref = metrics.filter((m) => m.referenceBased);
  return (
    <div className="grid gap-x-6 gap-y-3 border-t border-line pt-3 sm:grid-cols-2">
      <Group title="Reference-free" dot="var(--good)" metrics={free} show={!!result} />
      <Group title="Reference-based" dot="var(--vector)" metrics={ref} show={!!result && labeled}
        note={labeled ? undefined : "needs gold"} />
    </div>
  );
}
