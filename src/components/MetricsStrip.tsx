import { Loader2, Sparkles } from "lucide-react";
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

// the tiles shown before evaluation — all pending (no fabricated values)
const PLACEHOLDER: Metric[] = [
  { key: "faithfulness", label: "Faithfulness", referenceBased: false, value: 0, pending: true },
  { key: "answer_relevancy", label: "Answer Relevancy", referenceBased: false, value: 0, pending: true },
  { key: "context_precision", label: "Context Precision", referenceBased: false, value: 0, pending: true },
  { key: "context_recall", label: "Context Recall", referenceBased: true, value: 0, pending: true },
  { key: "context_entities_recall", label: "Context Entities Recall", referenceBased: true, value: 0, pending: true },
  { key: "noise_sensitivity", label: "Noise Sensitivity", referenceBased: true, value: 0, pending: true },
  { key: "recall_at_k", label: "recall@k", referenceBased: true, value: 0, pending: true },
];

function Row({ m }: { m: Metric }) {
  const lower = LOWER_IS_BETTER.has(m.key);
  // reference-based metrics that need a gold set read "needs gold"; others just "—"
  const pendingLabel = m.referenceBased ? "needs gold" : "—";
  const q = lower ? 1 - m.value : m.value;
  const tone = q >= 0.75 ? "var(--good)" : q >= 0.55 ? "var(--warn)" : "var(--bad)";
  return (
    <div className="flex items-center gap-2 py-[3px]" title={`${m.label}${lower ? " — lower is better" : ""}`}>
      <span className="min-w-0 flex-1 truncate text-xs text-muted">
        {SHORT[m.key] ?? m.label}{lower && <span className="ml-1 text-muted/60">↓</span>}
      </span>
      <span className="h-[3px] w-14 shrink-0 overflow-hidden rounded-full bg-line">
        {!m.pending && <span className="block h-full rounded-full transition-[width] duration-500" style={{ width: `${m.value * 100}%`, background: tone }} />}
      </span>
      <span className="shrink-0 text-right font-mono text-xs tabular text-ink" style={{ minWidth: m.pending ? "4.5rem" : "2rem" }}>
        {m.pending ? <span className="text-muted/70">{pendingLabel}</span> : m.value.toFixed(2)}
      </span>
    </div>
  );
}

function Group({ title, dot, metrics }: { title: string; dot: string; metrics: Metric[] }) {
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] text-muted">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: dot }} />
        <span className="uppercase tracking-wider">{title}</span>
      </div>
      {metrics.map((m) => <Row key={m.key} m={m} />)}
    </div>
  );
}

type Props = {
  result: RetrievalResult | null;
  computed: Metric[] | null;
  loading: boolean;
  onEvaluate: () => void;
};

/** Ragas-style evaluation, split into reference-free and reference-based (needs
 *  gold) groups. Values are computed on demand by the backend judge. */
export default function MetricsStrip({ result, computed, loading, onEvaluate }: Props) {
  const metrics = computed ?? PLACEHOLDER;
  const free = metrics.filter((m) => !m.referenceBased);
  const ref = metrics.filter((m) => m.referenceBased);
  const evaluated = !!computed;
  return (
    <div className="border-t border-line pt-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted">Evaluation · Ragas</span>
        <button
          onClick={onEvaluate}
          disabled={!result || loading}
          className="flex items-center gap-1.5 rounded border border-line px-2 py-1 font-mono text-[11px] text-muted transition-colors enabled:hover:border-local enabled:hover:text-ink disabled:opacity-40"
        >
          {loading ? <Loader2 size={11} className="animate-spin text-local" /> : <Sparkles size={11} />}
          {loading ? "evaluating…" : evaluated ? "re-evaluate" : "evaluate"}
        </button>
      </div>
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        <Group title="Reference-free" dot="var(--good)" metrics={free} />
        <Group title="Reference-based" dot="var(--vector)" metrics={ref} />
      </div>
    </div>
  );
}
