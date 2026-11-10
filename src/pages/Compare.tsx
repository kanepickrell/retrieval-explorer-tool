import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, Trophy } from "lucide-react";
import clsx from "clsx";
import { evaluate, runRetrieval, TECHNIQUES, type Metric, type RetrievalResult, type Technique } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import { useHotkeys } from "@/lib/hotkeys";
import { LOWER_IS_BETTER, overallScore, scoredMetrics } from "@/lib/score";
import FlowStepper from "@/components/FlowStepper";
import QuestionCard from "@/components/QuestionCard";

type Runs = Partial<Record<Technique, RetrievalResult>>;

type Row = {
  key: string;
  label: string;
  /** null = not available for this run (e.g. a metric the backend has not computed yet) */
  get: (r: RetrievalResult) => number | null;
  fmt: (v: number) => string;
  /** rank this row and mark the best cell */
  ranked: boolean;
  /** draw a 0–1 bar under the value */
  bar?: boolean;
  /** small annotation under the value (e.g. whether keywords were cached) */
  note?: (r: RetrievalResult) => string | undefined;
};

const metricRow = (key: string, label: string): Row => ({
  key, label, ranked: true, bar: true,
  get: (r) => {
    const m = r.metrics.find((x) => x.key === key);
    return m && !m.pending ? m.value : null;
  },
  fmt: (v) => v.toFixed(2),
});

const SECTIONS: { title: string; dot?: string; referenceBased?: boolean; rows: Row[] }[] = [
  {
    title: "Cost & footprint",
    rows: [
      // ranked on the search itself: keyword extraction is shared/cached across modes, so
      // whichever mode ran first would otherwise look slowest
      { key: "latency", label: "Search latency", ranked: true, get: (r) => r.timeMs - (r.keywordMs ?? 0), fmt: (v) => `${v} ms` },
      { key: "keyword_ms", label: "Keyword extraction", ranked: false, get: (r) => r.keywordMs, fmt: (v) => `${v} ms`,
        note: (r) => (r.keywordsCached === null ? undefined : r.keywordsCached ? "cached" : "cold LLM") },
      { key: "entities", label: "Entities", ranked: false, get: (r) => r.graph.nodes.length, fmt: String },
      { key: "relations", label: "Relations", ranked: false, get: (r) => r.graph.edges.length, fmt: String },
      { key: "chunks", label: "Chunks", ranked: false, get: (r) => r.chunks.length, fmt: String },
    ],
  },
  {
    title: "Reference-free",
    dot: "var(--good)",
    rows: [
      metricRow("context_precision", "Context precision"),
      metricRow("faithfulness", "Faithfulness"),
      metricRow("answer_relevancy", "Answer relevancy"),
    ],
  },
  {
    title: "Reference-based",
    dot: "var(--vector)",
    referenceBased: true,
    rows: [
      metricRow("context_recall", "Context recall"),
      metricRow("context_entities_recall", "Entities recall"),
      metricRow("noise_sensitivity", "Noise sensitivity"),
      metricRow("recall_at_k", "recall@k"),
      metricRow("precision_at_k", "precision@k"),
    ],
  },
];

/** Runs the current question through every technique and ranks them side by side. */
export default function Compare() {
  const { datasetId, dataset, query, run, runNonce, topK, chunkTopK, setTechnique, reportError } = useExplorer();
  const [input, setInput] = useState(query);
  const [rawRuns, setRuns] = useState<Runs>({});
  // metrics computed on demand per technique (LLM judge, slow) -- merged over the pending placeholders
  const [evals, setEvals] = useState<Partial<Record<Technique, Metric[]>>>({});
  const [evalNotes, setEvalNotes] = useState<Partial<Record<Technique, string>>>({});
  const [evaluating, setEvaluating] = useState<Technique | null>(null);
  const evalRun = useRef(0);
  const runs: Runs = useMemo(
    () => Object.fromEntries(Object.entries(rawRuns).map(([t, r]) => [t, evals[t as Technique] ? { ...r!, metrics: evals[t as Technique]! } : r])) as Runs,
    [rawRuns, evals],
  );
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const labeled = !!dataset?.labeled;

  useHotkeys({ "/": () => inputRef.current?.focus() });

  useEffect(() => {
    if (!query || !datasetId) return;
    let live = true;
    setLoading(true);
    setEvals({}); setEvalNotes({}); setEvaluating(null); evalRun.current++;
    Promise.all(TECHNIQUES.map((t) => runRetrieval(datasetId, query, t, topK, chunkTopK)))
      .then((rs) => { if (live) setRuns(Object.fromEntries(rs.map((r) => [r.strategy, r])) as Runs); })
      .catch((e) => { if (live) { setRuns({}); reportError(String(e.message ?? e)); } })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [datasetId, query, runNonce, topK, chunkTopK]);

  const ready = TECHNIQUES.every((t) => runs[t]);
  // no winner until at least one technique has a computed quality metric (a real backend may report them as pending)
  const scorable = ready && TECHNIQUES.some((t) => scoredMetrics(runs[t]!, labeled).length > 0);
  const scores = useMemo(
    () => (scorable ? Object.fromEntries(TECHNIQUES.map((t) => [t, overallScore(runs[t]!, labeled)])) as Record<Technique, number> : null),
    [runs, scorable, labeled],
  );
  const ranking = scores ? [...TECHNIQUES].sort((a, b) => scores[b] - scores[a]) : [];
  const winner = ranking[0];

  const open = (t: Technique) => { setTechnique(t); navigate("/"); };

  /** Evaluate every technique, one after another (each is retrieval + generation + several judge calls). */
  const evaluateAll = async () => {
    const id = ++evalRun.current;
    setEvals({}); setEvalNotes({});
    for (const t of TECHNIQUES) {
      if (id !== evalRun.current) return; // query changed mid-run
      setEvaluating(t);
      const runId = rawRuns[t]?.runId;
      if (!runId) continue;
      // scores the exact retrieval shown in this column (same run id)
      const r = await evaluate(runId);
      if (id !== evalRun.current) return;
      if (r.metrics.length) setEvals((e) => ({ ...e, [t]: r.metrics }));
      if (r.note) setEvalNotes((n) => ({ ...n, [t]: r.note }));
    }
    setEvaluating(null);
  };
  const evaluatedCount = Object.keys(evals).length;

  return (
    <main className="mt-5 flex flex-col gap-4">
      <FlowStepper active={2} />
      <QuestionCard value={input} onChange={setInput} onRun={run} loading={loading} activeQuery={ready ? query : ""} inputRef={inputRef} />

      {!query ? (
        <section className="card py-12 text-center">
          <p className="text-sm text-muted">Ask a question to run it through all five techniques at once.</p>
          <p className="mt-1 text-xs text-muted/70">top_k {topK} · chunk_top_k {chunkTopK} — set in the Explorer's mode card</p>
        </section>
      ) : (
        <>
          {scores && winner && (
            <section className="card flex flex-wrap items-center gap-x-6 gap-y-3">
              <div className="flex items-center gap-3">
                <span className="grid h-9 w-9 place-items-center rounded-full border border-good/40 bg-good/10 text-good"><Trophy size={16} /></span>
                <div>
                  <div className="caption">Best for this question</div>
                  <div className="font-title text-lg font-bold">
                    {winner} <span className="font-mono text-sm font-normal text-muted">{scores[winner].toFixed(2)}</span>
                  </div>
                </div>
              </div>
              <div className="font-mono text-xs text-muted">
                +{(scores[winner] - scores[ranking[1]]).toFixed(2)} over {ranking[1]}
              </div>
              <ol className="ml-auto flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
                {ranking.map((t, i) => (
                  <li key={t} className={clsx("rounded border px-2 py-0.5", i === 0 ? "border-good/50 text-ink" : "border-line text-muted")}>
                    {i + 1}. {t}
                  </li>
                ))}
              </ol>
              <p className="w-full text-xs text-muted/80">
                Overall = mean of the {labeled ? "reference-free and reference-based" : "reference-free"} metrics
                (noise sensitivity inverted){labeled ? "" : " — this dataset has no gold labels, so recall metrics are left out"}.
                Search latency (excluding keyword extraction, which is shared and cached across modes) is ranked but not scored.
              </p>
            </section>
          )}

          {ready && (
            <section className="card flex flex-wrap items-center gap-3 text-sm">
              <button className="btn-primary h-8 px-3" onClick={evaluateAll} disabled={!!evaluating}>
                {evaluating ? <Loader2 size={14} className="animate-spin" /> : null}
                {evaluating ? `evaluating ${evaluating}… (${evaluatedCount}/${TECHNIQUES.length})` : evaluatedCount ? "re-evaluate all" : "evaluate all techniques"}
              </button>
              <span className="text-xs text-muted">
                {!scorable
                  ? "Quality metrics are computed on demand by the LLM judge (about a minute per technique). Until then there is no winner; cost and footprint are compared below."
                  : `${evaluatedCount}/${TECHNIQUES.length} techniques evaluated.`}
              </span>
              {Object.entries(evalNotes).map(([t, n]) => (
                <span key={t} className="w-full font-mono text-[11px] text-red-400">{t}: {n}</span>
              ))}
            </section>
          )}

          <section className={clsx("card overflow-x-auto p-0 transition-opacity", loading && "opacity-60")}>
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="border-b border-line">
                  <th className="w-48 px-4 py-3 text-left align-bottom">
                    <span className="caption">Metric</span>
                    {loading && <Loader2 size={13} className="ml-2 inline animate-spin text-local" />}
                  </th>
                  {TECHNIQUES.map((t) => {
                    const s = scores?.[t];
                    return (
                      <th key={t} className={clsx("px-3 py-3 text-left font-normal", t === winner && "bg-good/[0.06]")}>
                        <button onClick={() => open(t)} title={`Open ${t} in the Explorer`} className="group w-full text-left">
                          <span className="flex items-center gap-1.5 font-mono text-sm text-ink group-hover:text-local">
                            {t}{t === winner && <Trophy size={12} className="text-good" />}
                          </span>
                          <span className="mt-1 flex items-center gap-2">
                            <span className="h-[3px] flex-1 overflow-hidden rounded-full bg-line">
                              {s != null && <span className="block h-full rounded-full bg-ink/50" style={{ width: `${s * 100}%` }} />}
                            </span>
                            <span className="w-8 text-right font-mono text-[11px] tabular text-muted">{s != null ? s.toFixed(2) : "—"}</span>
                          </span>
                        </button>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              {SECTIONS.map((sec) => {
                const hidden = sec.referenceBased && !labeled;
                return (
                  <tbody key={sec.title}>
                    <tr>
                      <td colSpan={TECHNIQUES.length + 1} className="px-4 pb-1 pt-4">
                        <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-muted">
                          {sec.dot && <span className="h-1.5 w-1.5 rounded-full" style={{ background: sec.dot }} />}
                          {sec.title}
                          {hidden && <span className="ml-2 normal-case italic tracking-normal text-muted/70">needs gold — pick a labeled dataset</span>}
                        </span>
                      </td>
                    </tr>
                    {sec.rows.map((row) => <MetricRow key={row.key} row={row} runs={runs} hidden={!!hidden} winner={winner} />)}
                  </tbody>
                );
              })}
            </table>
          </section>
          <p className="font-mono text-[11px] text-muted">
            ran with top_k {topK} · chunk_top_k {chunkTopK} on {dataset?.name} · click a technique to open it in the Explorer
          </p>
        </>
      )}
    </main>
  );
}

function MetricRow({ row, runs, hidden, winner }: { row: Row; runs: Runs; hidden: boolean; winner?: Technique }) {
  const vals = TECHNIQUES.map((t) => (runs[t] ? row.get(runs[t]!) : null));
  const lower = LOWER_IS_BETTER.has(row.key);
  let best: number | null = null;
  const present = vals.filter((v): v is number => v != null);
  if (row.ranked && !hidden && present.length > 1) {
    best = lower ? Math.min(...present) : Math.max(...present);
  }
  return (
    <tr className="border-t border-line/50">
      <td className="px-4 py-2 text-xs text-muted">
        {row.label}{lower && <span className="ml-1 text-muted/60" title="lower is better">↓</span>}
      </td>
      {TECHNIQUES.map((t, i) => {
        const v = vals[i];
        const isBest = best != null && v === best;
        return (
          <td key={t} className={clsx("px-3 py-2", t === winner && "bg-good/[0.06]")}>
            {hidden || v == null ? (
              <span className="font-mono text-xs text-muted/50">—</span>
            ) : (
              <div className="flex items-center gap-2">
                <span className={clsx("w-14 font-mono text-xs tabular", isBest ? "font-semibold text-good" : "text-ink")}>{row.fmt(v)}</span>
                {row.note && runs[t] && row.note(runs[t]!) && (
                  <span className="font-mono text-[10px] text-muted">{row.note(runs[t]!)}</span>
                )}
                {row.bar && (
                  <span className="h-[3px] flex-1 overflow-hidden rounded-full bg-line">
                    <span className={clsx("block h-full rounded-full", isBest ? "bg-good" : "bg-ink/35")} style={{ width: `${v * 100}%` }} />
                  </span>
                )}
              </div>
            )}
          </td>
        );
      })}
    </tr>
  );
}
