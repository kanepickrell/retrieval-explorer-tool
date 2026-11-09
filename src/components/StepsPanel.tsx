import type { Step } from "@/lib/api";

/** Retrieval trace with a small timing waterfall: each bar is offset by the steps before it. */
export default function StepsPanel({ steps, totalMs }: { steps: Step[]; totalMs: number }) {
  const retrievalMs = steps.reduce((a, s) => a + (s.ms ?? 0), 0) || 1;
  let offset = 0;
  return (
    <div className="flex flex-col gap-1.5">
      <ol className="flex flex-col gap-1.5">
        {steps.map((s, i) => {
          const start = offset;
          offset += s.ms ?? 0;
          return (
            <li key={s.label} className="grid grid-cols-[18px_1fr_auto] items-baseline gap-x-2.5 rounded-md border border-line bg-surface-2/50 px-3 py-2">
              <span className="font-mono text-[11px] text-muted">{i + 1}</span>
              <div className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-medium">{s.label}</span>
                  {s.stat && <span className="truncate font-mono text-[11px] text-local">{s.stat}</span>}
                </div>
                <div className="truncate text-xs text-muted" title={s.detail}>{s.detail}</div>
              </div>
              <span className="font-mono text-[11px] tabular text-muted">{s.ms != null ? `${s.ms} ms` : ""}</span>
              {s.ms != null && (
                <span className="col-start-2 col-end-4 mt-1.5 h-[3px] rounded-full bg-line/60">
                  <span className="block h-full rounded-full bg-local/70"
                    style={{ marginLeft: `${(start / retrievalMs) * 100}%`, width: `${Math.max(1.5, ((s.ms ?? 0) / retrievalMs) * 100)}%` }} />
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <p className="flex justify-between px-1 pt-1 font-mono text-[11px] text-muted">
        <span>retrieval {retrievalMs} ms</span>
        <span>+ generation {Math.max(0, totalMs - retrievalMs)} ms = <span className="text-ink">{totalMs} ms</span></span>
      </p>
    </div>
  );
}
