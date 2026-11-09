import type { Step } from "@/lib/api";

export default function StepsPanel({ steps }: { steps: Step[] }) {
  return (
    <ol className="flex flex-col gap-2">
      {steps.map((s, i) => (
        <li key={s.label} className="grid grid-cols-[22px_1fr_auto] gap-x-3 rounded-md border border-line bg-surface-2/60 px-3 py-2.5">
          <span className="font-mono text-xs text-muted">{i + 1}</span>
          <div className="min-w-0">
            <div className="text-sm font-medium">{s.label}</div>
            <div className="truncate text-xs text-muted" title={s.detail}>{s.detail}</div>
            {s.stat && <div className="mt-1 font-mono text-[11px] text-local">{s.stat}</div>}
          </div>
          {s.ms != null && (
            <span className="self-start rounded border border-line px-1.5 py-0.5 font-mono text-[11px] tabular text-muted">{s.ms} ms</span>
          )}
        </li>
      ))}
    </ol>
  );
}
