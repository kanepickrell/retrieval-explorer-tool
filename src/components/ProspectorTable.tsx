import clsx from "clsx";
import type { Dataset } from "@/lib/api";

const KIND_COLOR: Record<Dataset["kind"], string> = { chat: "var(--local)", document: "var(--global)", user: "var(--vector)" };

type Props = { datasets: Dataset[]; activeId: string; onSelect: (id: string) => void };

export default function ProspectorTable({ datasets, activeId, onSelect }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead>
          <tr className="caption text-left">
            <th className="px-3 py-2 font-normal">name</th>
            <th className="px-3 py-2 font-normal">kind</th>
            <th className="px-3 py-2 text-right font-normal">docs</th>
            <th className="px-3 py-2 font-normal">status</th>
            <th className="px-3 py-2 font-normal">updated</th>
            <th className="px-3 py-2 font-normal">gold</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((d) => {
            const active = d.id === activeId;
            return (
              <tr key={d.id} onClick={() => onSelect(d.id)} aria-selected={active}
                className={clsx("cursor-pointer border-t border-line transition-colors", active ? "bg-surface-2 shadow-[inset_2px_0_0_var(--local)]" : "hover:bg-surface-2/50")}>
                <td className="px-3 py-2.5">
                  <span className="flex items-center gap-2">
                    <span className={clsx("h-1.5 w-1.5 rounded-full", active ? "bg-local" : "bg-transparent")} />
                    {d.name}
                  </span>
                </td>
                <td className="px-3 py-2.5 font-mono text-xs" style={{ color: KIND_COLOR[d.kind] }}>{d.kind}</td>
                <td className="px-3 py-2.5 text-right font-mono text-xs tabular">{d.rows.toLocaleString()}</td>
                <td className="px-3 py-2.5 font-mono text-xs text-muted">
                  {d.status === "ready" ? "ready" : `${d.status} ${d.rows}/${d.expectedRows}`}
                  {d.failedRows > 0 && <span className="ml-1 text-red-400">· {d.failedRows} failed</span>}
                </td>
                <td className="px-3 py-2.5 font-mono text-xs text-muted">{d.updated ?? "—"}</td>
                <td className="px-3 py-2.5 font-mono text-xs text-muted">
                  {d.labeled ? `${d.goldQueries} queries${d.hasReferenceAnswers ? " + answers" : ""}` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
