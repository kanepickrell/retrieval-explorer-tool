import { X } from "lucide-react";
import type { GraphNode } from "@/lib/api";
import { clusterColor, type Signals } from "@/lib/signals";

type Props = { node: GraphNode; degree: number; signals: Signals | null; onClose: () => void };

/** Rendered instantly from the node object — no fetch, no loading state. */
export default function NodeInspector({ node, degree, signals, onClose }: Props) {
  const roles: string[] = [];
  if (signals?.entity.has(node.id)) roles.push("entity hit");
  if (signals?.vector.has(node.id)) roles.push("vector chunk");
  if (signals?.context.has(node.id)) roles.push("in final context");
  return (
    <div className="absolute right-3 top-3 z-10 w-64 rounded-card border border-line bg-surface/95 p-3 shadow-xl backdrop-blur">
      <div className="flex items-start gap-2">
        <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: clusterColor(node.cluster) }} />
        <div className="min-w-0 flex-1">
          <div className="font-title text-sm font-semibold">{node.label}</div>
          <div className="font-mono text-[11px] text-muted">{node.type} · cluster {node.cluster} · degree {degree}</div>
        </div>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close inspector"><X size={14} /></button>
      </div>
      {node.description && <p className="mt-2 text-xs leading-relaxed text-muted">{node.description}</p>}
      {roles.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {roles.map((r) => <span key={r} className="rounded border border-line px-1.5 font-mono text-[10px] text-ink">{r}</span>)}
        </div>
      )}
    </div>
  );
}
