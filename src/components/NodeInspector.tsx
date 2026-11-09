import { X } from "lucide-react";
import type { GraphNode } from "@/lib/api";
import { clusterColor, type Signals } from "@/lib/signals";

type Props = {
  node: GraphNode;
  degree: number;
  signals: Signals | null;
  neighbours: GraphNode[];
  onPick: (id: string) => void;
  onClose: () => void;
};

/** Rendered instantly from the node object — no fetch, no loading state. */
export default function NodeInspector({ node, degree, signals, neighbours, onPick, onClose }: Props) {
  const roles: [string, string][] = [];
  if (signals?.entity.has(node.id)) roles.push(["entity hit", "var(--local)"]);
  if (signals?.vector.has(node.id)) roles.push(["vector chunk", "var(--vector)"]);
  if (signals?.context.has(node.id)) roles.push(["final context", "#f4f7fa"]);
  return (
    <div className="absolute right-3 top-3 z-10 w-72 rounded-card border border-line bg-surface/95 shadow-xl backdrop-blur">
      <div className="flex items-start gap-2.5 p-3 pb-2">
        <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: clusterColor(node.cluster) }} />
        <div className="min-w-0 flex-1">
          <div className="font-title text-sm font-semibold leading-snug">{node.label}</div>
          <div className="font-mono text-[11px] text-muted">{node.cluster} · {degree} links</div>
        </div>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close inspector"><X size={14} /></button>
      </div>
      {node.description && <p className="px-3 text-xs leading-relaxed text-muted">{node.description}</p>}
      {roles.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {roles.map(([r, c]) => (
            <span key={r} className="flex items-center gap-1 rounded border border-line px-1.5 py-0.5 font-mono text-[10px] text-ink">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />{r}
            </span>
          ))}
        </div>
      )}
      {neighbours.length > 0 && (
        <div className="mt-3 border-t border-line px-3 py-2.5">
          <div className="caption mb-1.5 text-[10px]">Neighbours</div>
          <div className="flex flex-wrap gap-1">
            {neighbours.map((n) => (
              <button key={n.id} onClick={() => onPick(n.id)}
                className="flex items-center gap-1 rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[11px] text-muted transition-colors hover:border-muted/60 hover:text-ink">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: clusterColor(n.cluster) }} />{n.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
