import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import clsx from "clsx";
import type { GraphNode } from "@/lib/api";
import { clusterColor } from "@/lib/signals";

type Props = { nodes: GraphNode[]; focusId: string | null; onFocus: (id: string | null) => void };

/** Faceted type-ahead over node names; picking one isolates its 1-hop neighbourhood. */
export default function NodeSearch({ nodes, focusId, onFocus }: Props) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);

  const matches = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return [];
    return nodes
      .filter((n) => n.label.toLowerCase().includes(s) || n.cluster.startsWith(s))
      .sort((a, b) => Number(!a.label.toLowerCase().startsWith(s)) - Number(!b.label.toLowerCase().startsWith(s)) || b.size - a.size)
      .slice(0, 8);
  }, [q, nodes]);

  const pick = (n: GraphNode) => { onFocus(n.id); setQ(n.label); setOpen(false); };
  const clear = () => { onFocus(null); setQ(""); };

  return (
    <div className="relative w-56">
      <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
      <input
        className="input py-1.5 pl-7 pr-7 text-xs"
        placeholder="Find a node…"
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true); setCursor(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(matches.length - 1, c + 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(0, c - 1)); }
          else if (e.key === "Enter" && matches[cursor]) pick(matches[cursor]);
          else if (e.key === "Escape") clear();
        }}
        aria-label="Find a node"
      />
      {(focusId || q) && (
        <button className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-ink" onClick={clear} aria-label="Clear search">
          <X size={13} />
        </button>
      )}
      {open && matches.length > 0 && (
        <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-line bg-surface shadow-lg">
          {matches.map((n, i) => (
            <li key={n.id}>
              <button
                onMouseDown={(e) => { e.preventDefault(); pick(n); }}
                className={clsx("flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs", i === cursor ? "bg-surface-2 text-ink" : "text-muted")}
              >
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: clusterColor(n.cluster) }} />
                <span className="truncate">{n.label}</span>
                <span className="ml-auto font-mono text-[10px] text-muted/70">{n.cluster}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
