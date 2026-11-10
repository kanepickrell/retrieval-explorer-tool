import clsx from "clsx";
import type { Chunk } from "@/lib/api";

const VIA: Record<string, string> = { E: "via entity", R: "via relation", C: "vector match" };
import { SOURCE_COLOR } from "@/lib/signals";

type Props = { chunks: Chunk[]; activeId: string | null; onSelect: (c: Chunk | null) => void };

export default function ContextPanel({ chunks, activeId, onSelect }: Props) {
  if (chunks.length === 0) return <p className="text-sm text-muted">No chunks retrieved.</p>;
  return (
    <ul className="flex flex-col gap-1.5">
      {chunks.map((c, i) => {
        const active = c.id === activeId;
        const color = SOURCE_COLOR[c.source] ?? "var(--muted)";
        return (
          <li key={c.id}>
            <button onClick={() => onSelect(active ? null : c)}
              className={clsx("grid w-full grid-cols-[22px_1fr_auto] gap-x-3 rounded-md border px-3 py-2.5 text-left transition-colors",
                active ? "border-local/70 bg-surface-2" : "border-line bg-surface-2/60 hover:border-muted/50")}>
              <span className="font-mono text-xs text-muted">{i + 1}</span>
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">{c.title}</div>
                {c.snippet && <div className={clsx("text-xs text-muted", !active && "line-clamp-1")}>{c.snippet}</div>}
                <span className="mt-1 inline-block rounded px-1.5 font-mono text-[10px]"
                  style={{ color, border: `1px solid color-mix(in srgb, ${color} 45%, transparent)` }}>{c.source}</span>
                {c.via && <span className="ml-1.5 font-mono text-[10px] text-muted">{VIA[c.via] ?? c.via}</span>}
              </div>
              <div className="flex w-12 flex-col items-end gap-1" title="cosine similarity between the query and this chunk (same embeddings for every technique)">
                <span className="font-mono text-xs tabular">{c.score == null ? "—" : c.score.toFixed(2)}</span>
                <span className="h-[3px] w-full rounded-full bg-line">
                  {c.score != null && <span className="block h-full rounded-full" style={{ width: `${Math.max(0, c.score) * 100}%`, background: color }} />}
                </span>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
