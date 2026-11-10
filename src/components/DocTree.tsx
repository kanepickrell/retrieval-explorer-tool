import { useMemo, useState } from "react";
import clsx from "clsx";
import { ChevronRight, FileText, Folder, FolderOpen, Loader2, Network, Search, Target } from "lucide-react";
import { listDocChunks, type DocChunk, type DocumentSummary } from "@/lib/api";

type Props = {
  datasetId: string;
  docs: DocumentSummary[];
  /** highlight these entity ids on the Explorer graph */
  onShow: (label: string, entityIds: string[]) => void;
};

type Tree = Map<string, Map<string, DocumentSummary[]>>;

/**
 * Browse a dataset as category › subcategory › document › chunks, the way the index
 * actually holds it. Search filters by title, id, gold query text or entity; the
 * "gold" filter shows only docs that some eval query expects as its answer.
 */
export default function DocTree({ datasetId, docs, onShow }: Props) {
  const [q, setQ] = useState("");
  const [goldOnly, setGoldOnly] = useState(false);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [chunks, setChunks] = useState<Record<string, DocChunk[] | "loading" | string>>({});

  const needle = q.trim().toLowerCase();
  const filtered = useMemo(() => docs.filter((d) => {
    if (goldOnly && d.gold.length === 0) return false;
    if (!needle) return true;
    return d.title.toLowerCase().includes(needle) || d.id.toLowerCase().includes(needle)
      || d.gold.some((g) => g.query.toLowerCase().includes(needle))
      || d.entities.some((e) => e.toLowerCase().includes(needle));
  }), [docs, needle, goldOnly]);

  const tree: Tree = useMemo(() => {
    const t: Tree = new Map();
    for (const d of filtered) {
      const sub = t.get(d.category) ?? t.set(d.category, new Map()).get(d.category)!;
      (sub.get(d.subcategory) ?? sub.set(d.subcategory, []).get(d.subcategory)!).push(d);
    }
    return t;
  }, [filtered]);

  // while searching, show every match expanded
  // categories / subcategories auto-expand while searching or filtering; documents open on click
  const isOpen = (key: string) => (needle || goldOnly ? true : open.has(key));
  const toggle = (key: string) => setOpen((s) => { const n = new Set(s); n.has(key) ? n.delete(key) : n.add(key); return n; });

  const toggleDoc = (d: DocumentSummary) => {
    const key = `doc:${d.id}`;
    toggle(key);
    if (!chunks[d.id]) {
      setChunks((c) => ({ ...c, [d.id]: "loading" }));
      listDocChunks(datasetId, d.id)
        .then((cs) => setChunks((c) => ({ ...c, [d.id]: cs })))
        .catch((e) => setChunks((c) => ({ ...c, [d.id]: `failed: ${e.message ?? e}` })));
    }
  };

  const goldDocs = docs.filter((d) => d.gold.length).length;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <label className="relative min-w-0 flex-1">
          <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
          <input className="input h-8 w-full pl-8 text-sm" placeholder="Search titles, ids, entities or gold questions…"
            value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search documents" />
        </label>
        <button onClick={() => setGoldOnly((v) => !v)} aria-pressed={goldOnly}
          title="Only documents that a gold eval query expects as its answer"
          className={clsx("flex h-8 items-center gap-1.5 rounded-md border px-2.5 font-mono text-[11px]",
            goldOnly ? "border-good/60 bg-good/10 text-ink" : "border-line text-muted hover:text-ink")}>
          <Target size={12} /> gold docs ({goldDocs})
        </button>
        <span className="font-mono text-[11px] text-muted">{filtered.length} of {docs.length} documents</span>
      </div>

      <ul className="flex flex-col gap-0.5 text-sm">
        {[...tree.entries()].map(([cat, subs]) => {
          const ck = `cat:${cat}`;
          const n = [...subs.values()].reduce((a, l) => a + l.length, 0);
          return (
            <li key={cat}>
              <Row depth={0} open={isOpen(ck)} onClick={() => toggle(ck)}
                icon={isOpen(ck) ? <FolderOpen size={14} /> : <Folder size={14} />} label={cat} count={n} strong />
              {isOpen(ck) && [...subs.entries()].map(([sub, list]) => {
                const sk = `sub:${cat}/${sub}`;
                return (
                  <div key={sub}>
                    <Row depth={1} open={isOpen(sk)} onClick={() => toggle(sk)}
                      icon={isOpen(sk) ? <FolderOpen size={13} /> : <Folder size={13} />} label={sub} count={list.length} />
                    {isOpen(sk) && list.map((d) => {
                      const dk = `doc:${d.id}`;
                      const st = chunks[d.id];
                      return (
                        <div key={d.id}>
                          <Row depth={2} open={open.has(dk)} onClick={() => toggleDoc(d)} icon={<FileText size={13} />}
                            label={d.title} title={d.id}
                            meta={`${d.chunks} chunk${d.chunks === 1 ? "" : "s"} · ${d.entities.length} entities`}
                            badge={d.gold.length ? `gold ×${d.gold.length}` : undefined}
                            badgeTitle={d.gold.map((g) => `${g.type || "eval"}: ${g.query}`).join("\n")}
                            action={() => onShow(d.title, d.entities)} />
                          {open.has(dk) && (
                            <div className="mb-1.5 ml-[60px] mr-1 flex flex-col gap-1.5">
                              {d.gold.length > 0 && (
                                <div className="rounded-md border border-good/30 bg-good/[0.06] px-2.5 py-1.5 text-xs">
                                  <div className="mb-0.5 font-mono text-[10px] uppercase tracking-wide text-good">gold for</div>
                                  {d.gold.map((g) => (
                                    <div key={g.id} className="truncate text-muted" title={g.query}>
                                      <span className="font-mono text-[10px] text-muted/70">{g.type.replace("generated_", "gen:") || "eval"} </span>{g.query}
                                    </div>
                                  ))}
                                </div>
                              )}
                              {st === "loading" && <span className="flex items-center gap-1.5 font-mono text-[11px] text-muted"><Loader2 size={12} className="animate-spin" /> loading chunks…</span>}
                              {typeof st === "string" && st !== "loading" && <span className="font-mono text-[11px] text-red-400">{st}</span>}
                              {Array.isArray(st) && st.map((c) => (
                                <div key={c.id} className="rounded-md border border-line bg-surface-2/50 px-2.5 py-1.5">
                                  <div className="mb-1 flex items-center gap-2 font-mono text-[10px] text-muted">
                                    <span>chunk {c.order}</span><span>{c.tokens} tokens</span><span>{c.entities.length} entities</span>
                                    {c.entities.length > 0 && (
                                      <button className="ml-auto flex items-center gap-1 text-local hover:underline"
                                        onClick={() => onShow(`${d.title} · chunk ${c.order}`, c.entities)}>
                                        <Network size={11} /> show on graph
                                      </button>
                                    )}
                                  </div>
                                  <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words font-sans text-xs text-ink/90">{c.content}</pre>
                                  {c.entities.length > 0 && (
                                    <div className="mt-1 flex flex-wrap gap-1">
                                      {c.entities.slice(0, 24).map((e) => (
                                        <span key={e} className="rounded border border-line px-1 font-mono text-[10px] text-muted">{e}</span>
                                      ))}
                                      {c.entities.length > 24 && <span className="font-mono text-[10px] text-muted">+{c.entities.length - 24}</span>}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </li>
          );
        })}
        {tree.size === 0 && <li className="py-4 text-center text-xs text-muted">No documents match.</li>}
      </ul>
    </div>
  );
}

function Row({ depth, open, onClick, icon, label, count, meta, badge, badgeTitle, title, strong, action }: {
  depth: number; open: boolean; onClick: () => void; icon: React.ReactNode; label: string;
  count?: number; meta?: string; badge?: string; badgeTitle?: string; title?: string; strong?: boolean; action?: () => void;
}) {
  return (
    <div className="group flex items-center gap-1.5 rounded px-1.5 py-1 hover:bg-surface-2/60" style={{ paddingLeft: 6 + depth * 18 }}>
      <button onClick={onClick} className="flex min-w-0 flex-1 items-center gap-1.5 text-left" title={title ?? label} aria-expanded={open}>
        <ChevronRight size={12} className={clsx("shrink-0 text-muted transition-transform", open && "rotate-90")} />
        <span className="shrink-0 text-muted">{icon}</span>
        <span className={clsx("truncate", strong ? "font-medium text-ink" : "text-ink/90")}>{label}</span>
        {count != null && <span className="shrink-0 font-mono text-[11px] text-muted">{count}</span>}
        {meta && <span className="ml-1 hidden shrink-0 font-mono text-[11px] text-muted sm:inline">{meta}</span>}
      </button>
      {badge && <span className="shrink-0 rounded border border-good/50 px-1.5 font-mono text-[10px] text-good" title={badgeTitle}>{badge}</span>}
      {action && (
        <button onClick={action} title="Highlight this document's entities on the graph"
          className="shrink-0 text-muted opacity-0 transition-opacity hover:text-local group-hover:opacity-100">
          <Network size={13} />
        </button>
      )}
    </div>
  );
}
