import { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import { ChevronDown, CircleHelp, Database, X } from "lucide-react";
import clsx from "clsx";
import { useExplorer } from "@/lib/explorer-context";
import { SHORTCUTS, useHotkeys } from "@/lib/hotkeys";

const ABOUT =
  "Run a question through a retrieval technique and inspect the subgraph it touched, " +
  "the chunks it pulled, the steps it took and how it scored.";

/** Dataset is app-wide context (Explorer queries it, Prospector browses it), so it lives in the header. */
function DatasetPicker() {
  const { datasets, datasetId, setDatasetId } = useExplorer();
  const label = (d: (typeof datasets)[number]) =>
    d.status === "ready"
      ? `${d.name} · ${d.rows} docs${d.labeled ? ` · gold ${d.goldQueries}` : ""}`
      : `${d.name} · indexing ${d.rows}/${d.expectedRows}`;
  return (
    <label className="relative flex items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm hover:border-muted/50">
      <Database size={14} className="text-muted" />
      <span className="sr-only">Dataset</span>
      <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}
        className="cursor-pointer appearance-none bg-transparent pr-5 text-ink outline-none">
        {datasets.length === 0 && <option value="">no datasets</option>}
        {datasets.map((d) => (
          <option key={d.id} value={d.id} disabled={d.status !== "ready"} className="bg-surface">{label(d)}</option>
        ))}
      </select>
      <ChevronDown size={14} className="pointer-events-none absolute right-2 text-muted" />
    </label>
  );
}

/** "?" button + popover listing what the app does and its keyboard shortcuts. */
function Help() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useHotkeys({ "?": () => setOpen((o) => !o) });
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", close);
    return () => { window.removeEventListener("mousedown", close); window.removeEventListener("keydown", close); };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label="Help and shortcuts" title="Help and shortcuts (?)"
        className={clsx("rounded-md p-1.5 transition-colors hover:text-ink", open ? "text-ink" : "text-muted")}>
        <CircleHelp size={17} />
      </button>
      {open && (
        <div role="dialog" aria-label="Help" className="absolute right-0 top-10 z-30 w-80 rounded-card border border-line bg-surface p-4 shadow-2xl">
          <div className="mb-2 flex items-start justify-between gap-2">
            <h2 className="font-title text-sm font-semibold">Retrieval Explorer</h2>
            <button onClick={() => setOpen(false)} className="text-muted hover:text-ink" aria-label="Close help"><X size={14} /></button>
          </div>
          <p className="text-xs leading-relaxed text-muted">{ABOUT}</p>
          <div className="caption mb-1.5 mt-4 text-[10px]">Keyboard shortcuts</div>
          <dl className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1.5 text-xs">
            {SHORTCUTS.map(([key, what]) => (
              <div key={key} className="contents">
                <dt><kbd className="inline-block min-w-[24px] rounded border border-line bg-bg px-1.5 py-0.5 text-center font-mono text-[11px] text-ink">{key}</kbd></dt>
                <dd className="text-muted">{what}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
}

/** Shown whenever the last backend call failed: no silent fallback to fake data. */
function ErrorBanner() {
  const { error, reportError, refreshDatasets } = useExplorer();
  if (!error) return null;
  return (
    <div role="alert" className="flex w-full items-center gap-3 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-ink">
      <span className="font-mono text-xs text-red-400">backend error</span>
      <span className="min-w-0 flex-1 truncate">{error}</span>
      <button className="btn" onClick={() => { reportError(null); refreshDatasets(); }}>retry</button>
    </div>
  );
}

export default function Header() {
  const link = ({ isActive }: { isActive: boolean }) =>
    clsx("rounded px-3 py-1 text-sm transition-colors", isActive ? "bg-surface-2 text-ink" : "text-muted hover:text-ink");
  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-line py-4">
      <div className="mr-auto flex items-center gap-3">
        <img src="/cactus.png" alt="ProtoGraph" className="h-9 w-9 object-contain" />
        <div>
          <div className="caption">ProtoGraph · Retrieval Explorer</div>
          <h1 className="font-title text-xl font-bold leading-tight tracking-tight">Retrieval Explorer</h1>
        </div>
      </div>
      <nav className="flex items-center gap-0.5 rounded-md border border-line bg-bg p-0.5">
        <NavLink to="/" end className={link}>Explorer</NavLink>
        <NavLink to="/compare" className={link}>Compare</NavLink>
        <NavLink to="/prospector" className={link}>Prospector</NavLink>
        <NavLink to="/live" className={link}>Live</NavLink>
      </nav>
      <DatasetPicker />
      <Help />
      <ErrorBanner />
    </header>
  );
}
