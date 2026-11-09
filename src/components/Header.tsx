import { NavLink } from "react-router-dom";
import { ChevronDown, CircleHelp, Database } from "lucide-react";
import clsx from "clsx";
import { useExplorer } from "@/lib/explorer-context";

const HELP =
  "Run a question through a retrieval technique and inspect the subgraph it touched, " +
  "the chunks it pulled, the steps it took and how it scored.";

/** Dataset is app-wide context (Explorer queries it, Prospector browses it), so it lives in the header. */
function DatasetPicker() {
  const { datasets, datasetId, setDatasetId } = useExplorer();
  return (
    <label className="relative flex items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm hover:border-muted/50">
      <Database size={14} className="text-muted" />
      <span className="sr-only">Dataset</span>
      <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}
        className="cursor-pointer appearance-none bg-transparent pr-5 text-ink outline-none">
        {datasets.map((d) => (
          <option key={d.id} value={d.id} className="bg-surface">{d.name}{d.labeled ? " · gold" : ""}</option>
        ))}
      </select>
      <ChevronDown size={14} className="pointer-events-none absolute right-2 text-muted" />
    </label>
  );
}

export default function Header() {
  const link = ({ isActive }: { isActive: boolean }) =>
    clsx("rounded px-3 py-1 text-sm transition-colors", isActive ? "bg-surface-2 text-ink" : "text-muted hover:text-ink");
  return (
    <header className="flex flex-wrap items-center gap-x-5 gap-y-3 border-b border-line py-4">
      <div className="mr-auto flex items-center gap-3">
        <img src="/logo.svg" alt="ProtoGraph" className="h-9 w-9" />
        <div>
          <div className="caption">ProtoGraph · Retrieval Explorer</div>
          <h1 className="font-title text-xl font-bold leading-tight tracking-tight">Retrieval Explorer</h1>
        </div>
      </div>
      <nav className="flex items-center gap-0.5 rounded-md border border-line bg-bg p-0.5">
        <NavLink to="/" end className={link}>Explorer</NavLink>
        <NavLink to="/prospector" className={link}>Prospector</NavLink>
      </nav>
      <DatasetPicker />
      <button className="rounded-md p-1.5 text-muted hover:text-ink" title={HELP} aria-label="Help"><CircleHelp size={17} /></button>
    </header>
  );
}
