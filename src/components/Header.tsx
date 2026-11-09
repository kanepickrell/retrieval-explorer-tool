import { NavLink } from "react-router-dom";
import { CircleHelp, Moon } from "lucide-react";
import clsx from "clsx";

const HELP =
  "Run a question through a retrieval technique and inspect the subgraph it touched, " +
  "the chunks it pulled, the steps it took and how it scored.";

export default function Header() {
  const link = ({ isActive }: { isActive: boolean }) =>
    clsx("rounded-md px-2.5 py-1 text-sm transition-colors", isActive ? "bg-surface-2 text-ink" : "text-muted hover:text-ink");
  return (
    <header className="flex flex-wrap items-center gap-4 border-b border-line py-4">
      <img src="/logo.svg" alt="ProtoGraph" className="h-9 w-9" />
      <div className="mr-auto">
        <div className="caption">ProtoGraph · Retrieval Explorer</div>
        <h1 className="font-title text-xl font-bold leading-tight tracking-tight">Retrieval Explorer</h1>
      </div>
      <nav className="flex items-center gap-1">
        <NavLink to="/" end className={link}>Explorer</NavLink>
        <NavLink to="/prospector" className={link}>Prospector</NavLink>
      </nav>
      <div className="flex items-center gap-1 text-muted">
        <button className="rounded-md p-1.5 hover:text-ink" title={HELP} aria-label="Help"><CircleHelp size={17} /></button>
        <button className="rounded-md p-1.5 opacity-60" title="Dark theme only in v1" aria-label="Theme" disabled><Moon size={16} /></button>
      </div>
    </header>
  );
}
