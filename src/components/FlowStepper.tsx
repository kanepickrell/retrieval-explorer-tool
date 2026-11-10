import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import clsx from "clsx";

const FLOW = [
  { name: "Prospector", role: "data", to: "/prospector" },
  { name: "Explorer", role: "retrieval", to: "/" },
  { name: "Compare", role: "which technique wins", to: "/compare" },
] as const;

/** Prospector (data) → Explorer (retrieval) → Compare (which technique wins). */
export default function FlowStepper({ active }: { active: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
      {FLOW.map((s, i) => (
        <li key={s.name} className="flex items-center gap-2">
          <Link to={s.to}
            className={clsx("rounded-full border px-2.5 py-0.5 transition-colors",
              i === active ? "border-local/60 text-ink" : "border-line text-muted hover:text-ink")}>
            {s.name} <span className="text-muted">({s.role})</span>
          </Link>
          {i < FLOW.length - 1 && <ArrowRight size={12} className="text-muted/60" />}
        </li>
      ))}
    </ol>
  );
}
