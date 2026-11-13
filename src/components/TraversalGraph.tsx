import { useEffect, useMemo, useRef, useState } from "react";
import {
  forceCollide, forceLink, forceSimulation, forceY,
  type SimulationLinkDatum, type SimulationNodeDatum,
} from "d3-force";
import clsx from "clsx";
import { Pause, Play, Radio } from "lucide-react";
import type { AgentTraversal, TraversalNode } from "@/lib/api";

/**
 * The agent's traversal, drawn on the records it touched. Left to right: the question,
 * its hypotheses, the records / terms each hypothesis named, then the scope, the hits,
 * what was read and the result. Real records keep their graph edges; the reasoning is
 * drawn on top. `step` replays the run up to that event (null = follow the latest).
 */

// strength of a role, for nodes that collect several over the run (hypotheses take the latest)
const RANK: Record<string, number> = {
  scope: 1, hit_out: 2, term: 3, missing: 3, dropped: 3, hit_in: 4, seed: 5,
  evidence: 6, confirmed: 7, result: 8, query: 9, goal: 9,
};
const HYP_ROLES = new Set(["pending", "validated", "weak", "rejected", "chosen", "failed"]);

export const ROLE_STYLE: Record<string, { color: string; label: string }> = {
  query: { color: "var(--local)", label: "question" },
  goal: { color: "var(--local)", label: "goal stated" },
  pending: { color: "var(--muted)", label: "hypothesis, unchecked" },
  validated: { color: "var(--good)", label: "hypothesis held up" },
  weak: { color: "var(--warn)", label: "hypothesis weak" },
  rejected: { color: "var(--bad)", label: "hypothesis rejected" },
  chosen: { color: "var(--local)", label: "route taken" },
  failed: { color: "var(--bad)", label: "route failed the check" },
  seed: { color: "var(--global)", label: "record it named" },
  term: { color: "var(--vector)", label: "term it named" },
  missing: { color: "var(--bad)", label: "named, doesn't exist" },
  dropped: { color: "var(--bad)", label: "not in the conversation" },
  scope: { color: "var(--muted)", label: "in scope" },
  hit_in: { color: "var(--good)", label: "hit inside scope" },
  hit_out: { color: "var(--warn)", label: "hit outside scope" },
  evidence: { color: "var(--vector)", label: "read as evidence" },
  confirmed: { color: "var(--good)", label: "confirmed by the check" },
  result: { color: "var(--good)", label: "in the result" },
};

const EDGE_STYLE: Record<string, { stroke: string; dash?: string; width: number; opacity: number }> = {
  hypothesis: { stroke: "var(--local)", width: 1.4, opacity: 0.7 },
  names: { stroke: "var(--global)", width: 1.2, opacity: 0.7 },
  scope: { stroke: "var(--muted)", dash: "2 3", width: 0.8, opacity: 0.5 },
  graph: { stroke: "var(--muted)", width: 0.8, opacity: 0.35 },
  contains: { stroke: "var(--vector)", dash: "1 3", width: 0.9, opacity: 0.6 },
  found: { stroke: "var(--good)", dash: "4 3", width: 1, opacity: 0.55 },
  read: { stroke: "var(--vector)", width: 1.4, opacity: 0.8 },
  answer: { stroke: "var(--good)", width: 2, opacity: 0.85 },
};

export function rolesAt(t: AgentTraversal, upto: number): Map<string, string> {
  const out = new Map<string, string>();
  for (const st of t.steps) {
    if (st.i > upto) break;
    for (const [id, role] of Object.entries(st.roles)) {
      const cur = out.get(id);
      if (!cur || HYP_ROLES.has(role) || (RANK[role] ?? 0) >= (RANK[cur] ?? 0)) out.set(id, role);
    }
  }
  return out;
}

const tierOf = (n: TraversalNode, role: string | undefined) =>
  n.type === "Question" ? 0 : n.type === "Hypothesis" ? 1
    : role === "seed" || role === "term" || role === "missing" || role === "dropped" ? 2 : 3;

type SimN = TraversalNode & SimulationNodeDatum & { tier: number };
type SimL = SimulationLinkDatum<SimN> & { kind: string };

const H = 380;      // minimum height; grows with the busiest column
const ROW = 24;     // vertical slot per node

export default function TraversalGraph({ t, step, onStep, dataset }: {
  t: AgentTraversal; step: number | null; onStep: (i: number | null) => void; dataset: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const pos = useRef(new Map<string, { x: number; y: number }>());
  const [hover, setHover] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(Math.max(420, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const lastI = t.steps.length ? t.steps[t.steps.length - 1].i : 0;
  const cur = step ?? lastI;
  const finalRoles = useMemo(() => rolesAt(t, Infinity), [t]);
  const roles = useMemo(() => rolesAt(t, cur), [t, cur]);
  const changed = useMemo(() => new Set(Object.keys(t.steps.find((s) => s.i === cur)?.roles ?? {})), [t, cur]);
  const gold = useMemo(() => new Set(t.gold?.nodes ?? []), [t.gold]);

  // Column layout, once per node set (stable while scrubbing): x is pinned to the node's
  // column (question | hypotheses | named | scope·hits·result); within a column nodes start
  // evenly spaced, strongest roles first, and links only nudge related nodes vertically.
  const nodeKey = t.nodes.map((n) => n.id).join("|") + `#${t.edges.length}#${w}`;
  const layout = useMemo(() => {
    const m = 60, labelRoom = 150, colW = (w - m - labelRoom) / 3;
    const cols: SimN[][] = [[], [], [], []];
    for (const n of t.nodes) {
      const tier = tierOf(n, finalRoles.get(n.id));
      cols[tier].push({ ...n, tier });
    }
    const h = Math.max(H, Math.max(...cols.map((c) => c.length)) * ROW + 70);
    const rank = (n: SimN) => -(RANK[finalRoles.get(n.id) ?? ""] ?? 0) - (gold.has(n.id) ? 0.5 : 0);
    for (const col of cols) {
      col.sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label));
      col.forEach((n, k) => {
        n.fx = m + n.tier * colW;
        n.x = n.fx;
        n.y = pos.current.get(n.id)?.y ?? 40 + (k + 0.5) * ((h - 60) / col.length);
      });
    }
    const nodes = cols.flat();
    const home = new Map(nodes.map((n) => [n.id, n.y!]));
    const ids = new Set(nodes.map((n) => n.id));
    const links: SimL[] = t.edges.filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, kind: e.kind }));
    const sim = forceSimulation(nodes)
      .force("link", forceLink<SimN, SimL>(links).id((d) => d.id).distance(ROW * 2).strength(0.04))
      .force("collide", forceCollide(ROW / 2))
      .force("y", forceY<SimN>((d) => home.get(d.id)!).strength(0.25))
      .stop();
    for (let i = 0; i < 200; i++) sim.tick();
    for (const n of nodes) {
      n.y = Math.max(34, Math.min(h - 16, n.y!));
      pos.current.set(n.id, { x: n.x!, y: n.y });
    }
    return { nodes, by: new Map(nodes.map((n) => [n.id, n])), h, colX: [0, 1, 2, 3].map((k) => m + k * colW) };
  }, [nodeKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // play: step through the run from the start
  useEffect(() => {
    if (!playing) return;
    const order = t.steps.map((s) => s.i);
    let k = step == null || step >= lastI ? 0 : Math.max(0, order.indexOf(step));
    onStep(order[k]);
    const id = window.setInterval(() => {
      k += 1;
      if (k >= order.length) { setPlaying(false); onStep(null); return; }
      onStep(order[k]);
    }, 700);
    return () => window.clearInterval(id);
  }, [playing]); // eslint-disable-line react-hooks/exhaustive-deps

  const visible = (id: string) => roles.has(id) || gold.has(id);
  const hovered = hover ? layout.by.get(hover) : undefined;
  const stepKind = t.steps.find((s) => s.i === cur)?.kind;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
        <button className="btn h-7" onClick={() => setPlaying((p) => !p)} title="Replay the run step by step">
          {playing ? <><Pause size={12} /> Pause</> : <><Play size={12} /> Replay</>}
        </button>
        <input type="range" min={t.steps[0]?.i ?? 1} max={lastI || 1} value={cur}
          onChange={(e) => { setPlaying(false); const v = Number(e.target.value); onStep(v >= lastI ? null : v); }}
          className="w-48 accent-[var(--local)]" aria-label="Step" />
        <span className="tabular text-ink">step {cur} / {lastI}</span>
        {stepKind && <span>· {stepKind}</span>}
        <button className={clsx("btn h-7", step == null && "border-local/60 text-local")} onClick={() => { setPlaying(false); onStep(null); }}
          title="Follow the latest step (live while the run is going)">
          <Radio size={12} className={clsx(step == null && t.status === "running" && "animate-pulse")} /> latest
        </button>
        {t.gold && (
          <span className={clsx("ml-auto rounded border px-1.5 py-px",
            t.gold.hit5 == null ? "border-line" : t.gold.hit5 ? "border-good/60 text-good" : "border-bad/60 text-bad")}
            title={`gold question ${t.gold.id} (${t.gold.type}): ${t.gold.goldIds.join(", ")}`}>
            gold {t.gold.hit5 == null ? "pending" : t.gold.hit5 ? "reached in top 5" : "missed in top 5"}
          </span>
        )}
      </div>

      <div ref={wrapRef} className="relative overflow-hidden rounded-md border border-line bg-bg">
        <svg width={w} height={layout.h} role="img" aria-label="Agent traversal graph">
          {["question", "hypotheses", "named", "scope · hits · result"].map((label, k) => (
            <text key={label} x={layout.colX[k] - 8} y={16} fontSize={10} fill="var(--muted)" opacity={0.7}
              style={{ textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "JetBrains Mono, monospace" }}>{label}</text>
          ))}
          <defs>
            <marker id="tg-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--muted)" />
            </marker>
          </defs>
          {t.edges.map((e, i) => {
            const a = layout.by.get(e.source), b = layout.by.get(e.target);
            if (!a || !b || !visible(e.source) || !visible(e.target)) return null;
            const st = EDGE_STYLE[e.kind] ?? EDGE_STYLE.graph;
            const dim = hover && hover !== e.source && hover !== e.target;
            return (
              <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={st.stroke} strokeWidth={st.width}
                strokeDasharray={st.dash} opacity={dim ? 0.08 : st.opacity}
                markerEnd={e.kind === "graph" ? undefined : "url(#tg-arrow)"} />
            );
          })}
          {layout.nodes.map((n) => {
            if (!visible(n.id)) return null;
            const role = roles.get(n.id);
            const color = role ? ROLE_STYLE[role]?.color ?? "var(--muted)" : "var(--muted)";
            const isNew = changed.has(n.id);
            const dim = hover && hover !== n.id;
            const isGold = gold.has(n.id);
            const r = n.type === "Question" ? 10 : role === "result" || role === "confirmed" ? 8 : role === "scope" ? 4.5 : 6.5;
            const label = n.type === "Hypothesis" ? `${n.label} · ${role ?? ""}` : n.type === "Question" ? "question" : n.label;
            return (
              <g key={n.id} transform={`translate(${n.x},${n.y})`} opacity={dim ? 0.3 : role ? 1 : 0.45}
                onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}
                onClick={() => n.doc && window.open(`/doc?doc=${encodeURIComponent(n.doc)}&dataset=${encodeURIComponent(dataset)}`, "_blank")}
                className={n.doc ? "cursor-pointer" : undefined}>
                {isNew && <circle r={r + 7} fill="none" stroke={color} strokeWidth={1.5} opacity={0.6} className="animate-ping" style={{ transformOrigin: "center", transformBox: "fill-box" }} />}
                {isGold && <circle r={r + 4} fill="none" stroke="var(--global)" strokeWidth={1.4} strokeDasharray="3 2" />}
                {n.type === "Hypothesis" || n.type === "Question" ? (
                  <rect x={-r - 2} y={-r} width={2 * r + 4} height={2 * r} rx={4} fill={role ? color : "none"}
                    fillOpacity={role === "pending" ? 0.25 : 0.85} stroke={color} strokeWidth={role === "chosen" ? 2.5 : 1.2} />
                ) : (
                  <circle r={r} fill={role === "scope" || role === "dropped" || role === "missing" || !role ? "var(--bg)" : color}
                    stroke={color} strokeWidth={role === "result" ? 2.2 : 1.3} />
                )}
                {(role === "dropped" || role === "missing") && <line x1={-r} y1={r} x2={r} y2={-r} stroke="var(--bad)" strokeWidth={1.4} />}
                {role !== "scope" || hover === n.id ? (
                  <text x={r + 5} y={3.5} fontSize={10.5} fill={role === "dropped" || role === "missing" ? "var(--bad)" : "var(--ink)"}
                    className="pointer-events-none select-none" style={{ textDecoration: role === "dropped" || role === "missing" ? "line-through" : undefined }}>
                    {label.length > 30 ? `${label.slice(0, 29)}…` : label}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
        {hovered && (
          <div className="pointer-events-none absolute right-2 top-7 max-w-[340px] rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-lg">
            <div className="font-mono text-[10px] uppercase tracking-wide text-muted">
              {hovered.type}{roles.get(hovered.id) ? ` · ${ROLE_STYLE[roles.get(hovered.id)!]?.label ?? roles.get(hovered.id)}` : ""}
              {gold.has(hovered.id) ? " · gold (expected)" : ""}
            </div>
            <div className="text-ink">{hovered.label}</div>
            {hovered.claim && <div className="mt-0.5 text-muted">{hovered.claim}</div>}
            {hovered.answer_type && <div className="mt-0.5 font-mono text-[10px] text-muted">{hovered.answer_type} → {hovered.target}</div>}
            {hovered.doc && <div className="mt-0.5 font-mono text-[10px] text-muted">{hovered.doc} · click to open</div>}
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] text-muted">
        {["validated", "weak", "rejected", "chosen", "seed", "term", "dropped", "scope", "hit_in", "hit_out", "evidence", "result"].map((k) => (
          <span key={k} className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: ROLE_STYLE[k].color, opacity: k === "scope" ? 0.6 : 1 }} />
            {ROLE_STYLE[k].label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full border border-dashed" style={{ borderColor: "var(--global)" }} />gold doc (expected)
        </span>
      </div>
    </div>
  );
}
