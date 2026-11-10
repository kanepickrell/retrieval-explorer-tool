import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
  type Simulation, type SimulationLinkDatum, type SimulationNodeDatum,
} from "d3-force";
import { select } from "d3-selection";
import { zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from "d3-zoom";
import "d3-transition";
import { Crosshair, Filter, Maximize2, Shrink, Tag, X } from "lucide-react";
import clsx from "clsx";
import type { GraphEdge, GraphNode } from "@/lib/api";
import { clusterColor, type Signals } from "@/lib/signals";
import { useHotkeys } from "@/lib/hotkeys";
import ClusterRegions, { type Orb } from "./ClusterRegions";
import NodeSearch from "./NodeSearch";
import NodeInspector from "./NodeInspector";

// TODO(semantic-zoom): level-of-detail rendering — show region titles when zoomed
// out, fade in node labels as k grows, and aggregate edges between collapsed regions.

type SimNode = GraphNode & SimulationNodeDatum & { r: number };
type SimLink = SimulationLinkDatum<SimNode> & { id: string; weight: number };

type Props = {
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  signals: Signals | null;
  /** Bumps every time a new result lands, to drop into detail view + follow. */
  resultKey: number;
  highlight: { label: string; ids: string[] } | null;
  onClearHighlight: () => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
};

const nodeRadius = (degree: number) => 6 + Math.min(8, degree * 0.8);
/** cap on nodes laid out in result-only view (force layout + SVG cost grows with this) */
const RESULT_VIEW_MAX = 400;

function layoutOrbs(nodes: GraphNode[]): Orb[] {
  const counts = new Map<string, number>();
  for (const n of nodes) counts.set(n.cluster, (counts.get(n.cluster) ?? 0) + 1);
  const orbs = [...counts.entries()].map(([cluster, count], i, arr) => {
    const a = (i / arr.length) * Math.PI * 2;
    return { cluster, count, r: 24 + Math.sqrt(count) * 9, x: Math.cos(a) * 120, y: Math.sin(a) * 90 } as Orb & SimulationNodeDatum;
  });
  const sim = forceSimulation(orbs)
    .force("charge", forceManyBody().strength(-40))
    .force("collide", forceCollide<Orb>((d) => d.r + 34))
    .force("x", forceX(0).strength(0.06))
    .force("y", forceY(0).strength(0.09))
    .stop();
  for (let i = 0; i < 300; i++) sim.tick();
  return orbs.map((o) => ({ cluster: o.cluster, count: o.count, r: o.r, x: o.x!, y: o.y! }));
}

export default function GraphCanvas({ graph, signals, resultKey, highlight, onClearHighlight, selectedId, onSelect }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const simRef = useRef<Simulation<SimNode, SimLink> | null>(null);
  const posRef = useRef(new Map<string, { x: number; y: number }>());
  const simNodesRef = useRef<SimNode[]>([]);
  const simLinksRef = useRef<SimLink[]>([]);

  const [size, setSize] = useState({ w: 800, h: 600 });
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity);
  const [, setTick] = useState(0);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [hoveredOrb, setHoveredOrb] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [allLabels, setAllLabels] = useState(false);
  // result-only view: after a search, lay out just the retrieved nodes + their nearest
  // neighbours instead of every node in the dataset (thousands -> a few hundred)
  const [resultOnly, setResultOnly] = useState(true);

  const detail = expanded.size > 0;
  const orbs = useMemo(() => layoutOrbs(graph.nodes), [graph.nodes]);
  const orbBy = useMemo(() => new Map(orbs.map((o) => [o.cluster, o])), [orbs]);
  const nodeBy = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph.nodes]);

  const degree = useMemo(() => {
    const d = new Map<string, number>();
    for (const e of graph.edges) { d.set(e.source, (d.get(e.source) ?? 0) + 1); d.set(e.target, (d.get(e.target) ?? 0) + 1); }
    return d;
  }, [graph.edges]);

  const neighbours = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of graph.edges) {
      (m.get(e.source) ?? m.set(e.source, new Set()).get(e.source)!).add(e.target);
      (m.get(e.target) ?? m.set(e.target, new Set()).get(e.target)!).add(e.source);
    }
    return m;
  }, [graph.edges]);

  // ── size ──
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.max(200, width), h: Math.max(200, height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── zoom ──
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const z = zoom<SVGSVGElement, unknown>().scaleExtent([0.25, 4]).on("zoom", (e) => setTransform(e.transform));
    zoomRef.current = z;
    select(svg).call(z).on("dblclick.zoom", null);
    return () => { select(svg).on(".zoom", null); };
  }, []);

  /** Zoom to fit a set of points (each with a radius). */
  const fitTo = useCallback((pts: { x: number; y: number; r: number }[], duration = 650) => {
    const svg = svgRef.current, z = zoomRef.current;
    if (!svg || !z || pts.length === 0) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p.x - p.r); y0 = Math.min(y0, p.y - p.r); x1 = Math.max(x1, p.x + p.r); y1 = Math.max(y1, p.y + p.r); }
    const pad = 48;
    const k = Math.max(0.3, Math.min(1.6, Math.min((size.w - pad * 2) / Math.max(1, x1 - x0), (size.h - pad * 2) / Math.max(1, y1 - y0))));
    const t = zoomIdentity.translate(size.w / 2 - ((x0 + x1) / 2) * k, size.h / 2 - ((y0 + y1) / 2) * k).scale(k);
    select(svg).transition().duration(duration).call(z.transform, t);
  }, [size]);

  const fitVisible = useCallback((ids?: string[]) => {
    if (!detail) return fitTo(orbs.map((o) => ({ x: o.x, y: o.y, r: o.r * 1.3 + 20 })));
    const want = ids ? new Set(ids) : null;
    const pts = simNodesRef.current.filter((n) => !want || want.has(n.id)).map((n) => ({ x: n.x ?? 0, y: n.y ?? 0, r: n.r + 14 }));
    fitTo(pts.length ? pts : simNodesRef.current.map((n) => ({ x: n.x ?? 0, y: n.y ?? 0, r: n.r + 14 })));
  }, [detail, fitTo, orbs]);

  // re-fit whenever the graph or canvas size changes
  useEffect(() => {
    if (detail) fitVisible();
    else fitTo(orbs.map((o) => ({ x: o.x, y: o.y, r: o.r * 1.3 + 20 })), 0);
  }, [orbs, size]); // eslint-disable-line react-hooks/exhaustive-deps

  // reset when the dataset graph changes
  useEffect(() => { setExpanded(new Set()); setFocusId(null); posRef.current.clear(); }, [graph]);

  const retrievedIds = useMemo(() => {
    if (!signals) return [] as string[];
    return [...new Set([...signals.entity, ...signals.vector, ...signals.context])];
  }, [signals]);

  /** Nodes drawn in result-only view: retrieved + focused/highlighted nodes, plus the
   *  neighbours that connect the most of them (then highest degree), up to RESULT_VIEW_MAX.
   *  null = no scoping (browse by clusters). */
  const scopeIds = useMemo<Set<string> | null>(() => {
    const core = new Set<string>([...retrievedIds, ...(highlight?.ids ?? [])].filter((id) => nodeBy.has(id)));
    if (focusId && nodeBy.has(focusId)) core.add(focusId);
    if (!resultOnly || core.size === 0) return null;
    const out = new Set(core);
    if (focusId) for (const n of neighbours.get(focusId) ?? []) out.add(n);
    const links = new Map<string, number>();
    for (const id of core) for (const n of neighbours.get(id) ?? []) if (!out.has(n)) links.set(n, (links.get(n) ?? 0) + 1);
    const ranked = [...links.entries()].sort((a, b) => b[1] - a[1] || (degree.get(b[0]) ?? 0) - (degree.get(a[0]) ?? 0));
    for (const [id] of ranked) { if (out.size >= RESULT_VIEW_MAX) break; out.add(id); }
    return out;
  }, [resultOnly, retrievedIds, highlight, focusId, neighbours, degree, nodeBy]);
  const scopeKey = scopeIds ? [...scopeIds].sort().join("|") : "";

  // ── detail-view force simulation: the scoped result view, or members of expanded clusters ──
  const expandedKey = [...expanded].sort().join("|");
  useEffect(() => {
    simRef.current?.stop();
    const visible = scopeIds
      ? graph.nodes.filter((n) => scopeIds.has(n.id))
      : graph.nodes.filter((n) => expanded.has(n.cluster));
    const ids = new Set(visible.map((n) => n.id));
    const nodes: SimNode[] = visible.map((n) => {
      const prev = posRef.current.get(n.id);
      const orb = orbBy.get(n.cluster)!;
      const jitter = () => (Math.random() - 0.5) * 18;
      return { ...n, r: nodeRadius(degree.get(n.id) ?? 1), x: prev?.x ?? orb.x + jitter(), y: prev?.y ?? orb.y + jitter() };
    });
    const links: SimLink[] = graph.edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({ id: e.id, source: e.source, target: e.target, weight: e.weight }));
    simNodesRef.current = nodes;
    simLinksRef.current = links;
    if (nodes.length === 0) { setTick((t) => t + 1); return; }

    const sim = forceSimulation(nodes)
      .force("link", forceLink<SimNode, SimLink>(links).id((d) => d.id).distance(46).strength(0.18))
      .force("charge", forceManyBody().strength(-55).distanceMax(240))
      .force("x", forceX<SimNode>((d) => orbBy.get(d.cluster)!.x).strength(0.07))
      .force("y", forceY<SimNode>((d) => orbBy.get(d.cluster)!.y).strength(0.07))
      .force("collide", forceCollide<SimNode>((d) => d.r + 3))
      .on("tick", () => setTick((t) => t + 1));
    simRef.current = sim;
    return () => {
      sim.stop();
      for (const n of nodes) posRef.current.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
    };
  }, [expandedKey, scopeKey, graph, orbBy, degree]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── a new retrieval drops into detail view and (optionally) follows it ──
  useEffect(() => {
    if (!resultKey) return;
    setFocusId(null);
    if (!resultOnly) { setExpanded(new Set(orbs.map((o) => o.cluster))); return; }
    // result-only: open just the clusters the result touches; a result with no graph
    // nodes (vector / bm25) stays in the cluster overview instead of laying out everything
    const touched = new Set(retrievedIds.map((id) => nodeBy.get(id)?.cluster).filter(Boolean) as string[]);
    setExpanded(touched);
  }, [resultKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!resultKey || !follow) return;
    const t = window.setTimeout(() => fitVisible(retrievedIds.length ? retrievedIds : undefined), 900);
    return () => window.clearTimeout(t);
  }, [resultKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // highlight from a chunk / Prospector document: make sure it's visible
  useEffect(() => {
    if (!highlight || highlight.ids.length === 0) return;
    const clusters = new Set(highlight.ids.map((id) => nodeBy.get(id)?.cluster).filter(Boolean) as string[]);
    setExpanded((prev) => (([...clusters].every((c) => prev.has(c))) ? prev : new Set([...prev, ...clusters])));
    if (follow) {
      const t = window.setTimeout(() => fitVisible(highlight.ids), 700);
      return () => window.clearTimeout(t);
    }
  }, [highlight, graph]); // re-apply after the dataset graph (re)loads  eslint-disable-line react-hooks/exhaustive-deps

  const focusNode = (id: string | null) => {
    setFocusId(id);
    if (!id) return;
    onSelect(id);
    const n = nodeBy.get(id);
    if (!n) return;
    const wasOpen = expanded.has(n.cluster);
    if (!wasOpen) setExpanded((prev) => new Set([...prev, n.cluster]));
    const hood = [id, ...(neighbours.get(id) ?? [])];
    window.setTimeout(() => fitVisible(hood), wasOpen ? 0 : 750);
  };

  // ── emphasis: what stays bright (everything else dims to ~8%) ──
  const emphasis = useMemo<Set<string> | null>(() => {
    if (focusId) return new Set([focusId, ...(neighbours.get(focusId) ?? [])]);
    if (highlight) return new Set(highlight.ids);
    if (signals) return new Set(retrievedIds);
    return null;
  }, [focusId, highlight, signals, retrievedIds, neighbours]);

  const edgeEmphasized = (l: SimLink) => {
    const s = (l.source as SimNode).id, t = (l.target as SimNode).id;
    if (focusId) return s === focusId || t === focusId;
    if (highlight) return emphasis!.has(s) && emphasis!.has(t);
    if (signals) return signals.relation.has(l.id);
    return true;
  };

  const nodeFill = (n: SimNode) => {
    if (signals) {
      if (signals.context.has(n.id)) return "#f4f7fa";
      if (signals.entity.has(n.id)) return "var(--local)";
      if (signals.vector.has(n.id)) return "var(--vector)";
      return "#5d6b7c";
    }
    return clusterColor(n.cluster);
  };

  const highlightSet = useMemo(() => new Set(highlight?.ids ?? []), [highlight]);
  const showLabel = (n: SimNode) =>
    allLabels || n.id === hoveredNode || n.id === selectedId || n.id === focusId || highlightSet.has(n.id) ||
    (!!signals && !focusId && !highlight && emphasis!.has(n.id)) || (!!focusId && emphasis!.has(n.id));

  const collapseAll = () => {
    setExpanded(new Set());
    setFocusId(null);
    onSelect(null);
    window.setTimeout(() => fitTo(orbs.map((o) => ({ x: o.x, y: o.y, r: o.r * 1.3 + 20 }))), 50);
  };

  useHotkeys({
    f: () => fitVisible(),
    l: () => setAllLabels((v) => !v),
    c: () => { if (detail) collapseAll(); },
    Escape: () => setFocusId(null),
  });

  const selected = selectedId ? nodeBy.get(selectedId) : undefined;
  const visibleNodes = simNodesRef.current;
  const visibleLinks = simLinksRef.current;

  const labelPriority = (n: SimNode) =>
    (n.id === selectedId ? 1000 : n.id === focusId ? 900 : n.id === hoveredNode ? 800
      : highlightSet.has(n.id) ? 600 : signals?.context.has(n.id) ? 500
      : signals?.entity.has(n.id) ? 400 : signals?.vector.has(n.id) ? 300 : 0) + (degree.get(n.id) ?? 0);

  // Positions move every tick, so this recomputes each render (cheap at this graph size).
  const labels = placeLabels(
    visibleNodes.filter(showLabel),
    visibleNodes.filter((n) => !emphasis || emphasis.has(n.id)),
    labelPriority,
    transform.k,
  );

  return (
    <section className="card flex h-full min-h-0 flex-col p-0">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2.5">
        <div className="mr-auto flex items-baseline gap-2">
          <h2 className="font-title text-sm font-semibold">Knowledge graph</h2>
          <span className="font-mono text-[11px] text-muted">
            {scopeIds ? `showing ${visibleNodes.length} of ${graph.nodes.length} nodes` : `${graph.nodes.length} nodes · ${graph.edges.length} edges`}
          </span>
        </div>
        <NodeSearch nodes={graph.nodes} focusId={focusId} onFocus={focusNode} />
        <div className="flex items-center rounded-md border border-line bg-bg p-0.5">
          <ToolToggle on={follow} onClick={() => setFollow((f) => !f)} icon={<Crosshair size={13} />} label="follow"
            title="Follow: auto-zoom to each new result" />
          <ToolToggle on={allLabels} onClick={() => setAllLabels((f) => !f)} icon={<Tag size={13} />} label="labels"
            title="Label every node that fits (l)" />
          <ToolToggle on={resultOnly} onClick={() => setResultOnly((v) => !v)} icon={<Filter size={13} />} label="result only"
            title={`Result only: draw just the retrieved nodes and their nearest neighbours (up to ${RESULT_VIEW_MAX}); off = full graph`} />
          <span className="mx-0.5 h-4 w-px bg-line" />
          <ToolButton onClick={() => fitVisible()} title="Fit to view (f)"><Maximize2 size={13} /></ToolButton>
          <ToolButton onClick={collapseAll} disabled={!detail} title="Collapse all regions (c)"><Shrink size={13} /></ToolButton>
        </div>
      </div>

      <div ref={wrapRef} className="relative min-h-0 flex-1 overflow-hidden rounded-b-card border-t border-line bg-bg/60">
        <svg ref={svgRef} width={size.w} height={size.h} className="absolute inset-0 block cursor-grab active:cursor-grabbing"
          onClick={(e) => { if (e.target === e.currentTarget) { onSelect(null); } }}>
          <g transform={transform.toString()}>
            <ClusterRegions orbs={orbs} expanded={expanded} detail={detail} hovered={hoveredOrb} onHover={setHoveredOrb}
              onExpand={(c) => { setExpanded((prev) => new Set([...prev, c])); setHoveredOrb(null); }} />

            {/* faint region titles while browsing; hidden once something is emphasised so they never sit on top of results */}
            {!emphasis && orbs.filter((o) => expanded.has(o.cluster)).map((o) => (
              <text key={o.cluster} x={o.x} y={o.y - o.r - 22} textAnchor="middle" style={{ fontSize: 10 / transform.k }}
                className="pointer-events-none select-none fill-muted font-mono uppercase tracking-widest" opacity={0.4}>
                {o.cluster}
              </text>
            ))}

            <g>
              {visibleLinks.map((l) => {
                const s = l.source as SimNode, t = l.target as SimNode;
                if (s.x == null || t.x == null) return null;
                const emph = edgeEmphasized(l);
                const rel = signals?.relation.has(l.id);
                return (
                  <line key={l.id} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                    stroke={rel ? "var(--global)" : "#3a4a5e"} strokeWidth={(rel ? 1.3 : 1) / Math.max(1, transform.k)}
                    opacity={emphasis ? (emph ? 0.9 : 0.08) : 0.6} style={{ transition: "opacity 300ms" }} />
                );
              })}
            </g>
            <g>
              {visibleNodes.map((n) => {
                const dim = emphasis && !emphasis.has(n.id);
                const isSel = n.id === selectedId;
                return (
                  <g key={n.id} transform={`translate(${n.x ?? 0},${n.y ?? 0})`} style={{ cursor: "pointer", opacity: dim ? 0.08 : 1, transition: "opacity 300ms" }}
                    onMouseEnter={() => setHoveredNode(n.id)} onMouseLeave={() => setHoveredNode(null)}
                    onClick={(e) => { e.stopPropagation(); onSelect(isSel ? null : n.id); }}>
                    {(isSel || highlightSet.has(n.id)) && (
                      <circle r={n.r + 4} fill="none" stroke={isSel ? "var(--local)" : "var(--ink)"} strokeWidth={1.5} opacity={0.9} />
                    )}
                    <circle r={n.r} fill={nodeFill(n)} stroke="var(--bg)" strokeWidth={1.5} />
                  </g>
                );
              })}
            </g>
            {/* labels in their own top layer so a later node never paints over an earlier label */}
            <g>
              {labels.map((lb) => (
                <text key={lb.id} x={lb.x} y={lb.y} textAnchor={lb.anchor} className="cursor-pointer select-none fill-ink font-mono"
                  style={{ fontSize: 10.5 / transform.k, paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 / transform.k }}
                  onMouseEnter={() => setHoveredNode(lb.id)} onMouseLeave={() => setHoveredNode(null)}
                  onClick={(e) => { e.stopPropagation(); onSelect(lb.id === selectedId ? null : lb.id); }}>
                  {lb.text}
                </text>
              ))}
            </g>
          </g>
        </svg>

        {highlight && (
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-md border border-line bg-surface/95 px-2.5 py-1 text-xs">
            <span className="h-1.5 w-1.5 rounded-full bg-ink" />
            <span className="max-w-[260px] truncate">{highlight.label}</span>
            <button onClick={onClearHighlight} className="text-muted hover:text-ink" aria-label="Clear highlight"><X size={12} /></button>
          </div>
        )}
        {selected && (
          <NodeInspector
            node={selected}
            degree={degree.get(selected.id) ?? 0}
            signals={signals}
            neighbours={[...(neighbours.get(selected.id) ?? [])].map((id) => nodeBy.get(id)).filter((n): n is GraphNode => !!n)}
            onPick={focusNode}
            onClose={() => onSelect(null)}
          />
        )}

        <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-end justify-between gap-2">
          <Legend signals={!!signals} clusters={orbs.map((o) => o.cluster)} />
          <span className="font-mono text-[10px] text-muted/70">
            {detail ? "scroll to zoom · drag to pan · click a node" : "click a region to expand it"}
          </span>
        </div>
      </div>
    </section>
  );
}

type Box = { x0: number; y0: number; x1: number; y1: number };
type PlacedLabel = { id: string; text: string; x: number; y: number; anchor: "start" | "middle" | "end" };

const boxesOverlap = (a: Box, b: Box) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;
function circleHitsBox(c: SimNode, b: Box) {
  const cx = c.x ?? 0, cy = c.y ?? 0;
  const dx = cx - Math.max(b.x0, Math.min(cx, b.x1));
  const dy = cy - Math.max(b.y0, Math.min(cy, b.y1));
  return dx * dx + dy * dy < c.r * c.r;
}

/**
 * Greedy label placement, highest priority first. Each label tries below,
 * above, right, then left of its node, and takes the first slot that hits
 * neither an already-placed label nor another visible node. Labels with no
 * free slot are dropped, except must-show ones (selected / focused / hovered),
 * which fall back to "below". "All labels" therefore means "every label that fits".
 * Works in graph units; label size is constant on screen, so it scales by 1/k.
 */
function placeLabels(candidates: SimNode[], obstacles: SimNode[], priority: (n: SimNode) => number, k: number): PlacedLabel[] {
  const charW = 6.4 / k, lh = 12 / k, gap = 3 / k, pad = 1.5 / k;
  const placed: Box[] = [];
  const out: PlacedLabel[] = [];
  for (const n of [...candidates].sort((a, b) => priority(b) - priority(a))) {
    const x = n.x ?? 0, y = n.y ?? 0, w = n.label.length * charW;
    const slots: { x0: number; y0: number; tx: number; anchor: PlacedLabel["anchor"] }[] = [
      { x0: x - w / 2, y0: y + n.r + gap, tx: x, anchor: "middle" },
      { x0: x - w / 2, y0: y - n.r - gap - lh, tx: x, anchor: "middle" },
      { x0: x + n.r + gap, y0: y - lh / 2, tx: x + n.r + gap, anchor: "start" },
      { x0: x - n.r - gap - w, y0: y - lh / 2, tx: x - n.r - gap, anchor: "end" },
    ];
    const boxOf = (s: (typeof slots)[number]): Box => ({ x0: s.x0 - pad, y0: s.y0 - pad, x1: s.x0 + w + pad, y1: s.y0 + lh + pad });
    const free = (b: Box) => !placed.some((p) => boxesOverlap(p, b)) && !obstacles.some((o) => o.id !== n.id && circleHitsBox(o, b));
    let slot = slots.find((s) => free(boxOf(s)));
    if (!slot && priority(n) >= 800) slot = slots[0];
    if (!slot) continue;
    placed.push(boxOf(slot));
    out.push({ id: n.id, text: n.label, x: slot.tx, y: slot.y0 + lh * 0.8, anchor: slot.anchor });
  }
  return out;
}

function ToolToggle({ on, onClick, icon, label, title }: { on: boolean; onClick: () => void; icon: ReactNode; label: string; title: string }) {
  return (
    <button onClick={onClick} aria-pressed={on} title={title}
      className={clsx("flex items-center gap-1.5 rounded px-2 py-1 font-mono text-[11px] transition-colors",
        on ? "bg-surface-2 text-local" : "text-muted hover:text-ink")}>
      {icon}{label}
    </button>
  );
}

function ToolButton({ onClick, disabled, title, children }: { onClick: () => void; disabled?: boolean; title: string; children: ReactNode }) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} aria-label={title}
      className="rounded p-1.5 text-muted transition-colors hover:text-ink disabled:pointer-events-none disabled:opacity-30">
      {children}
    </button>
  );
}

/** Before a query nodes are coloured by cluster; after one, by retrieval signal. */
function Legend({ signals, clusters }: { signals: boolean; clusters: string[] }) {
  const items: [string, string, boolean?][] = signals
    ? [["entity", "var(--local)"], ["relation", "var(--global)", true], ["vector", "var(--vector)"], ["in final context", "#f4f7fa"]]
    : clusters.map((c): [string, string] => [c, clusterColor(c)]);
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 rounded-md border border-line bg-surface/90 px-2.5 py-1.5 font-mono text-[10px] text-muted">
      {items.map(([label, color, line]) => (
        <span key={label} className="flex items-center gap-1.5">
          {line ? <span className="h-[2px] w-3 rounded" style={{ background: color }} /> : <span className="h-2 w-2 rounded-full" style={{ background: color }} />}
          {label}
        </span>
      ))}
    </div>
  );
}
