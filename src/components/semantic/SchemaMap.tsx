import { useMemo, useState } from "react";
import clsx from "clsx";
import type { SchemaMetrics, SemanticSchema } from "@/lib/api";

/**
 * The semantic model as a map: one column per top-level class group (Actor, Activity,
 * Artifact, ...) with its classes stacked inside, and the relations drawn between them.
 * Counts are the selected dataset's instances; dashed = proposed (not built yet);
 * red = edges that break the relation's from/to. Hover a relation to trace it.
 */

const BOX_W = 132, BOX_H = 42, COL_W = 236, ROW_H = 64, TOP = 52, PAD = 16;

type Box = { name: string; x: number; y: number; col: number; abstract: boolean; proposed: boolean; unknown?: boolean };
type Edge = { rel: string; from: string; to: string; i: number; proposed: boolean; bad: boolean; label: boolean };
type Group = { name: string; x: number; y: number; h: number; abstract: boolean };

export type MapSelection = { kind: "class" | "relation"; name: string } | null;

export default function SchemaMap({ schema, metrics, selected, onSelect }: {
  schema: SemanticSchema; metrics: SchemaMetrics | null; selected: MapSelection; onSelect: (s: MapSelection) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);

  const { boxes, groups, edges, width, height } = useMemo(() => {
    const classes = schema.classes ?? {};
    const names = Object.keys(classes);
    const children = (p: string) => names.filter((c) => classes[c]?.parent === p);
    const roots = names.filter((c) => !classes[c]?.parent || !classes[classes[c]!.parent!]);
    // groups with members first (in file order), then lone classes in one shared column
    const grouped = roots.filter((r) => children(r).length > 0);
    const lone = roots.filter((r) => children(r).length === 0);
    const boxes = new Map<string, Box>();
    const groups: Group[] = [];
    let col = 0, maxRows = 1;
    const place = (c: string, x: number, row: number) =>
      boxes.set(c, { name: c, x, y: TOP + row * ROW_H, col, abstract: !!classes[c]?.abstract,
                     proposed: classes[c]?.status === "proposed" });
    for (const r of grouped) {
      const x = PAD + col * COL_W;
      const members: string[] = [];
      const walk = (p: string) => { for (const c of children(p)) { members.push(c); walk(c); } };
      walk(r);
      if (!classes[r]?.abstract) members.unshift(r);
      members.forEach((c, row) => place(c, x, row));
      groups.push({ name: r, x: x - 8, y: TOP - 30, h: members.length * ROW_H + 22, abstract: !!classes[r]?.abstract });
      if (classes[r]?.abstract) boxes.set(r, { name: r, x, y: TOP - 30, col, abstract: true, proposed: false });
      maxRows = Math.max(maxRows, members.length);
      col++;
    }
    const unknown = Object.keys(metrics?.unknown_classes ?? {});
    if (lone.length || unknown.length) {
      const x = PAD + col * COL_W;
      lone.forEach((c, row) => place(c, x, row));
      unknown.forEach((c, k) => boxes.set(c, { name: c, x, y: TOP + (lone.length + k) * ROW_H, col, abstract: false,
                                                proposed: false, unknown: true }));
      maxRows = Math.max(maxRows, lone.length + unknown.length);
      col++;
    }

    const edges: Edge[] = [];
    const pairCount = new Map<string, number>();
    for (const [rel, r] of Object.entries(schema.relations ?? {})) {
      const from = (r?.from ?? []).filter((c) => boxes.has(c));
      const to = (r?.to ?? []).filter((c) => boxes.has(c));
      const same = from.length === to.length && from.every((c) => to.includes(c));
      const pairs = same ? from.map((c) => [c, c] as const) : from.flatMap((f) => to.map((t) => [f, t] as const));
      const m = metrics?.relations?.[rel];
      pairs.forEach(([f, t], k) => {
        const key = [f, t].sort().join("|");
        const i = pairCount.get(key) ?? 0;
        pairCount.set(key, i + 1);
        edges.push({ rel, from: f, to: t, i, proposed: r?.status === "proposed", bad: !!m?.violations, label: k === 0 });
      });
    }
    return { boxes, groups, edges, width: PAD * 2 + col * COL_W - (COL_W - BOX_W), height: TOP + maxRows * ROW_H + 24 };
  }, [schema, metrics]);

  // Spread each box side's edge ends along its height (ordered by where the other end is),
  // so edges into a busy class (Topic, Agent) don't all meet at one point.
  const ports = useMemo(() => {
    type End = { k: number; end: "a" | "b"; box: string; side: "l" | "r"; otherY: number };
    const ends: End[] = [];
    edges.forEach((e, k) => {
      const a = boxes.get(e.from), b = boxes.get(e.to);
      if (!a || !b) return;
      if (a.name === b.name) return;                              // self-loops keep their own shape
      const sameCol = a.col === b.col, right = b.x > a.x;
      ends.push({ k, end: "a", box: a.name, side: sameCol || right ? "r" : "l", otherY: b.y });
      ends.push({ k, end: "b", box: b.name, side: sameCol || !right ? "r" : "l", otherY: a.y });
    });
    const out = new Map<string, number>();
    const bySide = new Map<string, End[]>();
    for (const e of ends) { const key = `${e.box}|${e.side}`; bySide.set(key, [...(bySide.get(key) ?? []), e]); }
    for (const list of bySide.values()) {
      list.sort((x, y) => x.otherY - y.otherY || x.k - y.k);
      list.forEach((e, i) => out.set(`${e.k}${e.end}`, ((i + 1) / (list.length + 1)) * BOX_H));
    }
    return out;
  }, [edges, boxes]);

  const count = (c: string) => metrics?.classes?.[c]?.count ?? metrics?.unknown_classes?.[c];
  const relCount = (r: string) => metrics?.relations?.[r]?.count;
  const active = hover ?? (selected?.kind === "relation" ? selected.name : null);
  const activeClass = selected?.kind === "class" ? selected.name : null;

  function path(e: Edge, k: number) {
    const a = boxes.get(e.from)!, b = boxes.get(e.to)!;
    const pa = ports.get(`${k}a`) ?? BOX_H / 2, pb = ports.get(`${k}b`) ?? BOX_H / 2;
    if (a.name === b.name) {         // self-loop on the right side
      const x = a.x + BOX_W, y = a.y + BOX_H / 2;
      return { d: `M${x},${y - 8} C${x + 34},${y - 26} ${x + 34},${y + 26} ${x},${y + 8}`, lx: x + 30, ly: y };
    }
    if (a.col === b.col) {           // same column: bow out to the right
      const x = a.x + BOX_W, y1 = a.y + pa, y2 = b.y + pb, bow = 30 + Math.abs(y2 - y1) * 0.12;
      return { d: `M${x},${y1} C${x + bow},${y1} ${x + bow},${y2} ${x},${y2}`, lx: x + bow * 0.75, ly: (y1 + y2) / 2 };
    }
    const right = b.x > a.x;
    const x1 = right ? a.x + BOX_W : a.x, x2 = right ? b.x : b.x + BOX_W;
    const y1 = a.y + pa, y2 = b.y + pb;
    const dx = (x2 - x1) * 0.5;
    return { d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`,
             lx: (x1 + x2) / 2, ly: (y1 + y2) / 2 };
  }

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ minWidth: Math.min(width, 900), maxWidth: width }}
        role="img" aria-label="Schema map: classes and the relations between them" className="select-none font-mono">
        <defs>
          <marker id="sm-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--muted)" />
          </marker>
          <marker id="sm-arrow-on" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--local)" />
          </marker>
        </defs>

        {groups.map((g) => (
          <g key={g.name}>
            <rect x={g.x} y={g.y} width={BOX_W + 16} height={g.h + 30} rx={10}
              fill="var(--surface-2)" fillOpacity={0.35} stroke="var(--line)" />
            <text x={g.x + 10} y={g.y + 16} fontSize={10} letterSpacing="0.08em" fill="var(--muted)"
              className="cursor-pointer" onClick={() => onSelect({ kind: "class", name: g.name })}>
              {g.name.toUpperCase()}{g.abstract ? "" : ""}
            </text>
          </g>
        ))}

        {edges.map((e, k) => {
          const { d, lx, ly } = path(e, k);
          const on = active === e.rel || activeClass === e.from || activeClass === e.to;
          const dim = (active || activeClass) && !on;
          const stroke = on ? "var(--local)" : e.bad ? "var(--bad)" : "var(--muted)";
          return (
            <g key={`${e.rel}-${k}`} opacity={dim ? 0.15 : 1}
              onMouseEnter={() => setHover(e.rel)} onMouseLeave={() => setHover(null)}
              onClick={() => onSelect({ kind: "relation", name: e.rel })} className="cursor-pointer">
              <path d={d} fill="none" stroke="transparent" strokeWidth={10} />
              <path d={d} fill="none" stroke={stroke} strokeOpacity={on ? 1 : 0.55} strokeWidth={on ? 1.8 : 1.1}
                strokeDasharray={e.proposed ? "4 4" : undefined} markerEnd={`url(#${on ? "sm-arrow-on" : "sm-arrow"})`} />
              {(e.label || on) && (
                <text x={lx} y={ly - 3} fontSize={9.5} textAnchor="middle" fill={on ? "var(--ink)" : "var(--muted)"}
                  paintOrder="stroke" stroke="var(--surface)" strokeWidth={3}>
                  {e.rel}{relCount(e.rel) ? ` ${relCount(e.rel)}` : ""}
                </text>
              )}
            </g>
          );
        })}

        {[...boxes.values()].filter((b) => !(b.abstract && b.y < TOP)).map((b) => {
          const sel = activeClass === b.name;
          const n = count(b.name);
          return (
            <g key={b.name} transform={`translate(${b.x},${b.y})`} className="cursor-pointer"
              onClick={() => onSelect(sel ? null : { kind: "class", name: b.name })}>
              <rect width={BOX_W} height={BOX_H} rx={7}
                fill={sel ? "var(--surface-2)" : "var(--surface)"}
                stroke={b.unknown ? "var(--warn)" : sel ? "var(--local)" : "var(--line)"} strokeWidth={sel ? 1.6 : 1}
                strokeDasharray={b.proposed || b.unknown ? "4 3" : undefined} />
              <text x={10} y={17} fontSize={12} fill={b.proposed ? "var(--muted)" : "var(--ink)"} fontFamily="IBM Plex Sans, sans-serif" fontWeight={600}>
                {b.name}
              </text>
              <text x={10} y={32} fontSize={10} fill={b.unknown ? "var(--warn)" : "var(--muted)"}>
                {b.unknown ? `not in schema · ${n}` : b.proposed ? "proposed" : n != null ? `${n} in graph` : "—"}
              </text>
            </g>
          );
        })}
      </svg>
      <div className={clsx("mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10.5px] text-muted")}>
        <span>── implemented</span><span>- - proposed</span>
        <span className="text-bad">── edges that break from/to</span>
        <span className="text-warn">- - node type the schema doesn't define</span>
        <span>numbers: instances in the selected dataset's metadata graph</span>
      </div>
    </div>
  );
}
