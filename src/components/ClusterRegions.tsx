import { clusterColor } from "@/lib/signals";

export type Orb = { cluster: string; count: number; r: number; x: number; y: number };

type Props = {
  orbs: Orb[];
  expanded: Set<string>;
  detail: boolean;
  hovered: string | null;
  onHover: (c: string | null) => void;
  onExpand: (c: string) => void;
};

/**
 * "The nebula": one glowing region orb per cluster. Purely 2D SVG —
 * radial gradients + a blurred bloom, with a slow CSS drift on each orb.
 * In detail view, expanded clusters dissolve and the rest fade back.
 */
export default function ClusterRegions({ orbs, expanded, detail, hovered, onHover, onExpand }: Props) {
  return (
    <g>
      <defs>
        <filter id="orb-bloom" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="14" />
        </filter>
        {orbs.map((o) => {
          const c = clusterColor(o.cluster);
          return (
            <radialGradient key={o.cluster} id={`orb-${o.cluster}`}>
              <stop offset="0%" stopColor={c} stopOpacity={0.95} />
              <stop offset="35%" stopColor={c} stopOpacity={0.45} />
              <stop offset="100%" stopColor={c} stopOpacity={0} />
            </radialGradient>
          );
        })}
      </defs>
      {orbs.map((o, i) => {
        const isOpen = expanded.has(o.cluster);
        const isHover = hovered === o.cluster;
        const opacity = isOpen ? 0 : detail ? (isHover ? 0.6 : 0.28) : isHover ? 1 : 0.82;
        const scale = isOpen ? 1.6 : detail ? 0.6 : isHover ? 1.06 : 1;
        return (
          <g key={o.cluster} transform={`translate(${o.x},${o.y})`}>
            <g className="orb-drift" style={{ animationDelay: `${-i * 1.7}s`, animationDuration: `${8 + (i % 3) * 2}s` }}>
              <g
                style={{
                  opacity,
                  transform: `scale(${scale})`,
                  transition: "opacity 600ms ease, transform 600ms ease",
                  pointerEvents: isOpen ? "none" : "auto",
                  cursor: "pointer",
                }}
                onMouseEnter={() => onHover(o.cluster)}
                onMouseLeave={() => onHover(null)}
                onClick={(e) => { e.stopPropagation(); onExpand(o.cluster); }}
              >
                <circle r={o.r * 1.4} fill={clusterColor(o.cluster)} opacity={0.18} filter="url(#orb-bloom)" />
                <circle r={o.r} fill={`url(#orb-${o.cluster})`} />
                <circle r={o.r * 0.18} fill="#fff" opacity={0.35} />
                <text y={o.r + 16} textAnchor="middle" className="select-none fill-ink font-mono text-[11px]">
                  {o.cluster} · {o.count}
                </text>
              </g>
            </g>
          </g>
        );
      })}
    </g>
  );
}
