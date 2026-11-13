import { useEffect, useRef } from "react";

export const SHORTCUTS: [string, string][] = [
  ["/", "Focus the question"],
  ["1 – 4", "Switch mode (vector · bm25 · bm25+vector · agent)"],
  ["f", "Fit graph to view"],
  ["l", "Toggle all labels"],
  ["c", "Collapse graph regions"],
  ["Esc", "Clear selection and highlights"],
  ["?", "Show these shortcuts"],
];

type Handler = (e: KeyboardEvent) => void;

/**
 * Global single-key shortcuts. Ignored while typing in a field (except Esc,
 * which blurs the field) and when a modifier key is held.
 */
export function useHotkeys(map: Record<string, Handler>) {
  const ref = useRef(map);
  ref.current = map;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) {
        if (e.key === "Escape") target.blur();
        return;
      }
      const fn = ref.current[e.key];
      if (fn) { e.preventDefault(); fn(e); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
