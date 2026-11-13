import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { Pause, Play, Radio, Trash2 } from "lucide-react";
import { listLive, TECHNIQUE_LABEL, type AssistResult, type LiveEvent } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import { Chime } from "@/components/ChatPanel";

const POLL_MS = 2000;
const KEEP = 100;

type Source = "all" | LiveEvent["source"];
const SOURCES: { id: Source; label: string; title: string }[] = [
  { id: "all", label: "all", title: "Every chime-in call" },
  { id: "open-webui", label: "Open WebUI", title: "Real Chime In calls from people chatting in Open WebUI" },
  { id: "explorer", label: "Explorer", title: "Calls from this app's own chat replay" },
];

/**
 * Live: every /api/assist call as it reaches the backend, newest first: what the user
 * was saying, the query the engine searched with, each stage's timing and what it
 * suggested (or why it stayed quiet). "Inspect" opens the retrieval in the Explorer.
 */
export default function Live() {
  const { datasetId, showResult } = useExplorer();
  const navigate = useNavigate();
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [paused, setPaused] = useState(false);
  const [source, setSource] = useState<Source>("all");
  const [down, setDown] = useState<string | null>(null);
  const after = useRef(0);

  useEffect(() => {
    if (paused) return;
    let stop = false;
    let t: number | undefined;
    const tick = async () => {
      try {
        const d = await listLive(after.current);
        if (d.latest < after.current) after.current = 0;   // backend restarted: start over
        if (d.events.length) {
          after.current = d.events[d.events.length - 1].seq;
          setEvents((cur) => [...d.events.slice().reverse(), ...cur].slice(0, KEEP));
        }
        setDown(null);
      } catch (e: any) {
        setDown(String(e?.message ?? e));
      }
      if (!stop) t = window.setTimeout(tick, POLL_MS);
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [paused]);

  const shown = events.filter((e) => source === "all" || e.source === source);
  const open = (_e: LiveEvent, r: AssistResult) => {
    if (!r.retrieval) return;
    showResult(r.retrieval);
    navigate("/");
  };

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Radio size={16} className={clsx(paused || down ? "text-muted" : "animate-pulse text-good")} />
          <h2 className="font-title text-lg font-semibold">Live chime-ins</h2>
        </div>
        <span className="font-mono text-[11px] text-muted">
          {down ? `backend unreachable (${down})` : paused ? "paused" : `watching · checks every ${POLL_MS / 1000} s`}
        </span>
        <div role="radiogroup" aria-label="Source" className="ml-auto flex rounded-md border border-line bg-bg p-0.5">
          {SOURCES.map(({ id, label, title }) => (
            <button key={id} role="radio" aria-checked={source === id} title={title} onClick={() => setSource(id)}
              className={source === id ? "rounded bg-surface-2 px-3 py-1 text-xs text-ink" : "rounded px-3 py-1 text-xs text-muted hover:text-ink"}>
              {label}
            </button>
          ))}
        </div>
        <button className="btn h-8" onClick={() => setPaused((p) => !p)}>
          {paused ? <><Play size={12} /> Resume</> : <><Pause size={12} /> Pause</>}
        </button>
        <button className="btn h-8" onClick={() => setEvents([])} title="Clear this list (the backend keeps its own last 100)">
          <Trash2 size={12} /> Clear
        </button>
      </div>

      {shown.length === 0 ? (
        <div className="card text-sm text-muted">
          Nothing yet. Chime-ins appear here as they happen: start chatting in Open WebUI with Chime In on,
          or replay a chat in the Explorer's chat mode.
        </div>
      ) : (
        <ul className="space-y-3">
          {shown.map((e) => <LiveRow key={e.seq} e={e} canOpen={!e.dataset || e.dataset === datasetId} onOpen={open} />)}
        </ul>
      )}
    </div>
  );
}

function LiveRow({ e, canOpen, onOpen }: { e: LiveEvent; canOpen: boolean; onOpen: (e: LiveEvent, r: AssistResult) => void }) {
  const lastUser = [...e.turns].reverse().find((t) => t.role === "user");
  const time = new Date(e.at * 1000).toLocaleTimeString();
  return (
    <li className="card space-y-2.5 p-3">
      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
        <span className="text-ink">{time}</span>
        <span className={clsx("rounded border px-1.5 py-px",
          e.source === "open-webui" ? "border-vector/50 text-vector" : "border-line")}>
          {e.source === "open-webui" ? "Open WebUI" : e.source}
        </span>
        <span>{TECHNIQUE_LABEL[e.technique] ?? e.technique}</span>
        <span>· {e.queryStrategy}</span>
        <span>· min {e.minScore}</span>
        {e.llmFilter && <span>· LLM check</span>}
        {e.dataset && <span className="ml-auto truncate" title="dataset">{e.dataset}</span>}
      </div>
      {lastUser && (
        <div className="rounded-lg bg-local/15 px-3 py-2 text-sm">
          <div className="mb-0.5 font-mono text-[10px] uppercase tracking-wide text-muted/70">
            user said{e.turns.length > 1 ? ` · ${e.turns.length} turns sent` : ""}
          </div>
          <span className="line-clamp-4 whitespace-pre-wrap">{lastUser.content}</span>
        </div>
      )}
      {e.error ? (
        <div className="font-mono text-[11px] text-red-400">chime-in failed: {e.error}</div>
      ) : e.result && (
        <>
          <Chime entry={{ kind: "chime", status: "done", res: e.result }}
            onOpen={(r) => (canOpen ? onOpen(e, r) : undefined)} />
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted">
            {e.result.stages.map((s, i) => (
              <span key={i} title={s.detail}>{s.label}: <span className="text-ink/80">{s.stat}</span> · {Math.round(s.ms)} ms</span>
            ))}
            {!canOpen && <span className="text-warn">switch the dataset to {e.dataset} to inspect</span>}
          </div>
        </>
      )}
    </li>
  );
}
