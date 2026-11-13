import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import clsx from "clsx";
import {
  Bot, CheckCircle2, CircleDashed, Eye, FileSearch, Filter, GraduationCap, Loader2, MessageSquareQuote,
  Pause, Play, Send, ShieldCheck, SkipForward, Sparkles, Trash2, UserRoundPen, XCircle,
} from "lucide-react";
import {
  getGraph, getPingSession, listPingChats, listPingSessions, newPingSession, pingNext, pingSay, resetPing,
  PING_STAGES, type GraphEdge, type GraphNode, type PingChat, type PingCycle, type PingHypothesis,
  type PingSession, type PingSessionSummary, type PingStageName,
} from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import type { Signals } from "@/lib/signals";
import GraphCanvas from "@/components/GraphCanvas";

const POLL_MS = 700;

const STAGE: Record<PingStageName, { n: number; label: string; icon: any; what: string }> = {
  observe: { n: 1, label: "Observe", icon: Eye, what: "the new message; did it take a follow-up offered last turn?" },
  update: { n: 2, label: "Update profile", icon: UserRoundPen, what: "goal, who/what/why, and each hypothesis supported or contradicted (1.a / 1.b)" },
  check: { n: 3, label: "Check on L1", icon: ShieldCheck, what: "every live hypothesis against the metadata graph, no LLM" },
  probe: { n: 4, label: "Probe", icon: MessageSquareQuote, what: "the 3 follow-ups shown after the reply, some testing a hypothesis" },
  gate: { n: 5, label: "Gate", icon: Filter, what: "enough to suggest? confident, held up, confirmed, about work, not a repeat" },
  dive: { n: 6, label: "Dive L2", icon: FileSearch, what: "scoped retrieval for the top hypothesis + the evidence check" },
  suggest: { n: 7, label: "Suggest", icon: Sparkles, what: "one record, framed as artifact / insight / idea, or stay quiet" },
  learn: { n: 8, label: "Learn", icon: GraduationCap, what: "confidence after this turn and what moved" },
};
const KIND_STYLE: Record<string, string> = { artifact: "text-local border-local/50", insight: "text-global border-global/50", idea: "text-vector border-vector/50" };
const VERDICT_STYLE: Record<string, string> = { validated: "text-good", weak: "text-warn", rejected: "text-bad" };
const pairKey = (a: string, b: string) => (a < b ? `${a}|||${b}` : `${b}|||${a}`);

/**
 * The Ping cycle, observed on real data: replay a real chat from the export one user
 * turn at a time (its real replies, its real next message as ground truth), and watch
 * each stage of the cycle per turn, the profile's hypotheses gain or lose confidence,
 * and the metadata graph narrow onto the clusters they point at.
 */
export default function Agent() {
  const { datasets, datasetId } = useExplorer();
  // the cycle runs on a real export (one with a metadata graph), never the synthetic corpus
  const exports = useMemo(() => datasets.filter((d) => d.id.endsWith("~metadata")).map((d) => d.id.split("~")[0]), [datasets]);
  const base = exports.includes(datasetId.split("~")[0]) ? datasetId.split("~")[0] : exports[0] ?? "";

  const [graph, setGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({ nodes: [], edges: [] });
  const [sessions, setSessions] = useState<PingSessionSummary[]>([]);
  const [params, setParams] = useSearchParams();
  const [sid, setSidState] = useState<string | null>(params.get("session"));
  const setSid = (id: string | null) => {   // the open session is in the URL, so it can be linked
    setSidState(id);
    setParams(id ? { session: id } : {}, { replace: true });
  };
  const [s, setS] = useState<PingSession | null>(null);
  const [turn, setTurn] = useState<number | null>(null);      // selected turn (null = latest)
  const [err, setErr] = useState<string | null>(null);
  const [pollErr, setPollErr] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef(false);

  useEffect(() => { if (base) getGraph(`${base}~metadata`).then(setGraph).catch((e) => setErr(String(e.message ?? e))); }, [base]);

  // session list
  const refreshList = useCallback(() => listPingSessions().then(setSessions).catch(() => undefined), []);
  useEffect(() => { refreshList(); const t = window.setInterval(refreshList, 2500); return () => window.clearInterval(t); }, [refreshList]);

  // the open session, polled while a turn runs
  useEffect(() => {
    if (!sid) { setS(null); return; }
    let stop = false, t: number | undefined;
    const tick = async () => {
      try {
        const d = await getPingSession(sid);
        if (stop) return;
        setS(d);
        setPollErr(null);
        if (d.status === "running") t = window.setTimeout(tick, POLL_MS);
      } catch (e: any) {   // a dropped poll: say so, retry, and clear it once one gets through
        if (stop) return;
        setPollErr(String(e?.message ?? e));
        t = window.setTimeout(tick, POLL_MS * 3);
      }
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [sid, s?.turn, s?.status === "running"]); // eslint-disable-line react-hooks/exhaustive-deps

  const waitIdle = async (id: string) => {
    for (;;) {
      const d = await getPingSession(id);
      setS(d);
      if (d.status !== "running") return d;
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  };
  const next = async () => {
    if (!sid) return;
    setErr(null); setTurn(null);
    try { await pingNext(sid); setS(await getPingSession(sid)); } catch (e: any) { setErr(String(e?.message ?? e)); }
  };
  const play = async () => {
    if (!sid) return;
    playRef.current = true; setPlaying(true); setTurn(null);
    try {
      let d = await waitIdle(sid);
      while (playRef.current && d.chat && (d.turns_total ?? 0) > d.turn && d.status !== "error") {
        await pingNext(sid);
        d = await waitIdle(sid);
      }
    } catch (e: any) { setErr(String(e?.message ?? e)); }
    playRef.current = false; setPlaying(false);
  };
  const stop = () => { playRef.current = false; setPlaying(false); };
  const reset = async () => {
    stop();
    await resetPing().catch((e) => setErr(String(e?.message ?? e)));
    setSid(null); setS(null); setTurn(null); refreshList();
  };

  const cycle: PingCycle | null = s?.cycles.length ? (s.cycles.find((c) => c.turn === turn) ?? s.cycles[s.cycles.length - 1]) : null;

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Bot size={18} className="text-good" />
        <h2 className="font-title text-lg font-semibold">Ping cycle</h2>
        <span className="font-mono text-[11px] text-muted">real data: {base || "no export with a metadata graph"} · replays real chats, real replies, real next messages</span>
        <button className="btn ml-auto h-8" onClick={reset} title="Clear every session and agent run: a clean dashboard">
          <Trash2 size={12} /> Reset dashboard
        </button>
      </div>
      <StageLegend />
      {err && <div className="card border-bad/40 py-2 font-mono text-[11px] text-bad">{err}</div>}
      {pollErr && <div className="font-mono text-[11px] text-warn">lost contact with the backend ({pollErr}); retrying…</div>}

      <div className="grid gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
        <div className="space-y-4">
          <StartCard base={base} onStarted={(id) => { setSid(id); setTurn(null); refreshList(); }} />
          {s && (
            <ConversationCard s={s} cycle={cycle} onPick={setTurn} onNext={next} onPlay={play} onStop={stop} playing={playing}
              onSay={async (m) => { setErr(null); setTurn(null); try { await pingSay(s.id, m); setS(await getPingSession(s.id)); } catch (e: any) { setErr(String(e?.message ?? e)); } }} />
          )}
          <SessionsCard sessions={sessions} current={sid} onPick={(id) => { setSid(id); setTurn(null); }} />
        </div>

        <div className="min-w-0 space-y-4">
          {!s ? (
            <div className="card text-sm text-muted">
              Pick a real chat from the export and press <b>Start</b>, then step through it turn by turn (or play it).
              Each turn runs the full cycle; everything below shows what happened at each stage.
            </div>
          ) : (
            <>
              <StageRail cycle={cycle} running={s.status === "running"} />
              <GraphPanel graph={graph} s={s} cycle={cycle} />
              <ProfileCard s={s} cycle={cycle} />
              {cycle && <StageCards cycle={cycle} s={s} base={base} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ───────────────────────── legend + rail ─────────────────────────

function StageLegend() {
  return (
    <ol className="grid grid-cols-2 gap-1.5 sm:grid-cols-4 xl:grid-cols-8">
      {PING_STAGES.map((k) => {
        const st = STAGE[k];
        const Icon = st.icon;
        return (
          <li key={k} className="rounded-md border border-line bg-surface px-2 py-1.5" title={st.what}>
            <div className="flex items-center gap-1.5 text-xs text-ink"><span className="font-mono text-[10px] text-muted">{st.n}</span><Icon size={12} />{st.label}</div>
            <div className="mt-0.5 line-clamp-2 text-[10.5px] leading-tight text-muted">{st.what}</div>
          </li>
        );
      })}
    </ol>
  );
}

function StageRail({ cycle, running }: { cycle: PingCycle | null; running: boolean }) {
  if (!cycle) return null;
  const firstPending = PING_STAGES.find((k) => cycle.stages[k]?.status === "pending");
  return (
    <div className="card flex flex-wrap items-stretch gap-1.5 p-2.5">
      <span className="self-center px-1 font-mono text-[11px] text-muted">turn {cycle.turn}</span>
      {PING_STAGES.map((k, i) => {
        const st = cycle.stages[k] ?? { status: "pending" };
        const isRunning = running && k === firstPending;
        const Icon = STAGE[k].icon;
        const tone = st.status === "done" ? (k === "gate" ? (st.data?.open ? "border-good/60 text-good" : "border-warn/60 text-warn")
          : k === "suggest" ? "border-good/60 text-good" : "border-line text-ink")
          : st.status === "skip" ? "border-line text-muted/60" : isRunning ? "border-local/60 text-local" : "border-dashed border-line text-muted/50";
        return (
          <a key={k} href={`#stage-${k}`} className={clsx("flex min-w-[104px] flex-1 items-center gap-1.5 whitespace-nowrap rounded-md border px-2 py-1.5 text-xs", tone)}
            title={`${st.title ?? STAGE[k].what}${st.ms ? ` (${(st.ms / 1000).toFixed(1)} s)` : ""}`}>
            {isRunning ? <Loader2 size={12} className="animate-spin" /> : st.status === "skip" ? <CircleDashed size={12} /> : <Icon size={12} />}
            <span><span className="font-mono text-[10px] text-muted">{STAGE[k].n}</span> {STAGE[k].label}</span>
            {i < PING_STAGES.length - 1 && <span className="sr-only">then</span>}
          </a>
        );
      })}
    </div>
  );
}

// ───────────────────────── left column ─────────────────────────

function StartCard({ base, onStarted }: { base: string; onStarted: (id: string) => void }) {
  const [chats, setChats] = useState<PingChat[]>([]);
  const [chatId, setChatId] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (base) listPingChats(base).then(setChats).catch((e) => setErr(String(e.message ?? e))); }, [base]);
  const start = async (withMsg?: string) => {
    setBusy(true); setErr(null);
    try {
      const s = await newPingSession(base, withMsg ? undefined : chatId);
      if (withMsg) await pingSay(s.id, withMsg);
      onStarted(s.id);
      setMsg("");
    } catch (e: any) { setErr(String(e?.message ?? e)); }
    setBusy(false);
  };
  return (
    <section className="card space-y-2.5">
      <h3 className="caption">Replay a real chat</h3>
      <select value={chatId} onChange={(e) => setChatId(e.target.value)} className="w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm">
        <option value="">{chats.length ? `pick one of ${chats.length} real chats…` : "loading chats…"}</option>
        {chats.map((c) => <option key={c.chat_id} value={c.chat_id}>{c.title} · {c.user_turns} turns{c.author ? ` · ${c.author}` : ""}</option>)}
      </select>
      <button className="btn-primary" disabled={!chatId || busy} onClick={() => start()}>
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Start
      </button>
      <div className="border-t border-line pt-2.5">
        <h3 className="caption mb-1.5">Or type a message</h3>
        <div className="flex gap-2">
          <input value={msg} onChange={(e) => setMsg(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && msg.trim()) start(msg.trim()); }}
            placeholder="starts a new session" className="min-w-0 flex-1 rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none focus:border-muted/60" />
          <button className="btn" disabled={!msg.trim() || busy} onClick={() => start(msg.trim())}><Send size={12} /></button>
        </div>
      </div>
      {err && <div className="font-mono text-[11px] text-bad">{err}</div>}
    </section>
  );
}

function ConversationCard({ s, cycle, onPick, onNext, onPlay, onStop, playing, onSay }: {
  s: PingSession; cycle: PingCycle | null; onPick: (t: number | null) => void; onNext: () => void;
  onPlay: () => void; onStop: () => void; playing: boolean; onSay: (m: string) => void;
}) {
  const [msg, setMsg] = useState("");
  const running = s.status === "running";
  const atEnd = !!s.chat && (s.turns_total ?? 0) <= s.turn;
  const probesBy = useMemo(() => {
    const m = new Map<number, PingSession["profile"]["probes"]>();
    for (const p of s.profile.probes) m.set(p.turn, [...(m.get(p.turn) ?? []), p]);
    return m;
  }, [s.profile.probes]);
  return (
    <section className="card space-y-2.5">
      <div className="flex items-baseline gap-2">
        <h3 className="caption">Conversation</h3>
        <span className="truncate text-xs text-ink">{s.title}</span>
        {s.author && <span className="font-mono text-[10px] text-muted">{s.author}</span>}
      </div>
      {s.chat ? (
        <div className="flex flex-wrap items-center gap-2">
          <button className="btn" disabled={running || atEnd || playing} onClick={onNext}>
            {running ? <Loader2 size={12} className="animate-spin" /> : <SkipForward size={12} />}
            {atEnd ? "End of chat" : `Turn ${s.turn + 1} of ${s.turns_total}`}
          </button>
          {playing ? <button className="btn" onClick={onStop}><Pause size={12} /> Stop</button>
            : <button className="btn" disabled={running || atEnd} onClick={onPlay}><Play size={12} /> Play to the end</button>}
        </div>
      ) : (
        <div className="flex gap-2">
          <input value={msg} onChange={(e) => setMsg(e.target.value)} disabled={running}
            onKeyDown={(e) => { if (e.key === "Enter" && msg.trim()) { onSay(msg.trim()); setMsg(""); } }}
            placeholder="next message" className="min-w-0 flex-1 rounded-md border border-line bg-bg px-2 py-1.5 text-sm outline-none" />
          <button className="btn" disabled={running || !msg.trim()} onClick={() => { onSay(msg.trim()); setMsg(""); }}><Send size={12} /></button>
        </div>
      )}
      <ol className="max-h-[640px] space-y-2 overflow-auto pr-1">
        {s.cycles.map((c) => {
          const sel = cycle?.turn === c.turn;
          const offered = probesBy.get(c.turn) ?? [];
          return (
            <li key={c.turn} className={clsx("space-y-1.5 rounded-md p-1.5", sel && "bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]")}>
              <button onClick={() => onPick(sel ? null : c.turn)} className="block w-full rounded bg-local/15 px-2 py-1.5 text-left text-xs">
                <span className="mb-0.5 block font-mono text-[10px] text-muted">turn {c.turn} · user</span>
                <span className="line-clamp-4 whitespace-pre-wrap">{c.user}</span>
              </button>
              {c.suggestion && (
                <div className="rounded border border-good/40 bg-good/10 px-2 py-1.5 text-xs">
                  <span className="font-mono text-[10px] text-good">Ping · {c.suggestion.kind}</span>
                  <div className="text-ink">{c.suggestion.title}</div>
                  <div className="text-muted">{c.suggestion.why}</div>
                </div>
              )}
              {c.reply && (
                <div className="rounded bg-surface-2 px-2 py-1.5 text-xs text-muted">
                  <span className="mb-0.5 block font-mono text-[10px]">assistant (real reply)</span>
                  <span className="line-clamp-3 whitespace-pre-wrap">{c.reply}</span>
                </div>
              )}
              {offered.length > 0 && (
                <ul className="space-y-1 pl-2">
                  {offered.map((p) => (
                    <li key={p.id} className="flex gap-1.5 text-[11px]">
                      <span className={clsx("shrink-0 font-mono text-[10px]", p.used ? "text-good" : p.used === false ? "text-muted/60" : "text-muted")}
                        title={p.used == null ? "waiting for the next message" : p.used ? "the next real message took this follow-up" : `not taken${p.sim != null ? ` (similarity ${p.sim.toFixed(2)})` : ""}`}>
                        {p.used ? "✓" : p.used === false ? "–" : "…"} {p.id}{p.tests ? `→${p.tests}` : ""}
                      </span>
                      <span className={clsx(p.used ? "text-ink" : "text-muted")}>{p.text}</span>
                    </li>
                  ))}
                </ul>
              )}
              {c.error && <div className="font-mono text-[10px] text-bad">{c.error}</div>}
            </li>
          );
        })}
        {running && <li className="flex items-center gap-2 px-1.5 text-xs text-muted"><Loader2 size={12} className="animate-spin" /> running the cycle…</li>}
      </ol>
    </section>
  );
}

function SessionsCard({ sessions, current, onPick }: { sessions: PingSessionSummary[]; current: string | null; onPick: (id: string) => void }) {
  return (
    <section className="card space-y-1.5 p-3">
      <h3 className="caption px-1">Sessions · {sessions.length}</h3>
      {!sessions.length && <div className="px-1 text-xs text-muted">None. (Kept in memory; Reset or a backend restart clears them.)</div>}
      <ul className="space-y-1">
        {sessions.map((x) => (
          <li key={x.id}>
            <button onClick={() => onPick(x.id)} className={clsx("w-full rounded-md px-2 py-1.5 text-left", current === x.id ? "bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]" : "hover:bg-surface-2/60")}>
              <div className="flex items-center gap-1.5 text-xs">
                {x.status === "running" ? <Loader2 size={12} className="animate-spin text-local" /> : x.status === "error" ? <XCircle size={12} className="text-bad" /> : <CheckCircle2 size={12} className="text-good" />}
                <span className="truncate text-ink">{x.title}</span>
              </div>
              <div className="mt-0.5 flex flex-wrap gap-x-2 pl-5 font-mono text-[10px] text-muted">
                <span>turn {x.turn}{x.turns_total ? `/${x.turns_total}` : ""}</span>
                <span title="follow-ups the next real message took">follow-ups taken {x.probes_used}/{x.probes_offered}</span>
                <span title="turns where the top hypothesis was supported by the next real message">predicted {x.top_supported}/{x.top_scored}</span>
                <span>suggested {x.suggestions}</span>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ───────────────────────── the orb graph, honing in ─────────────────────────

function GraphPanel({ graph, s, cycle }: { graph: { nodes: GraphNode[]; edges: GraphEdge[] }; s: PingSession; cycle: PingCycle | null }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [key, setKey] = useState(0);
  // A turn only has a focus once its L1 check has run; until then keep showing the last
  // turn that has one, so the graph moves from turn to turn instead of blanking out.
  const shown = cycle?.focus?.hyps ? cycle
    : [...s.cycles].reverse().find((c) => c.focus?.hyps && (!cycle || c.turn < cycle.turn)) ?? null;
  const f = shown?.focus ?? {};
  const top = f.top ? f.hyps?.[f.top] : undefined;
  // redraw whenever the focus changes: scopes at the L1 check, evidence at the dive, the suggestion
  const focusKey = shown ? `${shown.turn}|${JSON.stringify(shown.focus)}` : "";
  const stageNow = cycle ? [...PING_STAGES].reverse().find((k) => cycle.stages[k]?.status === "done" || cycle.stages[k]?.status === "skip") : undefined;
  const signals = useMemo<Signals | null>(() => {
    if (!shown || !f.hyps) return null;
    const ids = new Set(graph.nodes.map((n) => n.id));
    const keep = (xs: string[] | undefined) => (xs ?? []).filter((x) => ids.has(x));
    const entity = new Set(top ? [...keep(top.seeds), ...keep(top.hits), ...keep(top.scope)] : []);
    const context = new Set([...keep(f.evidence), ...keep(f.suggestion)]);
    const vector = new Set<string>();
    for (const [hid, h] of Object.entries(f.hyps)) {
      if (hid === f.top || h.status !== "active") continue;
      for (const x of [...keep(h.seeds), ...keep(h.hits), ...keep(h.scope)]) if (!entity.has(x)) vector.add(x);
    }
    const strong = new Set([...entity, ...context]);
    const relation = new Set(graph.edges.filter((e) => strong.has(e.source) && strong.has(e.target)).map((e) => pairKey(e.source, e.target)));
    return { entity, relation, vector, context };
  }, [focusKey, graph]); // eslint-disable-line react-hooks/exhaustive-deps
  // each new focus re-opens the clusters it touches and zooms to them: the honing-in
  useEffect(() => { setKey((k) => k + 1); }, [focusKey]);
  const hyp = (id?: string | null) => s.profile.hypotheses.find((h) => h.id === id);
  const t = hyp(f.top);
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2 px-1 font-mono text-[11px] text-muted">
        {shown ? (
          <span>
            graph: <span className="text-ink">turn {shown.turn}</span>
            {shown === cycle
              ? <> · {stageNow ? `after ${STAGE[stageNow].label.toLowerCase()}` : "starting"}{f.suggestion?.length ? " · suggestion in white" : f.evidence?.length ? " · evidence in white" : ""}</>
              : <> · turn {cycle?.turn} is at {stageNow ? STAGE[stageNow].label.toLowerCase() : "observe"}; its hypotheses light up once they're checked on L1</>}
          </span>
        ) : <span>graph: no hypotheses checked yet; they light up after the first L1 check</span>}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-xs">
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--local)" }} />
          <span className="text-ink">{t ? `${t.id} (${top?.conf.toFixed(2)})` : "no top hypothesis"}</span>
          {t && <span className="max-w-[420px] truncate text-muted">{t.claim}</span>}
        </span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: "var(--vector)" }} />
          <span className="text-muted">competing hypotheses ({Object.entries(f.hyps ?? {}).filter(([id, h]) => id !== f.top && h.status === "active").length})</span></span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border border-line" style={{ background: "#f4f7fa" }} />
          <span className="text-muted">read / suggested</span></span>
      </div>
      <div className="h-[520px]">
        <GraphCanvas graph={graph} signals={signals} resultKey={key} highlight={null} onClearHighlight={() => undefined}
          selectedId={selected} onSelect={setSelected} />
      </div>
    </div>
  );
}

// ───────────────────────── profile ─────────────────────────

function Spark({ s, h }: { s: PingSession; h: PingHypothesis }) {
  const pts = s.cycles.map((c) => c.snapshot?.find((x) => x.id === h.id)?.conf ?? null);
  const W = 90, H = 22, n = Math.max(1, pts.length - 1);
  const path = pts.map((v, i) => (v == null ? null : `${(i / n) * W},${H - v * H}`)).filter(Boolean).join(" ");
  return (
    <svg width={W} height={H} className="shrink-0">
      <line x1={0} x2={W} y1={H - 0.65 * H} y2={H - 0.65 * H} stroke="var(--good)" strokeDasharray="2 2" opacity={0.4} />
      <polyline points={path} fill="none" stroke={h.status === "active" ? "var(--local)" : "var(--muted)"} strokeWidth={1.5} />
    </svg>
  );
}

function ProfileCard({ s, cycle }: { s: PingSession; cycle: PingCycle | null }) {
  const P = s.profile;
  const [showRetired, setShowRetired] = useState(false);
  const hyps = [...P.hypotheses].filter((h) => showRetired || h.status === "active").sort((a, b) => b.confidence - a.confidence);
  return (
    <section className="card space-y-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="caption">Profile</h3>
        <span className="font-mono text-[11px] text-muted">after turn {s.turn}{cycle && cycle.turn !== s.turn ? ` (viewing turn ${cycle.turn} above)` : ""}</span>
        <label className="ml-auto flex items-center gap-1.5 font-mono text-[10px] text-muted">
          <input type="checkbox" checked={showRetired} onChange={(e) => setShowRetired(e.target.checked)} /> show retired
        </label>
      </div>
      <div className="grid gap-2 text-sm sm:grid-cols-[1fr_auto]">
        <div><span className="caption mr-2">goal</span>{P.goal || "—"}</div>
        <div><span className="caption mr-2">stage</span>{P.stage || "—"}</div>
      </div>
      <div className="grid gap-1 text-xs sm:grid-cols-3">
        {(["who", "what", "why"] as const).map((k) => (
          <div key={k} className="rounded border border-line px-2 py-1"><span className="caption mr-1.5">{k}</span>{P.facts[k] || <span className="text-muted">not said yet</span>}</div>
        ))}
      </div>
      <table className="w-full text-xs">
        <thead><tr className="text-left font-mono text-[10px] uppercase tracking-wide text-muted">
          <th className="py-1 pr-2">id</th><th className="pr-2">kind</th><th className="pr-2">hypothesis</th><th className="pr-2">confidence</th><th className="pr-2">over turns</th><th>L1</th>
        </tr></thead>
        <tbody>
          {hyps.map((h) => (
            <tr key={h.id} className={clsx("border-t border-line align-top", h.status === "retired" && "opacity-45")}>
              <td className="py-1.5 pr-2 font-mono">{h.id}{h.confirmed && <span className="ml-1 text-good" title="confirmed by the user">✓</span>}</td>
              <td className="pr-2"><span className={clsx("rounded border px-1 font-mono text-[10px]", KIND_STYLE[h.kind])}>{h.kind}</span></td>
              <td className="pr-2">
                <div className="text-ink">{h.claim}</div>
                <div className="font-mono text-[10px] text-muted">{h.answer_type} → {h.target} · “{h.query}”{h.status === "retired" ? " · retired" : ""}</div>
              </td>
              <td className="pr-2">
                <div className="flex items-center gap-1.5">
                  <div className="h-1.5 w-16 rounded bg-line"><div className={clsx("h-1.5 rounded", h.confidence >= 0.65 ? "bg-good" : "bg-local")} style={{ width: `${h.confidence * 100}%` }} /></div>
                  <span className="font-mono tabular">{h.confidence.toFixed(2)}</span>
                </div>
              </td>
              <td className="pr-2"><Spark s={s} h={h} /></td>
              <td className={clsx("font-mono text-[10px]", VERDICT_STYLE[h.l1?.verdict ?? ""])} title={h.l1?.why}>{h.l1?.verdict ?? "—"}</td>
            </tr>
          ))}
          {!hyps.length && <tr><td colSpan={6} className="py-2 text-muted">No hypotheses yet.</td></tr>}
        </tbody>
      </table>
      <div className="font-mono text-[10px] text-muted">dashed line on the sparkline = the gate's confidence bar (0.65)</div>
    </section>
  );
}

// ───────────────────────── one card per stage ─────────────────────────

function StageCards({ cycle, s, base }: { cycle: PingCycle; s: PingSession; base: string }) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      {PING_STAGES.map((k) => <StageCard key={k} k={k} cycle={cycle} s={s} base={base} />)}
    </div>
  );
}

function StageCard({ k, cycle, s, base }: { k: PingStageName; cycle: PingCycle; s: PingSession; base: string }) {
  const st = cycle.stages[k] ?? { status: "pending" };
  const d = st.data ?? {};
  const Icon = STAGE[k].icon;
  const doc = (x: string, title?: string) => (
    <Link to={`/doc?doc=${encodeURIComponent(x)}&dataset=${encodeURIComponent(base)}`} target="_blank" className="text-ink hover:underline">{title || x}</Link>
  );
  return (
    <section id={`stage-${k}`} className={clsx("card space-y-1.5 p-3", st.status === "skip" && "opacity-60", st.status === "pending" && "border-dashed opacity-50")}>
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-[10px] text-muted">{STAGE[k].n}</span><Icon size={13} className="text-muted" />
        <span className="caption">{STAGE[k].label}</span>
        {st.ms ? <span className="ml-auto font-mono text-[10px] text-muted">{(st.ms / 1000).toFixed(1)} s</span> : null}
      </div>
      <div className="text-sm text-ink">{st.title ?? (st.status === "pending" ? "waiting…" : "")}</div>
      {st.detail && <div className="text-xs text-muted">{st.detail}</div>}

      {k === "observe" && d.sims && Object.keys(d.sims).length > 0 && (
        <ul className="font-mono text-[10px] text-muted">
          {Object.entries(d.sims as Record<string, number>).map(([pid, v]) => <li key={pid}>{pid}: similarity {v.toFixed(2)}{v >= 0.85 ? " (counts as taken)" : ""}</li>)}
        </ul>
      )}
      {k === "update" && (
        <ul className="space-y-0.5 text-xs">
          {d.probe_used && <li className="text-good">follow-up {d.probe_used} was taken (1.a)</li>}
          {(d.changes ?? []).map((c: any, i: number) => (
            <li key={i}><span className="font-mono">{c.id}</span>{" "}
              <span className={c.effect.startsWith("support") ? "text-good" : c.effect === "contradict" ? "text-bad" : "text-muted"}>{c.effect.replace("_", " ")}</span>
              {" → "}{c.conf?.toFixed(2)} <span className="text-muted">{c.why}</span></li>
          ))}
          {(d.born ?? []).length > 0 && <li className="text-muted">new: {(d.born as string[]).join(", ")}</li>}
        </ul>
      )}
      {k === "check" && (
        <ul className="space-y-0.5 text-xs">
          {(d.checks ?? []).map((c: any) => (
            <li key={c.id}><span className="font-mono">{c.id}</span> <span className={VERDICT_STYLE[c.verdict]}>{c.verdict}</span> → {c.conf.toFixed(2)}
              {c.status === "retired" && <span className="text-bad"> retired</span>} <span className="text-muted">{c.why}</span></li>
          ))}
        </ul>
      )}
      {k === "probe" && (
        <ul className="space-y-0.5 text-xs">
          {(d.probes ?? []).map((p: any) => {
            const now = s.profile.probes.find((x) => x.id === p.id);
            return (
              <li key={p.id}><span className="font-mono text-[10px] text-muted">{p.id}{p.tests ? ` tests ${p.tests}` : ""}</span>{" "}
                <span className={now?.used ? "text-good" : undefined}>{p.text}</span>
                {now?.used != null && <span className="font-mono text-[10px] text-muted"> · {now.used ? "taken next turn" : "not taken"}</span>}</li>
            );
          })}
          {cycle.eval.next_user && <li className="pt-1 text-muted">next real message: “{cycle.eval.next_user}”</li>}
          {cycle.eval.top_effect && (
            <li className={cycle.eval.top_supported ? "text-good" : "text-muted"}>
              the top hypothesis was {cycle.eval.top_effect.replace("_", " ")} by it
            </li>
          )}
        </ul>
      )}
      {k === "gate" && (
        <ul className="space-y-0.5 text-xs">
          {(d.criteria ?? []).map((c: any) => (
            <li key={c.name} className={c.ok ? "text-good" : "text-warn"}>{c.ok ? "✓" : "✗"} {c.name} <span className="text-muted">({String(c.value)})</span></li>
          ))}
        </ul>
      )}
      {k === "dive" && st.status === "done" && (
        <div className="space-y-1 text-xs">
          {(d.steps ?? []).filter((e: any) => e.kind === "evidence").flatMap((e: any) => e.data.evidence ?? []).map((x: any, i: number) => (
            <div key={i} className="rounded border border-line bg-bg px-2 py-1">
              <div>[{i + 1}] {doc(x.doc, x.title)}</div>
              <div className="line-clamp-3 text-muted">{x.excerpt}</div>
            </div>
          ))}
          {(d.steps ?? []).filter((e: any) => e.kind === "search" || e.kind === "fallback").map((e: any, i: number) => (
            <div key={i} className="font-mono text-[10px] text-muted">{e.title}</div>
          ))}
        </div>
      )}
      {k === "suggest" && d.suggestion && (
        <div className="rounded border border-good/40 bg-good/10 px-2 py-1.5 text-xs">
          <span className={clsx("mr-1.5 rounded border px-1 font-mono text-[10px]", KIND_STYLE[d.suggestion.kind])}>{d.suggestion.kind}</span>
          {doc(d.suggestion.doc, d.suggestion.title)}
          <div className="text-muted">from {d.suggestion.hypothesis} · check: {d.suggestion.verdict}</div>
        </div>
      )}
      {k === "learn" && (d.moves ?? []).length > 0 && (
        <div className="font-mono text-[10px] text-muted">{(d.moves as string[]).join(" · ")}</div>
      )}
      {k === "observe" && d.last_suggestion && <div className="text-xs text-muted">last suggestion: {doc(d.last_suggestion)}</div>}
      {st.status === "pending" && <div className="text-[11px] text-muted">{STAGE[k].what}</div>}
    </section>
  );
}
