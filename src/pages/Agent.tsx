import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import clsx from "clsx";
import {
  Bot, CheckCircle2, CircleDashed, Compass, FileText, GitBranch, Lightbulb, Loader2, Play,
  RotateCcw, Search, ShieldCheck, SkipForward, Target, TriangleAlert, XCircle,
} from "lucide-react";
import {
  getAgentRun, listAgentRuns, listReplayChats, startAgent,
  type AgentEvent, type AgentRunDetail, type AgentRunSummary, type ChatTurn, type ReplayChat,
} from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";

const RUNS_POLL_MS = 1500;
const RUN_POLL_MS = 500;

const SOURCE_LABEL: Record<string, string> = {
  explorer: "Agent page", search: "Explorer search", bench: "benchmark", test: "test",
};
const ROUTE_HINT: Record<string, string> = {
  records: "answer from the graph's records",
  document: "find the record that covers it",
  detail: "dive into content, scoped by the graph",
};

/**
 * Agent: the routing loop, live. Every run -- started here, from an Explorer search in
 * "agent" mode, or by the benchmark -- shows each step as it happens: the goal it settled
 * on, the hypotheses about where the answer lives, how each held up against the metadata
 * graph, the route, the scoped search, the evidence it read and its own check of it.
 */
export default function Agent() {
  const { datasetId } = useExplorer();
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [down, setDown] = useState<string | null>(null);

  // the run list (any source), newest first
  useEffect(() => {
    let stop = false, t: number | undefined;
    const tick = async () => {
      try {
        const r = await listAgentRuns();
        setRuns(r);
        setDown(null);
        if (follow && r.length) setSelected((cur) => (r[0].id !== cur ? r[0].id : cur));
      } catch (e: any) {
        setDown(String(e?.message ?? e));
      }
      if (!stop) t = window.setTimeout(tick, RUNS_POLL_MS);
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [follow]);

  const started = (id: string) => { setSelected(id); setFollow(false); };

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Bot size={18} className={clsx(down ? "text-muted" : "text-good")} />
        <h2 className="font-title text-lg font-semibold">Routing agent</h2>
        <span className="font-mono text-[11px] text-muted">
          {down ? `backend unreachable (${down})` : "goal → hypotheses → validate on the graph → route → scoped dive → check"}
        </span>
        <label className="ml-auto flex items-center gap-1.5 font-mono text-[11px] text-muted">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          follow the newest run
        </label>
      </div>

      <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-4">
          <AskCard datasetId={datasetId} onStarted={started} />
          <ReplayCard datasetId={datasetId} onStarted={started} />
          <RunList runs={runs} selected={selected} onSelect={(id) => { setSelected(id); setFollow(false); }} />
        </div>
        <div className="min-w-0">
          {selected ? <RunView id={selected} /> : (
            <div className="card text-sm text-muted">
              No runs yet. Ask a question, replay a chat, or search in the Explorer with the <b>agent</b> mode:
              every run shows up here as it happens.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ───────────────────────── starting runs ─────────────────────────

function AskCard({ datasetId, onStarted }: { datasetId: string; onStarted: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [gate, setGate] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    if (!q.trim()) return;
    setBusy(true); setErr(null);
    try { onStarted(await startAgent({ datasetId, query: q.trim(), requireClear: gate })); }
    catch (e: any) { setErr(String(e?.message ?? e)); }
    setBusy(false);
  };
  return (
    <section className="card space-y-2.5">
      <h3 className="caption">Ask</h3>
      <textarea value={q} onChange={(e) => setQ(e.target.value)} rows={3}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go(); }}
        placeholder="e.g. who has been working on Caldera, and what broke?"
        className="w-full resize-y rounded-md border border-line bg-bg px-2.5 py-2 text-sm outline-none focus:border-muted/60" />
      <div className="flex items-center gap-3">
        <button className="btn-primary" disabled={busy || !q.trim()} onClick={go}>
          {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} Run
        </button>
        <label className="flex items-center gap-1.5 text-xs text-muted" title="Stop at the goal step when the question doesn't say what is needed yet">
          <input type="checkbox" checked={gate} onChange={(e) => setGate(e.target.checked)} />
          wait until the goal is clear
        </label>
      </div>
      {err && <div className="font-mono text-[11px] text-bad">{err}</div>}
    </section>
  );
}

function ReplayCard({ datasetId, onStarted }: { datasetId: string; onStarted: (id: string) => void }) {
  const [chats, setChats] = useState<ReplayChat[]>([]);
  const [chatId, setChatId] = useState("");
  const [upTo, setUpTo] = useState(0);          // user turns sent so far
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setChats([]); setChatId(""); setUpTo(0);
    listReplayChats(datasetId).then(setChats).catch((e) => setErr(String(e?.message ?? e)));
  }, [datasetId]);
  const chat = chats.find((c) => c.chatId === chatId);
  const userIdx = useMemo(() => (chat ? chat.turns.flatMap((t, i) => (t.role === "user" ? [i] : [])) : []), [chat]);
  const next = async () => {
    if (!chat || upTo >= userIdx.length) return;
    const turns: ChatTurn[] = chat.turns.slice(0, userIdx[upTo] + 1);
    setErr(null);
    try {
      onStarted(await startAgent({ datasetId, turns, requireClear: true, excludeDocs: [chat.doc] }));
      setUpTo(upTo + 1);
    } catch (e: any) { setErr(String(e?.message ?? e)); }
  };
  return (
    <section className="card space-y-2.5">
      <h3 className="caption">Replay a chat</h3>
      <p className="text-xs text-muted">Send a real chat one user turn at a time and watch the goal and hypotheses firm up. The chat's own doc is excluded.</p>
      <select value={chatId} onChange={(e) => { setChatId(e.target.value); setUpTo(0); }}
        className="w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm">
        <option value="">{chats.length ? `pick one of ${chats.length} chats…` : "no replayable chats in this dataset"}</option>
        {chats.map((c) => <option key={c.chatId} value={c.chatId}>{c.title} · {c.userTurns} turns{c.author ? ` · ${c.author}` : ""}</option>)}
      </select>
      {chat && (
        <>
          <ol className="max-h-56 space-y-1.5 overflow-auto">
            {chat.turns.slice(0, upTo ? userIdx[upTo - 1] + 1 : 0).map((t, i) => (
              <li key={i} className={clsx("rounded px-2 py-1 text-xs", t.role === "user" ? "bg-local/15" : "bg-surface-2 text-muted")}>
                <span className="line-clamp-3 whitespace-pre-wrap">{t.content}</span>
              </li>
            ))}
          </ol>
          <div className="flex items-center gap-2">
            <button className="btn" disabled={upTo >= userIdx.length} onClick={next}>
              <SkipForward size={12} /> {upTo === 0 ? "Send the first turn" : upTo >= userIdx.length ? "End of chat" : `Send turn ${upTo + 1} of ${userIdx.length}`}
            </button>
            {upTo > 0 && <button className="btn" onClick={() => setUpTo(0)}><RotateCcw size={12} /> Restart</button>}
          </div>
        </>
      )}
      {err && <div className="font-mono text-[11px] text-bad">{err}</div>}
    </section>
  );
}

// ───────────────────────── run list ─────────────────────────

function StatusIcon({ s }: { s: AgentRunSummary["status"] }) {
  if (s === "running") return <Loader2 size={13} className="shrink-0 animate-spin text-local" />;
  if (s === "done") return <CheckCircle2 size={13} className="shrink-0 text-good" />;
  if (s === "waiting") return <CircleDashed size={13} className="shrink-0 text-warn" />;
  return <XCircle size={13} className="shrink-0 text-bad" />;
}

function RunList({ runs, selected, onSelect }: { runs: AgentRunSummary[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <section className="card space-y-2 p-3">
      <h3 className="caption px-1">Recent runs · {runs.length}</h3>
      {!runs.length && <div className="px-1 text-xs text-muted">None yet (kept in memory; a backend restart clears them).</div>}
      <ul className="max-h-[420px] space-y-1 overflow-auto">
        {runs.map((r) => (
          <li key={r.id}>
            <button onClick={() => onSelect(r.id)}
              className={clsx("w-full rounded-md px-2 py-1.5 text-left transition-colors",
                selected === r.id ? "bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]" : "hover:bg-surface-2/60")}>
              <div className="flex items-center gap-1.5">
                <StatusIcon s={r.status} />
                <span className="truncate text-xs text-ink">{r.query || "(no message)"}</span>
              </div>
              <div className="mt-0.5 flex gap-2 pl-5 font-mono text-[10px] text-muted">
                <span>{new Date(r.at * 1000).toLocaleTimeString()}</span>
                <span>{SOURCE_LABEL[r.source] ?? r.source}</span>
                {r.route && <span className="text-ink/70">{r.route}</span>}
                <span className="ml-auto">{(r.ms / 1000).toFixed(1)} s</span>
              </div>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ───────────────────────── one run, live ─────────────────────────

function RunView({ id }: { id: string }) {
  const [run, setRun] = useState<AgentRunDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const after = useRef(0);

  useEffect(() => {
    let stop = false, t: number | undefined;
    after.current = 0;
    setRun(null); setErr(null);
    const tick = async () => {
      try {
        const d = await getAgentRun(id, after.current);
        if (stop) return;   // a stale loop (dev mode mounts effects twice) must not append
        if (d.events.length) after.current = d.events[d.events.length - 1].i;
        setRun((cur) => {
          if (!cur || cur.id !== d.id) return d;
          const seen = new Set(cur.events.map((e) => e.i));
          return { ...d, events: [...cur.events, ...d.events.filter((e) => !seen.has(e.i))] };
        });
        if (d.status !== "running") return;   // finished: stop polling
      } catch (e: any) {
        setErr(String(e?.message ?? e));
        return;
      }
      if (!stop) t = window.setTimeout(tick, RUN_POLL_MS);
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [id]);

  if (err) return <div className="card text-sm text-bad">{err}</div>;
  if (!run) return <div className="card text-sm text-muted"><Loader2 size={14} className="mr-1 inline animate-spin" />loading run…</div>;
  const lastUser = [...run.turns].reverse().find((t) => t.role === "user");
  return (
    <section className="card space-y-3">
      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
        <StatusIcon s={run.status} />
        <span className="text-ink">{run.status}</span>
        <span>· {SOURCE_LABEL[run.source] ?? run.source}</span>
        <span>· {run.dataset}</span>
        <span className="ml-auto tabular">{(run.ms / 1000).toFixed(1)} s · {run.events.length} steps</span>
      </div>
      {lastUser && (
        <div className="rounded-lg bg-local/15 px-3 py-2 text-sm">
          <div className="mb-0.5 font-mono text-[10px] uppercase tracking-wide text-muted/70">
            user said{run.turns.length > 1 ? ` · ${run.turns.length} turns seen` : ""}
          </div>
          <span className="whitespace-pre-wrap">{lastUser.content}</span>
        </div>
      )}
      <ol className="relative space-y-2 border-l border-line pl-4">
        {run.events.map((e) => <EventRow key={e.i} e={e} dataset={run.dataset} />)}
        {run.status === "running" && (
          <li className="flex items-center gap-2 text-xs text-muted">
            <Loader2 size={13} className="animate-spin" /> {nextStep(run.events)}
          </li>
        )}
      </ol>
      {run.result && run.status !== "running" && <ResultBlock run={run} />}
    </section>
  );
}

function nextStep(evs: AgentEvent[]): string {
  const last = evs[evs.length - 1]?.kind;
  if (!last || last === "context") return "working out the goal and hypotheses (LLM)…";
  if (last === "validate" || last === "hypothesis") return "checking hypotheses against the graph…";
  if (last === "evidence") return "checking the evidence against the need (LLM)…";
  if (last === "revise") return "validating the revised hypotheses…";
  return "working…";
}

const KIND_ICON: Record<string, any> = {
  context: Compass, goal: Target, waiting: CircleDashed, hypothesis: Lightbulb, validate: ShieldCheck,
  revise: RotateCcw, route: GitBranch, search: Search, fallback: TriangleAlert, evidence: FileText,
  check: CheckCircle2, retry: RotateCcw, result: CheckCircle2, error: XCircle,
};

function verdictClass(v?: string) {
  return v === "validated" || v === "answers" ? "border-good/50 text-good"
    : v === "weak" || v === "partial" ? "border-warn/50 text-warn"
      : v === "rejected" || v === "no" ? "border-bad/50 text-bad" : "border-line text-muted";
}

const docHref = (dataset: string, doc: string) => `/doc?doc=${encodeURIComponent(doc)}&dataset=${encodeURIComponent(dataset)}`;

function DocLink({ dataset, doc, title, inScope }: { dataset: string; doc: string; title?: string; inScope?: boolean }) {
  return (
    <Link to={docHref(dataset, doc)} target="_blank" title={doc}
      className={clsx("truncate hover:underline", inScope ? "text-good" : "text-ink/80")}>
      {title || doc}
    </Link>
  );
}

function Chips({ label, items, struck }: { label: string; items: string[]; struck?: boolean }) {
  if (!items?.length) return null;
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      <span className="font-mono text-[10px] text-muted">{label}</span>
      {items.map((x) => (
        <span key={x} className={clsx("rounded border px-1.5 py-px font-mono text-[10px]",
          struck ? "border-bad/40 text-bad line-through" : "border-line text-ink/80")}>{x}</span>
      ))}
    </span>
  );
}

function EventRow({ e, dataset }: { e: AgentEvent; dataset: string }) {
  const Icon = KIND_ICON[e.kind] ?? CircleDashed;
  const d = e.data ?? {};
  const head = (
    <div className="flex items-start gap-2">
      <Icon size={14} className={clsx("mt-0.5 shrink-0", e.kind === "error" ? "text-bad" : e.kind === "fallback" ? "text-warn" : "text-muted")} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-mono text-[10px] uppercase tracking-wide text-muted">{e.kind}</span>
          {(e.kind === "validate" || e.kind === "check") && d.verdict ? (
            <span className={clsx("rounded border px-1.5 py-px font-mono text-[10px]", verdictClass(d.verdict))}>{d.verdict}</span>
          ) : null}
          <span className={clsx("text-sm", e.kind === "goal" ? "font-medium text-ink" : "text-ink/90")}>{e.title}</span>
          <span className="ml-auto shrink-0 font-mono text-[10px] text-muted tabular" title="time since the previous step">
            +{(e.tookMs / 1000).toFixed(1)} s
          </span>
        </div>
        {e.detail && <div className="mt-0.5 text-xs text-muted">{e.detail}</div>}
      </div>
    </div>
  );
  return (
    <li className={clsx("relative", e.kind === "validate" && "ml-5")}>
      <span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-line" />
      {head}
      {e.kind === "goal" && d.clear === false && d.missing && (
        <div className="ml-6 mt-1 text-xs text-warn">still unclear: {d.missing}</div>
      )}
      {e.kind === "goal" && d.overruled && (
        <div className="ml-6 mt-1 text-xs text-muted" title="The model proposes, the code decides: a stated goal and need behind a substantive message counts as clear">
          model said unclear ({d.overruled}); overruled, since the goal and need are stated
        </div>
      )}
      {e.kind === "hypothesis" && d.hypothesis && (
        <div className="ml-6 mt-1 flex flex-wrap gap-x-3 gap-y-1">
          <Chips label="users" items={d.hypothesis.scope?.users} />
          <Chips label="teams" items={d.hypothesis.scope?.teams} />
          <Chips label="agents" items={d.hypothesis.scope?.agents} />
          <Chips label="files" items={d.hypothesis.scope?.files} />
          <Chips label="terms" items={d.hypothesis.scope?.terms} />
        </div>
      )}
      {e.kind === "validate" && (
        <div className="ml-6 mt-1 space-y-1">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            <Chips label="not in the conversation" items={d.dropped} struck />
            <Chips label="no such record" items={d.missing} struck />
            {d.scope && <span className="font-mono text-[10px] text-muted">scope: {d.scope.length} doc(s)</span>}
          </div>
          {(d.passages ?? d.cards)?.length > 0 && (
            <ul className="space-y-0.5">
              <li className="font-mono text-[10px] text-muted">{d.passages ? "top passages" : "top record cards"} (green = inside the scope)</li>
              {(d.passages ?? d.cards).map((c: any, i: number) => (
                <li key={i} className="flex gap-2 text-xs">
                  <span className="w-9 shrink-0 font-mono text-[10px] text-muted tabular">{typeof c.score === "number" ? c.score.toFixed(2) : "—"}</span>
                  <DocLink dataset={dataset} doc={c.doc} title={c.title} inScope={c.in_scope} />
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {e.kind === "route" && d.route && (
        <div className="ml-6 mt-1 font-mono text-[10px] text-muted">{ROUTE_HINT[d.route]} · target {d.target}{d.attempt > 1 ? ` · attempt ${d.attempt}` : ""}</div>
      )}
      {e.kind === "evidence" && (
        <ul className="ml-6 mt-1.5 space-y-1.5">
          {(d.evidence ?? []).map((x: any, i: number) => (
            <li key={i} className="rounded-md border border-line bg-bg px-2.5 py-1.5">
              <div className="mb-0.5 flex gap-2 text-xs"><span className="font-mono text-muted">[{i + 1}]</span><DocLink dataset={dataset} doc={x.doc} title={x.title} /></div>
              <div className="line-clamp-4 text-xs text-muted">{x.excerpt}</div>
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function ResultBlock({ run }: { run: AgentRunDetail }) {
  const r = run.result!;
  if (r.error) return null;
  if (r.waiting) return <div className="rounded-md border border-warn/40 px-3 py-2 text-sm text-warn">Stayed quiet: {r.waiting}</div>;
  return (
    <div className="space-y-2 rounded-md border border-line bg-bg p-3">
      <div className="flex items-baseline gap-2">
        <h4 className="caption">Result</h4>
        {r.route && <span className="font-mono text-[11px] text-muted">{r.route} route · check: {r.check?.verdict ?? "—"}</span>}
      </div>
      {r.answer && <p className="text-sm">{r.answer}</p>}
      <ol className="space-y-1">
        {r.docs.slice(0, 10).map((d, i) => (
          <li key={d.doc} className="flex items-baseline gap-2 text-sm">
            <span className="w-5 shrink-0 font-mono text-[11px] text-muted tabular">{i + 1}</span>
            <DocLink dataset={run.dataset} doc={d.doc} title={d.title} />
            <span className="ml-auto shrink-0 font-mono text-[10px] text-muted">{d.via}{typeof d.score === "number" ? ` · ${d.score.toFixed(2)}` : ""}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
