import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import clsx from "clsx";
import { Activity, ChevronDown, ChevronRight, Pause, Play, Users, X } from "lucide-react";
import {
  getLlmStats, getOpsSummary, listOpsJobs, listOpsUsers,
  type LlmTask, type OpsJob, type OpsStage, type OpsSummary, type OpsUser,
} from "@/lib/api";

/**
 * Ops: ProtoGraph as a service. Every chime-in call (a Ping cycle from Open WebUI, or the
 * Explorer's chat replay) is a job, persisted by the backend (server/ops.py). The page
 * shows the whole picture first (who is using it, how much, how fast, how often Ping
 * speaks up, whether the models are healthy), then the job stream to review what happened.
 */

const WINDOWS = [{ h: 1, label: "1h" }, { h: 24, label: "24h" }, { h: 24 * 7, label: "7d" }];
const JOBS_MS = 2000;
const STATS_MS = 5000;
const KEEP = 200;

const STAGE_LABEL: Record<string, string> = {
  observe: "observe", update: "update", check: "L1", gate: "gate", dive: "dive", suggest: "suggest", probe: "probe", learn: "learn",
};

export default function Ops() {
  const [hours, setHours] = useState(24);
  const [paused, setPaused] = useState(false);
  const [user, setUser] = useState<string | null>(null);
  const [summary, setSummary] = useState<OpsSummary | null>(null);
  const [users, setUsers] = useState<OpsUser[]>([]);
  const [models, setModels] = useState<LlmTask[]>([]);
  const [jobs, setJobs] = useState<OpsJob[]>([]);
  const [down, setDown] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now() / 1000);
  const rev = useRef(0);

  // KPI strip, users and model health: a slower poll
  useEffect(() => {
    if (paused) return;
    let stop = false, t: number | undefined;
    const tick = async () => {
      try {
        const [s, u, m] = await Promise.all([getOpsSummary(hours), listOpsUsers(hours), getLlmStats().catch(() => ({ tasks: [] }))]);
        if (stop) return;
        setSummary(s); setUsers(u); setModels(m.tasks ?? []); setDown(null);
      } catch (e: any) {
        if (!stop) setDown(String(e?.message ?? e));
      }
      if (!stop) t = window.setTimeout(tick, STATS_MS);
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [hours, paused]);

  // job stream: only rows that changed since the last rev (new jobs, and running ones that finished)
  useEffect(() => {
    rev.current = 0;
    setJobs([]);
  }, [hours, user]);
  useEffect(() => {
    if (paused) return;
    let stop = false, t: number | undefined;
    const tick = async () => {
      try {
        const d = await listOpsJobs(hours, rev.current, user ?? undefined);
        if (stop) return;
        if (d.rev < rev.current) { rev.current = 0; setJobs([]); }   // db reset: start over
        if (d.jobs.length) {
          setJobs((cur) => {
            const byId = new Map(cur.map((j) => [j.id, j]));
            d.jobs.forEach((j) => byId.set(j.id, j));
            return [...byId.values()].sort((a, b) => b.id - a.id).slice(0, KEEP);
          });
        }
        rev.current = d.rev;
      } catch { /* the stats poll reports the outage */ }
      if (!stop) t = window.setTimeout(tick, JOBS_MS);
    };
    tick();
    return () => { stop = true; window.clearTimeout(t); };
  }, [hours, user, paused]);

  useEffect(() => { const t = window.setInterval(() => setNow(Date.now() / 1000), 5000); return () => window.clearInterval(t); }, []);

  const pickedName = users.find((u) => u.user === user)?.user_name ?? user;

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Activity size={16} className={clsx(paused || down ? "text-muted" : "text-good")} />
          <h2 className="font-title text-lg font-semibold">Ops</h2>
        </div>
        <span className="font-mono text-[11px] text-muted">
          {down ? `backend unreachable (${down})` : paused ? "paused" : "live · chime-in jobs from every user"}
        </span>
        <div role="radiogroup" aria-label="Time window" className="ml-auto flex rounded-md border border-line bg-bg p-0.5">
          {WINDOWS.map((w) => (
            <button key={w.h} role="radio" aria-checked={hours === w.h} onClick={() => setHours(w.h)}
              className={hours === w.h ? "rounded bg-surface-2 px-3 py-1 text-xs text-ink" : "rounded px-3 py-1 text-xs text-muted hover:text-ink"}>
              {w.label}
            </button>
          ))}
        </div>
        <button className="btn h-8" onClick={() => setPaused((p) => !p)}>
          {paused ? <><Play size={12} /> Resume</> : <><Pause size={12} /> Pause</>}
        </button>
      </div>

      <KpiStrip s={summary} />
      <ModelHealth tasks={models} />

      <div className="grid gap-4 lg:grid-cols-[minmax(240px,320px)_1fr]">
        <UsersPanel users={users} picked={user} onPick={(u) => setUser((cur) => (cur === u ? null : u))} now={now} />
        <section className="min-w-0 space-y-2">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold">Jobs</h3>
            <span className="font-mono text-[11px] text-muted">newest first · one per chime-in call</span>
            {user && (
              <button className="ml-auto flex items-center gap-1 rounded border border-line px-2 py-0.5 text-[11px] text-muted hover:text-ink"
                onClick={() => setUser(null)}>
                {pickedName} <X size={11} />
              </button>
            )}
          </div>
          {jobs.length === 0 ? (
            <div className="card text-sm text-muted">
              No jobs in this window yet. They appear as people chat in Open WebUI with Chime In on
              (or when you replay a chat in the Explorer's chat mode).
            </div>
          ) : (
            <ul className="space-y-2">{jobs.map((j) => <JobRow key={j.id} j={j} now={now} showUser={!user} />)}</ul>
          )}
        </section>
      </div>
    </div>
  );
}

// ───────────────────────── KPI strip ─────────────────────────

function KpiStrip({ s }: { s: OpsSummary | null }) {
  const v = (x: number | null | undefined, f: (n: number) => string = String) => (x === null || x === undefined ? "—" : f(x));
  const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const tiles: { label: string; value: string; sub: string; tone?: "good" | "warn" | "bad" }[] = [
    { label: "Users", value: v(s?.users), sub: `${v(s?.active_users)} active (30 min)` },
    { label: "Sessions", value: v(s?.sessions), sub: `${v(s?.active_sessions)} active chats` },
    { label: "Jobs", value: v(s?.jobs), sub: s?.running ? `${s.running} running now` : "none running" },
    { label: "Chime rate", value: v(s?.chime_rate, (n) => `${Math.round(n * 100)}%`),
      sub: s ? `${s.suggested} suggested · ${s.quiet} quiet` : "suggested / finished" },
    { label: "Latency", value: v(s?.p50_ms, secs), sub: `p95 ${v(s?.p95_ms, secs)}`,
      tone: s?.p95_ms && s.p95_ms > 25000 ? "warn" : undefined },
    { label: "Errors", value: v(s?.errors), sub: "failed or lost jobs", tone: s?.errors ? "bad" : "good" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
      {tiles.map((t) => (
        <div key={t.label} className="card p-3">
          <div className="font-mono text-[10px] uppercase tracking-wide text-muted">{t.label}</div>
          <div className={clsx("mt-0.5 font-title text-2xl font-semibold tabular-nums",
            t.tone === "bad" && "text-bad", t.tone === "warn" && "text-warn")}>{t.value}</div>
          <div className="truncate text-[11px] text-muted">{t.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ───────────────────────── model health ─────────────────────────

function ModelHealth({ tasks }: { tasks: LlmTask[] }) {
  // one chip per model: total calls, failures, median latency, generation speed
  const models = useMemo(() => {
    const by = new Map<string, { model: string; kind: string; calls: number; failed: number; p50: number[]; tps: number[] }>();
    for (const t of tasks) {
      const m = by.get(t.model) ?? { model: t.model, kind: t.kind, calls: 0, failed: 0, p50: [], tps: [] };
      m.calls += t.calls; m.failed += t.failed;
      if (t.p50_ms != null) m.p50.push(t.p50_ms);
      if (t.tokens_per_s != null) m.tps.push(t.tokens_per_s);
      by.set(t.model, m);
    }
    return [...by.values()];
  }, [tasks]);
  return (
    <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
      <span className="text-muted">models</span>
      {models.length === 0 && <span className="text-muted">no model calls since the engine started</span>}
      {models.map((m) => {
        const bad = m.failed > 0 && m.failed >= m.calls / 2;
        const p50 = m.p50.length ? Math.max(...m.p50) : null;
        const tps = m.tps.length ? m.tps.reduce((a, b) => a + b, 0) / m.tps.length : null;
        return (
          <span key={m.model} className="flex items-center gap-1.5 rounded-md border border-line bg-surface px-2 py-1"
            title={`${m.kind} · ${m.calls} calls since the engine started`}>
            <span className={clsx("h-1.5 w-1.5 rounded-full", bad ? "bg-bad" : m.failed ? "bg-warn" : "bg-good")} />
            <span className="text-ink">{m.model}</span>
            <span className="text-muted">
              {m.calls} calls{m.failed ? ` · ${m.failed} failed` : ""}{p50 != null ? ` · p50 ${(p50 / 1000).toFixed(1)}s` : ""}
              {tps != null ? ` · ${Math.round(tps)} tok/s` : ""}
            </span>
          </span>
        );
      })}
    </div>
  );
}

// ───────────────────────── users ─────────────────────────

function UsersPanel({ users, picked, onPick, now }: { users: OpsUser[]; picked: string | null; onPick: (u: string) => void; now: number }) {
  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2">
        <Users size={14} className="text-muted" />
        <h3 className="text-sm font-semibold">Users</h3>
        <span className="font-mono text-[11px] text-muted">click to filter jobs</span>
      </div>
      {users.length === 0 ? (
        <div className="card text-sm text-muted">Nobody yet in this window.</div>
      ) : (
        <ul className="space-y-1.5">
          {users.map((u) => (
            <li key={u.user}>
              <button onClick={() => onPick(u.user)}
                className={clsx("card w-full p-2.5 text-left transition-colors hover:bg-surface-2",
                  picked === u.user && "ring-1 ring-vector")}>
                <div className="flex items-center gap-2">
                  <span className={clsx("h-1.5 w-1.5 shrink-0 rounded-full",
                    u.running ? "animate-pulse bg-vector" : now - u.last_at < 1800 ? "bg-good" : "bg-muted/50")} />
                  <span className="truncate text-sm font-medium">{u.user_name}</span>
                  <span className="ml-auto shrink-0 font-mono text-[10px] text-muted">{ago(now, u.last_at)}</span>
                </div>
                <div className="mt-1 font-mono text-[10px] text-muted">
                  {u.sessions} chat{u.sessions === 1 ? "" : "s"} · {u.jobs} jobs · {u.suggested} suggested
                  {u.errors ? <span className="text-bad"> · {u.errors} errors</span> : null}
                </div>
                {u.confidence != null && <ConfBar conf={u.confidence} gate={0.65} near={0.5} className="mt-1.5" label="Ping on their latest chat" />}
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ───────────────────────── jobs ─────────────────────────

const STATUS: Record<OpsJob["status"], { label: string; cls: string }> = {
  running: { label: "running", cls: "border-vector/60 text-vector" },
  suggested: { label: "suggested", cls: "border-good/60 text-good" },
  quiet: { label: "quiet", cls: "border-line text-muted" },
  error: { label: "error", cls: "border-bad/60 text-bad" },
  lost: { label: "lost", cls: "border-warn/60 text-warn" },
};

function JobRow({ j, now, showUser }: { j: OpsJob; now: number; showUser: boolean }) {
  const [open, setOpen] = useState(false);
  const d = j.detail;
  const st = STATUS[j.status] ?? STATUS.quiet;
  const mood = d?.mood && "state" in d.mood ? d.mood : null;
  const sugg = d?.suggestions ?? [];
  return (
    <li className="card p-0">
      <button className="w-full space-y-1.5 p-3 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-muted">
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <span className="text-ink" title={new Date(j.at * 1000).toLocaleString()}>{new Date(j.at * 1000).toLocaleTimeString()}</span>
          {showUser && <span className="text-ink/80">{j.user_name}</span>}
          <span className={clsx("rounded border px-1.5 py-px", st.cls, j.status === "running" && "animate-pulse")}>{st.label}</span>
          {j.chat_model && <span title="the model the user chatted with">{j.chat_model}</span>}
          <span>{j.engine}</span>
          {j.source !== "open-webui" && <span>· {j.source}</span>}
          {mood && <span className={clsx(mood.state === "near" && "text-warn", mood.state === "delivered" && "text-good")}>· {mood.state}</span>}
          <span className="ml-auto">{j.total_ms != null ? `${(j.total_ms / 1000).toFixed(1)}s` : j.status === "running" ? `${Math.round(now - j.at)}s…` : ""}</span>
        </div>
        {j.message && <div className="line-clamp-2 text-sm">{j.message}</div>}
        {d?.stages?.length ? <StagePipe stages={d.stages} /> : null}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
          {sugg.length > 0 ? (
            <span className="text-good">→ {sugg.map((s) => s.title || s.doc).join(", ")}</span>
          ) : j.reason ? (
            <span className={j.status === "error" || j.status === "lost" ? "text-bad" : "text-muted"}>{j.reason}</span>
          ) : null}
          {j.confidence != null && (
            <ConfBar conf={j.confidence} gate={j.gate ?? 0.65} near={mood?.near ?? 0.5} delta={j.delta} className="ml-auto w-40" />
          )}
        </div>
      </button>
      {open && <JobDetail j={j} />}
    </li>
  );
}

function StagePipe({ stages }: { stages: OpsStage[] }) {
  return (
    <div className="flex flex-wrap items-center gap-1 font-mono text-[10px]">
      {stages.map((s, i) => (
        <span key={s.name + i} title={`${s.title}${s.ms ? ` · ${s.ms} ms` : ""}`}
          className={clsx("rounded px-1.5 py-px",
            s.status === "done" && "bg-surface-2 text-ink/80",
            s.status === "skip" && "text-muted/60 line-through decoration-muted/40",
            s.status === "pending" && "text-muted/40",
            s.status === "error" && "bg-bad/20 text-bad",
            s.name === "gate" && s.status === "done" && /^Open/.test(s.title) && "bg-good/20 text-good",
            s.name === "gate" && s.status === "done" && /^Closed/.test(s.title) && "bg-warn/15 text-warn")}>
          {STAGE_LABEL[s.name] ?? s.name}{s.ms >= 1000 ? ` ${(s.ms / 1000).toFixed(1)}s` : ""}
        </span>
      ))}
    </div>
  );
}

function JobDetail({ j }: { j: OpsJob }) {
  const d = j.detail;
  const mood = d?.mood && "state" in d.mood ? d.mood : null;
  return (
    <div className="space-y-3 border-t border-line px-3 pb-3 pt-2.5 text-[12px]">
      {d?.criteria?.length ? (
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-muted">gate · {d.gate_title}</div>
          <ul className="grid gap-x-4 gap-y-0.5 sm:grid-cols-2">
            {d.criteria.map((c) => (
              <li key={c.name} className="flex gap-1.5">
                <span className={c.ok ? "text-good" : "text-bad"}>{c.ok ? "✓" : "✗"}</span>
                <span>{c.name}</span>
                <span className="truncate text-muted" title={c.value}>{c.value}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {d?.hypotheses?.length ? (
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-muted">live hypotheses</div>
          <ul className="space-y-1">
            {d.hypotheses.slice().sort((a, b) => b.conf - a.conf).map((h) => (
              <li key={h.id} className="flex items-center gap-2">
                <span className="w-10 shrink-0 font-mono text-[10px] text-muted">{h.id}</span>
                <span className="min-w-0 flex-1 truncate" title={h.claim}>{h.claim}</span>
                <ConfBar conf={h.conf} gate={j.gate ?? 0.65} near={0.5} className="w-28 shrink-0" />
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {mood?.blocked?.length ? (
        <div className="text-warn">confident, but held back by: {mood.blocked.join(", ")}</div>
      ) : mood?.waiting?.length ? (
        <div className="text-warn">near: waiting for {mood.waiting.join(" and ")}</div>
      ) : null}
      {d?.follow_ups?.length ? (
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-muted">follow-ups Ping offered</div>
          <ul className="list-disc space-y-0.5 pl-4">{d.follow_ups.map((f) => <li key={f}>{f}</li>)}</ul>
        </div>
      ) : null}
      {d?.suggestions?.length ? (
        <div>
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-muted">suggested</div>
          {d.suggestions.map((s) => (
            <div key={s.doc}><span className="text-ink">{s.title || s.doc}</span>{s.why ? <span className="text-muted"> · {s.why}</span> : null}</div>
          ))}
        </div>
      ) : null}
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted">
        <span>job #{j.id}</span>
        {j.chat_id && <span>chat {j.chat_id.slice(0, 8)}</span>}
        {j.dataset && <span>{j.dataset}</span>}
        {d?.turns_run != null && <span>{d.turns_run} turn{d.turns_run === 1 ? "" : "s"} run</span>}
        {d?.engine_models?.llm && <span title="what Ping thought with">ping: {d.engine_models.llm} · judge {d.engine_models.judge}</span>}
        {j.session_id && (
          <Link to={`/agent?session=${encodeURIComponent(j.session_id)}`} className="text-vector hover:underline"
            title="Open the Ping session (profile, hypotheses, every cycle) on the Agent page. Sessions live in memory: gone after an engine restart.">
            open session {j.session_id} →
          </Link>
        )}
      </div>
    </div>
  );
}

// ───────────────────────── bits ─────────────────────────

/** Confidence vs the gate: fill = confidence, the solid tick = gate, the faint tick = where "near" starts. */
function ConfBar({ conf, gate, near, delta, className, label }:
  { conf: number; gate: number; near: number; delta?: number | null; className?: string; label?: string }) {
  const pct = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`;
  const tone = conf >= gate ? "bg-good" : conf >= near ? "bg-warn" : "bg-muted/60";
  return (
    <div className={clsx("flex items-center gap-1.5", className)}
      title={`${label ? label + ": " : ""}confidence ${conf.toFixed(2)} (gate ${gate.toFixed(2)})${delta ? `, ${delta > 0 ? "+" : ""}${delta.toFixed(2)} this job` : ""}`}>
      <div className="relative h-1.5 flex-1 rounded-full bg-surface-2">
        <div className={clsx("absolute inset-y-0 left-0 rounded-full", tone)} style={{ width: pct(conf) }} />
        <div className="absolute -inset-y-0.5 w-px bg-muted/40" style={{ left: pct(near) }} />
        <div className="absolute -inset-y-0.5 w-px bg-ink/70" style={{ left: pct(gate) }} />
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted">
        {conf.toFixed(2)}{delta ? <span className={delta > 0 ? "text-good" : "text-bad"}>{delta > 0 ? "↑" : "↓"}</span> : null}
      </span>
    </div>
  );
}

function ago(now: number, t: number): string {
  const s = Math.max(0, Math.round(now - t));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
