import { useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { Bot, Loader2, Play, RotateCcw, Send, SkipForward, Sparkles, Square } from "lucide-react";
import {
  assist, listReplayChats, TECHNIQUE_LABEL,
  type AssistResult, type ChatTurn, type QueryStrategy, type ReplayChat,
} from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";

/** A transcript entry: a chat message, or the retrieval explorer chiming in after a user message. */
type Entry =
  | { kind: "msg"; turn: ChatTurn }
  | { kind: "chime"; status: "loading" }
  | { kind: "chime"; status: "done"; res: AssistResult }
  | { kind: "chime"; status: "error"; error: string };

const STRATEGIES: { id: QueryStrategy; label: string; title: string }[] = [
  { id: "rewrite", label: "LLM rewrite", title: "The LLM writes one standalone search query for what would help now, or decides nothing is worth looking up" },
  { id: "last", label: "last message", title: "Search with the latest user message as-is" },
  { id: "window", label: "last 3 turns", title: "Search with the last three messages concatenated" },
];

/**
 * Chat mode: play out a conversation (replay a real held-out chat, or type your own)
 * and watch the retrieval explorer chime in after each user message with docs that
 * would help. Clicking a chime-in opens its retrieval in the graph / inspector panels.
 */
export default function ChatPanel() {
  const { datasetId, technique, chunkTopK, showResult, reportError } = useExplorer();
  const [chats, setChats] = useState<ReplayChat[]>([]);
  const [chatId, setChatId] = useState<string>("");          // "" = type your own
  const [entries, setEntries] = useState<Entry[]>([]);
  const [cursor, setCursor] = useState(0);                     // next turn index of the replayed chat
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [draft, setDraft] = useState("");
  const [strategy, setStrategy] = useState<QueryStrategy>("rewrite");
  const [llmFilter, setLlmFilter] = useState(true);
  const [minScore, setMinScore] = useState(0.5);
  const stopRef = useRef(false);
  const suggestedRef = useRef<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const chat = chats.find((c) => c.chatId === chatId);

  useEffect(() => {
    if (!datasetId) return;
    listReplayChats(datasetId).then(setChats).catch((e) => reportError(String(e.message ?? e)));
    reset();
  }, [datasetId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [entries]);

  function reset() {
    stopRef.current = true;
    setPlaying(false);
    setEntries([]);
    setCursor(0);
    suggestedRef.current = [];
  }

  const transcript = (es: Entry[]): ChatTurn[] =>
    es.filter((e): e is { kind: "msg"; turn: ChatTurn } => e.kind === "msg").map((e) => e.turn);

  /** Add a user message, then ask the backend whether to chime in. */
  async function userSays(text: string, base: Entry[]): Promise<Entry[]> {
    const withMsg: Entry[] = [...base, { kind: "msg", turn: { role: "user", content: text } }];
    const idx = withMsg.length;
    setEntries([...withMsg, { kind: "chime", status: "loading" }]);
    let chime: Entry;
    try {
      const res = await assist({
        datasetId, turns: transcript(withMsg), technique, queryStrategy: strategy,
        limit: chunkTopK, llmFilter, minScore,
        excludeDocs: chat ? [chat.doc] : [], alreadySuggested: suggestedRef.current,
      });
      suggestedRef.current = [...suggestedRef.current, ...res.suggestions.map((s) => s.doc)];
      chime = { kind: "chime", status: "done", res };
    } catch (e) {
      chime = { kind: "chime", status: "error", error: String((e as Error).message ?? e) };
    }
    const next = [...withMsg.slice(0, idx), chime];
    setEntries(next);
    return next;
  }

  /** Replay: the next user message (+ chime-in), then the original assistant reply. */
  async function step(base: Entry[], at: number): Promise<{ es: Entry[]; at: number } | null> {
    if (!chat || at >= chat.turns.length) return null;
    let es = base, i = at;
    while (i < chat.turns.length && chat.turns[i].role !== "user") {      // leading assistant turns
      es = [...es, { kind: "msg", turn: chat.turns[i] }]; i++;
    }
    if (i >= chat.turns.length) { setEntries(es); return null; }
    es = await userSays(chat.turns[i].content, es); i++;
    while (i < chat.turns.length && chat.turns[i].role !== "user") {
      es = [...es, { kind: "msg", turn: chat.turns[i] }]; i++;
    }
    setEntries(es);
    return { es, at: i };
  }

  async function nextTurn() {
    setBusy(true);
    const r = await step(entries, cursor);
    if (r) setCursor(r.at); else setCursor(chat?.turns.length ?? 0);
    setBusy(false);
  }

  async function playAll() {
    stopRef.current = false;
    setPlaying(true); setBusy(true);
    let es = entries, at = cursor;
    while (!stopRef.current) {
      const r = await step(es, at);
      if (!r) break;
      es = r.es; at = r.at; setCursor(at);
    }
    setPlaying(false); setBusy(false);
  }

  async function send() {
    const t = draft.trim();
    if (!t || busy) return;
    setDraft(""); setBusy(true);
    await userSays(t, entries);
    setBusy(false);
  }

  // conversation stats (what a perf run would aggregate)
  const done = entries.filter((e): e is Extract<Entry, { status: "done" }> => e.kind === "chime" && e.status === "done");
  const spoke = done.filter((e) => e.res.suggestions.length > 0).length;
  const avgMs = done.length ? Math.round(done.reduce((a, e) => a + e.res.totalMs, 0) / done.length) : 0;
  const finished = !!chat && cursor >= chat.turns.length;

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <select value={chatId} onChange={(e) => { reset(); setChatId(e.target.value); }}
          className="input h-8 min-w-0 max-w-[340px] flex-1 py-0 text-xs" aria-label="Conversation source">
          <option value="">Type your own conversation</option>
          {chats.map((c) => (
            <option key={c.chatId} value={c.chatId}>
              Replay: {c.title.slice(0, 60)} · {c.userTurns} turns{c.author ? ` · ${c.author}` : ""}
            </option>
          ))}
        </select>
        <div className="flex items-center rounded-md border border-line bg-bg p-0.5" role="radiogroup" aria-label="Query strategy">
          {STRATEGIES.map((s) => (
            <button key={s.id} role="radio" aria-checked={strategy === s.id} title={s.title} onClick={() => setStrategy(s.id)}
              className={clsx("rounded px-2 py-1 font-mono text-[11px]", strategy === s.id ? "bg-surface-2 text-ink" : "text-muted hover:text-ink")}>
              {s.label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 font-mono text-[11px] text-muted" title="LLM checks each candidate: would this help right now? (adds ~1-3 s per turn)">
          <input type="checkbox" checked={llmFilter} onChange={(e) => setLlmFilter(e.target.checked)} /> LLM check
        </label>
        <label className="flex items-center gap-1.5 font-mono text-[11px] text-muted" title="minimum cosine(query, chunk) for a doc to be considered">
          min score
          <input type="number" min={0} max={1} step={0.05} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))}
            className="input h-7 w-16 px-1.5 py-0 text-[11px]" />
        </label>
      </div>

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-2 overflow-y-auto rounded-md border border-line bg-bg/60 p-3">
        {entries.length === 0 && (
          <p className="py-6 text-center text-xs text-muted">
            {chat
              ? `Replays "${chat.title}" (${chat.userTurns} user turns). Its own doc is hidden from retrieval, so it can't find itself. Step through with Next turn or Play all.`
              : "Type messages as a user would. After each one, the retrieval explorer decides whether anything in the corpus would help, and chimes in."}
            <br />Retrieval uses the mode set in the Mode card ({TECHNIQUE_LABEL[technique]}).
          </p>
        )}
        {entries.map((e, i) => (e.kind === "msg"
          ? <Message key={i} turn={e.turn} />
          : <Chime key={i} entry={e} onOpen={(r) => r.retrieval && showResult(r.retrieval)} />))}
      </div>

      <div className="flex items-center gap-2">
        {chat ? (
          <>
            <button className="btn-primary h-8 px-3" onClick={nextTurn} disabled={busy || finished}>
              {busy && !playing ? <Loader2 size={13} className="animate-spin" /> : <SkipForward size={13} />} Next turn
            </button>
            {playing
              ? <button className="btn h-8" onClick={() => { stopRef.current = true; }}><Square size={12} /> Stop</button>
              : <button className="btn h-8" onClick={playAll} disabled={busy || finished}><Play size={12} /> Play all</button>}
          </>
        ) : (
          <form className="flex min-w-0 flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); send(); }}>
            <input className="input h-8 min-w-0 flex-1 text-sm" placeholder="Write a message as the user…" value={draft}
              onChange={(e) => setDraft(e.target.value)} aria-label="Message" />
            <button className="btn-primary h-8 px-3" disabled={busy || !draft.trim()}>
              {busy ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
            </button>
          </form>
        )}
        <button className="btn h-8" onClick={reset} title="Clear the conversation"><RotateCcw size={12} /></button>
        <span className="ml-auto font-mono text-[11px] text-muted" title="chime-ins with suggestions / user turns checked · average time per turn">
          {done.length ? `${spoke}/${done.length} turns chimed in · ${(avgMs / 1000).toFixed(1)} s avg` : ""}
          {chat ? ` · turn ${Math.min(cursor, chat.turns.length)}/${chat.turns.length}` : ""}
        </span>
      </div>
    </div>
  );
}

function Message({ turn }: { turn: ChatTurn }) {
  const user = turn.role === "user";
  return (
    <div className={clsx("flex", user ? "justify-end" : "justify-start")}>
      <div className={clsx("max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm",
        user ? "bg-local/15 text-ink" : "border border-line bg-surface text-muted")}>
        <div className="mb-0.5 font-mono text-[10px] uppercase tracking-wide text-muted/70">{user ? "user" : "assistant"}</div>
        <span className="line-clamp-[8]">{turn.content}</span>
      </div>
    </div>
  );
}

export function Chime({ entry, onOpen }: { entry: Exclude<Entry, { kind: "msg" }>; onOpen: (r: AssistResult) => void }) {
  if (entry.status === "loading") {
    return (
      <div className="flex items-center gap-2 pl-1 font-mono text-[11px] text-muted">
        <Loader2 size={12} className="animate-spin text-local" /> retrieval explorer is looking…
      </div>
    );
  }
  if (entry.status === "error") {
    return <div className="pl-1 font-mono text-[11px] text-red-400">chime-in failed: {entry.error}</div>;
  }
  const r = entry.res;
  const timing = r.stages.map((s) => `${s.label}: ${Math.round(s.ms)} ms`).join(" · ");
  if (r.suggestions.length === 0) {
    return (
      <div className="pl-1 font-mono text-[11px] text-muted/70" title={timing}>
        <Bot size={11} className="mr-1 inline" /> stayed quiet — {r.query ? `searched "${r.query}", ` : ""}{r.reason}
        {r.retrieval && <button className="ml-2 underline hover:text-ink" onClick={() => onOpen(r)}>inspect</button>}
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-vector/40 bg-vector/[0.07] p-2.5">
      <button onClick={() => onOpen(r)} className="mb-1.5 flex w-full items-center gap-1.5 text-left font-mono text-[11px] text-vector hover:underline"
        title={`Open this retrieval in the graph and inspector · ${timing}`}>
        <Sparkles size={12} /> retrieval explorer found {r.suggestions.length} doc{r.suggestions.length > 1 ? "s" : ""} that might help
        <span className="ml-auto text-muted">{(r.totalMs / 1000).toFixed(1)} s</span>
      </button>
      <ul className="space-y-1.5">
        {r.suggestions.map((s) => (
          <li key={s.doc} className="rounded-md border border-line bg-surface/70 px-2.5 py-1.5">
            <div className="flex items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate text-sm font-medium" title={s.doc}>{s.title}</span>
              {s.score != null && <span className="font-mono text-[10px] text-muted" title="cosine(query, chunk)">{s.score.toFixed(2)}</span>}
            </div>
            {s.why && <div className="text-xs text-ink/90">{s.why}</div>}
            <div className="line-clamp-2 text-[11px] text-muted">{s.snippet}</div>
          </li>
        ))}
      </ul>
      {r.query && <div className="mt-1.5 truncate font-mono text-[10px] text-muted/70" title={r.query}>query: {r.query}</div>}
    </div>
  );
}
