import * as Tabs from "@radix-ui/react-tabs";
import { Loader2 } from "lucide-react";
import type { Chunk, RetrievalResult } from "@/lib/api";
import StepsPanel from "./StepsPanel";
import ContextPanel from "./ContextPanel";
import AnswerPanel from "./AnswerPanel";
import MetricsStrip from "./MetricsStrip";

type Props = {
  result: RetrievalResult | null;
  loading: boolean;
  labeled: boolean;
  tab: string;
  onTab: (t: string) => void;
  activeChunkId: string | null;
  onChunk: (c: Chunk | null) => void;
};

const TAB =
  "flex items-center gap-1.5 rounded px-3 py-1.5 text-sm text-muted transition-colors hover:text-ink data-[state=active]:bg-surface-2 data-[state=active]:text-ink";

function Count({ n }: { n: number }) {
  return <span className="rounded bg-bg px-1 font-mono text-[10px] tabular text-muted">{n}</span>;
}

/** One-line run summary: technique, latency, and what made it into context. */
function Summary({ r, loading }: { r: RetrievalResult; loading: boolean }) {
  return (
    <div className="flex items-center gap-2 font-mono text-[11px] text-muted">
      {loading && <Loader2 size={12} className="animate-spin text-local" />}
      <span className="rounded border border-line px-1.5 text-ink">{r.strategy}</span>
      <span className="tabular" title="end-to-end latency">{(r.timeMs / 1000).toFixed(2)} s</span>
      <span className="hidden tabular sm:inline" title="entities · relations · chunks">
        {r.graph.nodes.length}E · {r.graph.edges.length}R · {r.chunks.length}C
      </span>
    </div>
  );
}

export default function InspectorTabs({ result, loading, labeled, tab, onTab, activeChunkId, onChunk }: Props) {
  const empty = (
    <div className="flex h-full flex-col items-center justify-center gap-1 py-10 text-center">
      {loading ? <Loader2 size={18} className="animate-spin text-local" /> : null}
      <p className="text-sm text-muted">{loading ? "Retrieving…" : "Run a question to see the trace."}</p>
      {!loading && <p className="text-xs text-muted/70">Steps, ranked context and the generated answer appear here.</p>}
    </div>
  );
  return (
    <section className="card flex h-full min-h-0 flex-col gap-3">
      <Tabs.Root value={tab} onValueChange={onTab} className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Tabs.List className="flex gap-0.5 rounded-md border border-line bg-bg p-0.5" aria-label="Inspector">
            <Tabs.Trigger value="steps" className={TAB}>Steps{result && <Count n={result.steps.length} />}</Tabs.Trigger>
            <Tabs.Trigger value="context" className={TAB}>Context{result && <Count n={result.chunks.length} />}</Tabs.Trigger>
            <Tabs.Trigger value="answer" className={TAB}>Answer</Tabs.Trigger>
          </Tabs.List>
          {result && <Summary r={result} loading={loading} />}
        </div>
        <div className={`min-h-0 flex-1 overflow-y-auto pr-1 transition-opacity ${loading && result ? "opacity-50" : ""}`}>
          <Tabs.Content value="steps" className="h-full">
            {result ? <StepsPanel steps={result.steps} totalMs={result.timeMs} /> : empty}
          </Tabs.Content>
          <Tabs.Content value="context" className="h-full">
            {result ? <ContextPanel chunks={result.chunks} activeId={activeChunkId} onSelect={onChunk} /> : empty}
          </Tabs.Content>
          <Tabs.Content value="answer" className="h-full">
            {result ? <AnswerPanel answer={result.answer} chunkCount={result.chunks.length} onShowContext={() => onTab("context")} /> : empty}
          </Tabs.Content>
        </div>
      </Tabs.Root>
      <MetricsStrip result={result} labeled={labeled} />
    </section>
  );
}
