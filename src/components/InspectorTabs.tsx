import * as Tabs from "@radix-ui/react-tabs";
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
  "rounded px-3 py-1.5 text-sm text-muted transition-colors hover:text-ink data-[state=active]:bg-surface-2 data-[state=active]:text-ink";

export default function InspectorTabs({ result, loading, labeled, tab, onTab, activeChunkId, onChunk }: Props) {
  const empty = (
    <p className="py-8 text-center text-sm text-muted">
      {loading ? "Retrieving…" : "Pick a sample question or type your own to see the trace."}
    </p>
  );
  return (
    <section className="card flex h-full min-h-0 flex-col gap-3">
      <Tabs.Root value={tab} onValueChange={onTab} className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex items-center justify-between">
          <Tabs.List className="flex gap-1 rounded-md border border-line bg-bg p-0.5" aria-label="Inspector">
            <Tabs.Trigger value="steps" className={TAB}>Steps</Tabs.Trigger>
            <Tabs.Trigger value="context" className={TAB}>Context</Tabs.Trigger>
            <Tabs.Trigger value="answer" className={TAB}>Answer</Tabs.Trigger>
          </Tabs.List>
          {result && <span className="font-mono text-[11px] text-muted">{result.strategy}{loading ? " · updating…" : ""}</span>}
        </div>
        <div className={`min-h-0 flex-1 overflow-y-auto pr-1 transition-opacity ${loading && result ? "opacity-60" : ""}`}>
          <Tabs.Content value="steps">{result ? <StepsPanel steps={result.steps} /> : empty}</Tabs.Content>
          <Tabs.Content value="context">
            {result ? <ContextPanel chunks={result.chunks} activeId={activeChunkId} onSelect={onChunk} /> : empty}
          </Tabs.Content>
          <Tabs.Content value="answer">
            {result ? <AnswerPanel answer={result.answer} chunkCount={result.chunks.length} /> : empty}
          </Tabs.Content>
        </div>
      </Tabs.Root>
      <MetricsStrip result={result} labeled={labeled} />
    </section>
  );
}
