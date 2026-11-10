import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { evaluate, generateAnswer, getGraph, TECHNIQUES, type Chunk, type GraphEdge, type GraphNode, type Metric } from "@/lib/api";
import { useExplorer } from "@/lib/explorer-context";
import { useHotkeys } from "@/lib/hotkeys";
import { deriveSignals } from "@/lib/signals";
import QuestionCard from "@/components/QuestionCard";
import ModeCard from "@/components/ModeCard";
import GraphCanvas from "@/components/GraphCanvas";
import InspectorTabs from "@/components/InspectorTabs";

const EMPTY = { nodes: [] as GraphNode[], edges: [] as GraphEdge[] };

export default function Explorer() {
  const {
    datasetId, highlight, setHighlight,
    query, run, technique, setTechnique, topK, setTopK, chunkTopK, setChunkTopK,
    result, resultKey, loading,
  } = useExplorer();

  const [graph, setGraph] = useState(EMPTY);
  const [input, setInput] = useState(query);
  const [tab, setTab] = useState("steps");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeChunk, setActiveChunk] = useState<Chunk | null>(null);
  const [answer, setAnswer] = useState<{ key: string; text: string } | null>(null);
  const [answerLoading, setAnswerLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const answerReq = useRef(0);                       // guards against superseded responses
  const answerCache = useRef(new Map<string, string>()); // generated answer per run

  // identifies the answer for the current run; changes when the question, technique or knobs change
  const answerKey = result ? `${query}:::${technique}:::${topK}:::${chunkTopK}:::${resultKey}` : "";

  // Generate the answer lazily — only when the Answer tab is open, once per run (cached).
  // Generation is a slow LLM call (global/hybrid ~7-8s), so retrieval never waits on it, and a
  // request-id guard means switching technique mid-flight supersedes cleanly instead of stalling.
  useEffect(() => {
    if (tab !== "answer" || !result || !answerKey) return;
    const cached = answerCache.current.get(answerKey);
    if (cached !== undefined) { setAnswer({ key: answerKey, text: cached }); setAnswerLoading(false); return; }
    const id = ++answerReq.current;
    setAnswerLoading(true);
    generateAnswer(datasetId, query, technique, chunkTopK).then((r) => {
      const text = r.answer || (r.note ? `_${r.note}_` : "_No answer was generated._");
      answerCache.current.set(answerKey, text);       // cache even if the view moved on
      if (id !== answerReq.current) return;            // a newer request superseded this one
      setAnswer({ key: answerKey, text });
      setAnswerLoading(false);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, answerKey]);

  // Ragas metrics: computed on demand (expensive: retrieval + generation + judge calls).
  const [evalState, setEvalState] = useState<{ key: string; metrics: Metric[] } | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const evalReq = useRef(0);
  const runEvaluate = useCallback(() => {
    if (!result || !answerKey) return;
    const key = answerKey;
    const id = ++evalReq.current;
    setEvalLoading(true);
    evaluate(datasetId, query, technique, topK, chunkTopK).then((r) => {
      if (id !== evalReq.current) return;
      setEvalState({ key, metrics: r.metrics });
      setEvalLoading(false);
    });
  }, [result, answerKey, datasetId, query, technique, topK, chunkTopK]);

  useEffect(() => {
    let live = true;
    getGraph(datasetId).then((g) => live && setGraph(g));
    return () => { live = false; };
  }, [datasetId]);

  // a new result invalidates the chunk the user had clicked
  useEffect(() => { setActiveChunk(null); }, [resultKey]);

  useHotkeys({
    "/": () => inputRef.current?.focus(),
    ...Object.fromEntries(TECHNIQUES.map((t, i) => [String(i + 1), () => setTechnique(t)])),
    Escape: () => { setSelectedId(null); setActiveChunk(null); setHighlight(null); },
  });

  const signals = useMemo(() => deriveSignals(result), [result]);
  const chunkHighlight = useMemo(
    () => (activeChunk ? { label: activeChunk.title, ids: activeChunk.entities ?? [] } : null),
    [activeChunk],
  );

  return (
    <main className="mt-5 flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <QuestionCard value={input} onChange={setInput} onRun={run} loading={loading}
          activeQuery={result ? query : ""} inputRef={inputRef} />
        <ModeCard technique={technique} onTechnique={setTechnique} topK={topK} onTopK={setTopK} chunkTopK={chunkTopK} onChunkTopK={setChunkTopK} />
      </div>
      <div className="grid grid-cols-1 gap-4 lg:h-[700px] lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div className="h-[560px] lg:h-full">
          <GraphCanvas
            graph={graph}
            signals={signals}
            resultKey={resultKey}
            highlight={chunkHighlight ?? highlight}
            onClearHighlight={() => { setActiveChunk(null); setHighlight(null); }}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>
        <div className="min-h-[560px] lg:h-full lg:min-h-0">
          <InspectorTabs
            result={result}
            loading={loading}
            tab={tab}
            onTab={setTab}
            activeChunkId={activeChunk?.id ?? null}
            onChunk={setActiveChunk}
            answer={answer?.key === answerKey ? answer.text : ""}
            answerLoading={answerLoading}
            evalMetrics={evalState?.key === answerKey ? evalState.metrics : null}
            evalLoading={evalLoading}
            onEvaluate={runEvaluate}
          />
        </div>
      </div>
    </main>
  );
}
