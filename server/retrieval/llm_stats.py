"""
In-memory record of every LLM and query-embedding call to the Ollama host, so the
backend can show how the models are doing: latency, failures, timeouts, empty
replies, and model (re)loads. Ollama reports load_duration per call; a load of more
than a second means the model was (re)loaded for that call, which is what happens
when two clients ask for the same model with different context sizes.

  GET /api/llm/stats   -> per-task summary + the last calls
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

MAX_CALLS = 500
CALLS: deque = deque(maxlen=MAX_CALLS)

# a call's task, from the first words of its system prompt
_TASKS = (
    ("You maintain a user profile", "ping: update profile"),
    ("You write the follow-up suggestions", "ping: follow-ups"),
    ("You are a helpful AI assistant", "chat reply"),
    ("You work out what a user needs", "agent: goal + hypotheses"),
    ("You check whether retrieved evidence", "evidence check"),
    ("You revise hypotheses", "agent: revise"),
    ("You are a precise assistant answering", "answer generation"),
    ("You write search queries", "assist: query"),
    ("You extract facts", "assist: facts"),
    ("You judge whether documents", "assist: judge"),
    ("You extract atomic", "metrics: claims"),
    ("You verify", "metrics: verify"),
    ("You judge context usefulness", "metrics: context precision"),
    ("You extract named entities", "metrics: entities"),
    ("You write questions", "metrics: answer relevancy"),
)


def task_of(system_prompt: Optional[str]) -> str:
    sp = (system_prompt or "").strip()
    for prefix, name in _TASKS:
        if sp.startswith(prefix):
            return name
    return sp[:40] or "(no system prompt: LightRAG / generation)"


def record(kind: str, model: str, task: str, t0: float, ok: bool, error: Optional[str] = None,
           prompt_chars: int = 0, out_chars: int = 0, ollama: Optional[dict] = None) -> None:
    o = ollama or {}
    ns = lambda k: round((o.get(k) or 0) / 1e6, 1)   # Ollama reports nanoseconds
    CALLS.append({
        "at": time.time(), "kind": kind, "model": model, "task": task, "ok": ok, "error": error,
        "ms": round((time.perf_counter() - t0) * 1000, 1), "prompt_chars": prompt_chars, "out_chars": out_chars,
        "load_ms": ns("load_duration"), "prompt_tokens": o.get("prompt_eval_count"), "out_tokens": o.get("eval_count"),
        "gen_ms": ns("eval_duration"),
    })


def _pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


def summary(last: int = 30) -> dict:
    by = {}
    for c in CALLS:
        by.setdefault((c["kind"], c["model"], c["task"]), []).append(c)
    tasks = []
    for (kind, model, task), cs in sorted(by.items()):
        ms = [c["ms"] for c in cs if c["ok"]]
        tps = [c["out_tokens"] / (c["gen_ms"] / 1000) for c in cs if c.get("out_tokens") and c.get("gen_ms")]
        tasks.append({
            "kind": kind, "model": model, "task": task, "calls": len(cs), "failed": sum(not c["ok"] for c in cs),
            "empty": sum(c["ok"] and c["kind"] == "llm" and c["out_chars"] == 0 for c in cs),
            "reloads": sum((c["load_ms"] or 0) > 1000 for c in cs),
            "p50_ms": _pct(ms, 0.5), "p95_ms": _pct(ms, 0.95), "max_ms": max(ms) if ms else None,
            "tokens_per_s": round(sum(tps) / len(tps), 1) if tps else None,
        })
    return {"window": len(CALLS), "tasks": tasks, "last": list(CALLS)[-last:]}
