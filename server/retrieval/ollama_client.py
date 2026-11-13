"""
The backend's one client for the Ollama host: chat (generation, Ping, the agent, the
judge, metrics) and embeddings (index builds), on a shared aiohttp session. Every
chat call is recorded in llm_stats (/api/llm/stats).

Config (env), shared with vector_provider:
  PROTOGRAPH_EMBED_HOST      Ollama host (default: http://10.128.241.200:11434)
  PROTOGRAPH_EMBED_MODEL     default: nomic-embed-text
  PROTOGRAPH_LLM             default: gemma3:12b
  PROTOGRAPH_JUDGE_LLM       default: gpt-oss:20b  (the evidence check only; gemma3 misjudged
                             similar-symptom records as answers, gpt-oss rejected them)

Every call sends the same num_ctx: Ollama reloads a model when a request asks for a
different context size, so a client with another num_ctx makes the two thrash.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import aiohttp
import numpy as np

from retrieval import llm_stats

HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
EMBED = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")
LLM = os.getenv("PROTOGRAPH_LLM", "gemma3:12b")
JUDGE_LLM = os.getenv("PROTOGRAPH_JUDGE_LLM", "gpt-oss:20b")
NUM_CTX = 16384

_SESSION: Optional[aiohttp.ClientSession] = None


async def _ensure_session():
    global _SESSION
    if _SESSION is None or _SESSION.closed:
        _SESSION = aiohttp.ClientSession()


async def _ollama_llm(prompt, system_prompt=None, history_messages=None, model=None, **kwargs):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += list(history_messages or [])
    msgs.append({"role": "user", "content": prompt})
    model = model or LLM
    body = {"model": model, "messages": msgs, "stream": False,
            "options": {"num_ctx": NUM_CTX, "temperature": 0.0}, "keep_alive": "30m"}
    t0, d = time.perf_counter(), {}
    try:
        async with _SESSION.post(f"{HOST}/api/chat", json=body,
                                 timeout=aiohttp.ClientTimeout(total=900)) as r:
            d = await r.json()
    except BaseException as e:   # incl. cancellation: a timed-out caller still shows up
        llm_stats.record("llm", model, llm_stats.task_of(system_prompt), t0, False, f"{type(e).__name__}: {e}"[:160],
                         prompt_chars=sum(len(m["content"]) for m in msgs))
        raise
    out = d.get("message", {}).get("content", "")
    llm_stats.record("llm", model, llm_stats.task_of(system_prompt), t0, not d.get("error"), d.get("error"),
                     prompt_chars=sum(len(m["content"]) for m in msgs), out_chars=len(out or ""), ollama=d)
    return out


async def _ollama_embed(texts, **kwargs):
    prefix = "search_query: " if kwargs.get("context") == "query" else "search_document: "
    async with _SESSION.post(f"{HOST}/api/embed",
                             json={"model": EMBED, "input": [prefix + t for t in texts],
                                   "keep_alive": "30m"},
                             timeout=aiohttp.ClientTimeout(total=300)) as r:
        d = await r.json()
    return np.array(d["embeddings"], dtype=np.float32)


_GEN_SYS = (
    "You are a precise assistant answering questions about a team's chat sessions "
    "and documents. Answer ONLY from the provided context. If the context does not "
    "contain the answer, say so plainly. Be concise and cite the source titles you used."
)


async def llm(prompt: str, system_prompt: str = None, model: str = None) -> str:
    """One-shot chat call (default model: PROTOGRAPH_LLM)."""
    await _ensure_session()
    return await _ollama_llm(prompt, system_prompt=system_prompt, model=model)


async def embed(texts, as_query: bool = False):
    """Embedding call (nomic). Returns an (n, dim) float32 np.array."""
    await _ensure_session()
    return await _ollama_embed(list(texts), context="query" if as_query else "document")


async def generate_over(query: str, context_text: str) -> str:
    """Answer a query grounded in the retrieved context (the Answer tab), with the same
    LLM for every technique so answers are comparable."""
    await _ensure_session()
    if not context_text.strip():
        return ""
    prompt = (f"Context:\n{context_text}\n\n"
              f"Question: {query}\n\n"
              f"Answer from the context above:")
    try:
        return await _ollama_llm(prompt, system_prompt=_GEN_SYS)
    except Exception:
        return ""


async def aclose():
    global _SESSION
    if _SESSION is not None:
        await _SESSION.close()
        _SESSION = None
