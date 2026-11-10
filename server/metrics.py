"""
Ragas-style evaluation metrics, computed for real with the same local judge
(gemma3 via lightrag_provider.llm) and embeddings (nomic via .embed) the rest of
the backend uses. These are faithful reimplementations of Ragas's algorithms,
not the `ragas` pip package (which is brittle with small local models) -- the
formulas match the Ragas docs and are easy to swap for the library later.

Reference-free (computable on ANY query):
  faithfulness       = supported_claims / total_claims
  answer_relevancy   = mean cosine(query, questions generated from the answer)
  context_precision  = Average Precision@K with per-chunk LLM relevance (no ref)

Reference-based (need gold) are handled by the caller:
  recall_at_k        = |retrieved_topk ∩ gold| / |gold|   (deterministic)
  context_recall / context_entities_recall / noise_sensitivity need a gold
  ANSWER/entities -> left to the gold-labeled datasets, reported as "needs gold".
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from retrieval import lightrag_provider as lp

# ───────────────────────── gold set (reference-based) ─────────────────────────

_GOLD_PATH = os.getenv("PROTOGRAPH_GOLD",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "datasets", "agenthub_focused.json"))
_GOLD: Optional[list] = None
_STOP = set("a an the and or of on in to for is are was were my this that these with "
            "get getting keeps up out".split())


def _load_gold() -> list:
    global _GOLD
    if _GOLD is None:
        try:
            with open(_GOLD_PATH, encoding="utf-8") as f:
                _GOLD = json.load(f)
        except Exception:
            _GOLD = []
    return _GOLD


def _toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def match_gold(query: str, threshold: float = 0.4) -> Optional[dict]:
    """Return the best-matching gold entry (by token Jaccard) if the query is a
    labeled eval query, else None -> reference-based metrics stay 'needs gold'."""
    q = _toks(query)
    if not q:
        return None
    best, best_s = None, 0.0
    for e in _load_gold():
        g = _toks(e.get("query", ""))
        if not g:
            continue
        s = len(q & g) / len(q | g)
        if s > best_s:
            best, best_s = e, s
    return best if best_s >= threshold else None

_NONCOMMITTAL = re.compile(r"\b(i (don'?t|do not) know|cannot (answer|determine)|no (answer|information|relevant))\b", re.I)


def _json_list(text: str):
    """Pull the first JSON array out of an LLM reply, tolerating code fences and
    prose around it. Returns a Python list, or [] on failure."""
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _cos(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ───────────────────────── reference-free metrics ─────────────────────────

async def faithfulness(answer: str, contexts: List[str]) -> Optional[float]:
    """supported_claims / total_claims. Decompose the answer into standalone
    claims, then verify each against the retrieved context."""
    if not answer.strip() or _NONCOMMITTAL.search(answer):
        return None
    ctx = "\n\n".join(contexts).strip()
    if not ctx:
        return None
    claims = _json_list(await lp.llm(
        "Break the ANSWER into a list of standalone, verifiable factual claims. "
        "Return ONLY a JSON array of short strings.\n\nANSWER:\n" + answer,
        system_prompt="You extract atomic factual claims. Output JSON only."))
    claims = [str(c) for c in claims if str(c).strip()][:15]
    if not claims:
        return None
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verdicts = _json_list(await lp.llm(
        "For each CLAIM, decide if it can be directly inferred from the CONTEXT. "
        "Return ONLY a JSON array of booleans, one per claim, in order.\n\n"
        f"CONTEXT:\n{ctx}\n\nCLAIMS:\n{numbered}",
        system_prompt="You verify claims against context. Output a JSON array of true/false only."))
    bools = [bool(v) for v in verdicts][:len(claims)]
    if not bools:
        return None
    return round(sum(1 for v in bools if v) / len(claims), 3)


async def answer_relevancy(query: str, answer: str, n: int = 3) -> Optional[float]:
    """Mean cosine similarity between the original query and questions generated
    from the answer. A noncommittal answer scores 0."""
    if not answer.strip():
        return None
    if _NONCOMMITTAL.search(answer):
        return 0.0
    qs = _json_list(await lp.llm(
        f"Generate {n} distinct questions that the ANSWER below directly and fully "
        f"answers. Return ONLY a JSON array of {n} question strings.\n\nANSWER:\n" + answer,
        system_prompt="You write questions a given answer would answer. Output JSON only."))
    qs = [str(q) for q in qs if str(q).strip()][:n]
    if not qs:
        return None
    vecs = await lp.embed([query] + qs, as_query=True)
    qv, gen = vecs[0], vecs[1:]
    sims = [_cos(qv, g) for g in gen]
    if not sims:
        return None
    return round(max(0.0, min(1.0, sum(sims) / len(sims))), 3)


async def context_precision(answer: str, contexts: List[str]) -> Optional[float]:
    """Average Precision@K (Ragas, reference-free): per-chunk relevance to the
    answer, weighted so relevant chunks ranked higher score better."""
    ctxs = [c for c in contexts if c and c.strip()]
    if not answer.strip() or not ctxs:
        return None
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(ctxs))
    verdicts = _json_list(await lp.llm(
        "Each CONTEXT chunk below is numbered. For each, decide if it was useful "
        "for arriving at the ANSWER. Return ONLY a JSON array of 0/1, one per chunk, in order.\n\n"
        f"ANSWER:\n{answer}\n\nCONTEXTS:\n{numbered}",
        system_prompt="You judge context usefulness. Output a JSON array of 0/1 only."))
    v = [1 if x in (1, True, "1") else 0 for x in verdicts][:len(ctxs)]
    if not v:
        return None
    total_rel = sum(v)
    if total_rel == 0:
        return 0.0
    # AP@K = Σ_k (precision@k · v_k) / total_relevant
    ap, hits = 0.0, 0
    for k, vk in enumerate(v, start=1):
        if vk:
            hits += 1
            ap += (hits / k)
    return round(ap / total_rel, 3)


# ───────────────────────── reference-based (deterministic) ─────────────────────────

def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """|retrieved_topk ∩ gold| / |gold|."""
    if not gold_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hit = sum(1 for g in gold_ids if g in top)
    return round(hit / len(gold_ids), 3)


# ───────────────────────── orchestrator ─────────────────────────

async def reference_free(query: str, answer: str, contexts: List[str]) -> Dict[str, Optional[float]]:
    """Compute the three reference-free metrics. Each may be None if it couldn't
    be computed (empty answer, judge parse failure); the caller renders None as
    'needs gold'/'—' rather than a fabricated number."""
    import asyncio
    f, ar, cp = await asyncio.gather(
        faithfulness(answer, contexts),
        answer_relevancy(query, answer),
        context_precision(answer, contexts),
    )
    return {"faithfulness": f, "answer_relevancy": ar, "context_precision": cp}
