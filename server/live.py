"""
Live feed of /api/assist calls, so the Live page can show chime-in traffic as it
arrives (from Open WebUI's Chime In, or from the Explorer's own chat replay).

In memory only: the last MAX_EVENTS calls, lost on restart. Each event keeps the
request settings, the last few turns the engine saw, and the full assist
response (query, stages, suggestions, retrieval), so the page can open the
retrieval in the Explorer exactly as it ran.

  GET /api/live?after=<seq>   -> {"latest": <seq>, "events": [newest last]}
"""
from __future__ import annotations

import itertools
import time
from collections import deque
from typing import Optional

from fastapi import APIRouter, Query

MAX_EVENTS = 100
TURNS_KEPT = 4
TURN_CHARS = 1200

_EVENTS: deque = deque(maxlen=MAX_EVENTS)
_SEQ = itertools.count(1)
_latest = 0

router = APIRouter(prefix="/api/live", tags=["live"])


def _source(user_agent: Optional[str]) -> str:
    """Open WebUI's chime_in router calls with httpx; the Explorer UI is a browser."""
    ua = (user_agent or "").lower()
    if "httpx" in ua or "python" in ua:
        return "open-webui"
    if "mozilla" in ua:
        return "explorer"
    return "other"


def record(req, resp: dict, user_agent: Optional[str] = None, error: Optional[str] = None) -> None:
    global _latest
    seq = next(_SEQ)
    _latest = seq
    _EVENTS.append({
        "seq": seq,
        "at": time.time(),
        "source": _source(user_agent),
        "dataset": req.dataset,
        "technique": req.technique,
        "query_strategy": req.query_strategy,
        "limit": req.limit,
        "top_k": req.top_k,
        "llm_filter": req.llm_filter,
        "min_score": req.min_score,
        "turns": [{"role": t.role, "content": t.content[-TURN_CHARS:]} for t in req.turns[-TURNS_KEPT:]],
        "error": error,
        "response": resp,
    })


@router.get("")
def live(after: int = Query(0, ge=0)):
    return {"latest": _latest, "events": [e for e in _EVENTS if e["seq"] > after]}
