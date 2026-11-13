"""
In-memory store of retrieval runs, so answer generation and evaluation operate
on EXACTLY the retrieval the user is looking at instead of re-running it.

A run is created by /api/search/unified and holds everything downstream steps
need: the retrieved doc ids (for recall@k), the chunk texts (for the judge), the
full context text the generator sees, and -- for LightRAG modes -- the exact
system prompt LightRAG built for that context. The answer is generated once and
cached on the run, so the Answer tab and the metrics score the same answer.

Bounded LRU; persistence comes with run history.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_RUNS = 300


@dataclass
class Run:
    id: str
    dataset: str
    query: str
    strategy: str
    limit: int
    retrieved_ids: List[str]          # doc ids in rank order (recall@k is doc-level)
    contexts: List[str]               # chunk texts in the final context (judge input)
    context_text: str                 # everything the generator sees
    retrieval_ms: float
    created_at: float = field(default_factory=time.time)
    answer: Optional[str] = None
    answer_ms: Optional[float] = None
    answer_note: Optional[str] = None


_RUNS: "OrderedDict[str, Run]" = OrderedDict()


def new_run(**kw) -> Run:
    r = Run(id=uuid.uuid4().hex[:12], **kw)
    _RUNS[r.id] = r
    while len(_RUNS) > MAX_RUNS:
        _RUNS.popitem(last=False)
    return r


def get(run_id: str) -> Optional[Run]:
    r = _RUNS.get(run_id)
    if r is not None:
        _RUNS.move_to_end(run_id)
    return r


def all_runs() -> Dict[str, Run]:
    return _RUNS
