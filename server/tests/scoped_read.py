"""
Named-record content questions: the question names a record (a user's file, a team...)
and the answer is a fact only inside that record's text. Passes when the agent returns
that record AND the passage it read contains the fact, i.e. it followed the graph's link
into the record's full text.

  AGENT_DEEP=20 .venv/Scripts/python.exe tests/scoped_read.py     (AGENT_SCOPED_READ=0 for the old way)

Expectations are for agenthub-export-2026-11-25-personas.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
import agent_loop as AL  # noqa: E402
from retrieval import ollama_client as ollama  # noqa: E402

DATASET = "agenthub-export-2026-11-25-personas"
CASES = [
    ("What is the stop word in the rules of engagement user-03 uploaded?", "file-60a65d40", ["lighthouse"]),
    ("In user-02's range revert runbook, which node hosts enclave 11?", "file-1ccbfbaf", ["pve01"]),
    ("What are the stages of user-05's GitLab pipeline file?", "file-00d5471a", ["verify"]),
    ("In user-04's MSEL, what is inject 2 about?", "file-2d8f6010", ["morris", "inv-20931"]),
    ("What default timeout does user-05's healthcheck script use?", "file-acfc7b11", ["600"]),
    ("What are the exercise dates in user-03's rules of engagement?", "file-60a65d40", ["2026-12-01"]),
]


async def main() -> int:
    print(f"scoped read {'ON' if AL.SCOPED_READ else 'OFF'} · search depth {AL.DEEP}")
    ok = 0
    for q, doc, facts in CASES:
        run = AL.AgentRun(DATASET, [{"role": "user", "content": q}], "eval")
        res = await AL.run_agent(run, require_clear=False)
        top = res.get("docs", [])[:3]
        hit = next((d for d in top if d["doc"].startswith(doc)), None)
        has = bool(hit) and any(f in (hit["text"] or "").lower() for f in facts)
        followed = any(e["kind"] == "search" and e["title"].startswith("Scoped read") for e in run.events)
        ok += has
        print(f"{'PASS' if has else 'FAIL'}  route {str(res.get('route')):8} {'followed link' if followed else '             '}"
              f"  record {'in top 3' if hit else 'MISSING '}  fact {'found' if has else 'not in passage'}  · {q}")
    print(f"\n{ok}/{len(CASES)} passed")
    await ollama.aclose()
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
