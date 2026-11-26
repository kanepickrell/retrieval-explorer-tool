"""
Agent-loop (Ping's retrieval) on every gold question of an export: real requests,
generated, structural and the answer key. Scores hit@1 / hit@5 against the gold docs
and records the route, whether a scoped read happened, and the evidence check.

  .venv/Scripts/python.exe tests/agent_eval.py [--dataset <id>] [--tag name] [--limit N]

Set AGENT_SCOPED_READ=0 for the old behaviour (filter a global chunk search by the
scope) to compare against the scoped read (search the scoped docs' own passages).
Writes tests/agent_eval.<tag>.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
import agent_loop as AL  # noqa: E402
import datasets as DS  # noqa: E402
from retrieval import ollama_client as ollama  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agenthub-export-2026-11-25-personas")
    ap.add_argument("--tag", default="scoped" if AL.SCOPED_READ else "baseline")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    gold = [g for g in DS.gold_entries(DS.get(a.dataset)) if g.get("gold_ids") and g.get("query")]
    if a.limit:
        gold = gold[:a.limit]
    print(f"{len(gold)} gold questions · scoped read {'ON' if AL.SCOPED_READ else 'OFF'}", flush=True)
    rows = []
    for i, g in enumerate(gold, 1):
        run = AL.AgentRun(a.dataset, [{"role": "user", "content": g["query"]}], "eval")
        t = time.perf_counter()
        res = await AL.run_agent(run, require_clear=False)
        docs = [d["doc"] for d in res.get("docs", [])]
        gid = set(g["gold_ids"])
        scoped = any(e.get("kind") == "search" and (e.get("data") or e).get("scoped") for e in run.events) \
            if run.events and isinstance(run.events[0], dict) else None
        row = {"id": g.get("id"), "type": (g.get("query_type") or "real").split("_")[0], "q": g["query"][:160],
               "gold": g["gold_ids"], "top5": docs[:5], "hit1": bool(docs[:1] and docs[0] in gid),
               "hit5": bool(gid & set(docs[:5])), "route": res.get("route"),
               "check": (res.get("check") or {}).get("verdict"), "scoped": scoped,
               "error": res.get("error"), "s": round(time.perf_counter() - t, 1)}
        rows.append(row)
        print(f"{i:3}/{len(gold)} {'H1' if row['hit1'] else ('h5' if row['hit5'] else '--')} "
              f"{row['type']:10} {str(row['route']):9} {str(row['check']):8} {row['s']:5}s  {row['q'][:70]}", flush=True)
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for k in (r["type"], "ALL"):
            by[k][0] += 1
            by[k][1] += r["hit1"]
            by[k][2] += r["hit5"]
    print("\n type        n  hit@1  hit@5")
    for k, (n, h1, h5) in sorted(by.items(), key=lambda kv: kv[0] == "ALL"):
        print(f" {k:10} {n:3}  {h1:5}  {h5:5}")
    print(f" errors: {sum(1 for r in rows if r['error'])}")
    out = Path(__file__).with_name(f"agent_eval.{a.tag}.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    await ollama.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
