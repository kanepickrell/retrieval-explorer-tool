"""
Batch retrieval eval: every gold query of a dataset (eval_set.json +
eval_set_generated.json) through each strategy, on the dataset and its index
variants, scored deterministically -- no answer generation, no LLM judge.

  hit@k   share of queries whose gold doc is among the top-k distinct docs
  MRR     mean 1/rank of the first gold doc (0 when missed within `limit` chunks)
  ms      median retrieval latency (LightRAG modes include keyword extraction,
          cached after the first mode sees a query)

Rows are split by query type: "title" (the doc's own opening request, a smoke
test) vs generated_* (reworded questions, the real benchmark).

Usage:
  python eval_recall.py <dataset_id> [--modes vector,bm25,bm25_vector,local,global,hybrid,mix]
                        [--k 5] [--limit 20] [--out results.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import datasets as DS  # noqa: E402
import graph_search as GS  # noqa: E402


def distinct_docs(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


async def run(dataset_id: str, modes: list[str], k: int, limit: int) -> list[dict]:
    variants = [d for d in DS._registry() if d == dataset_id or d.startswith(dataset_id + "~")]
    gold = DS.gold_entries(DS.get(dataset_id))
    print(f"{len(gold)} queries x {len(modes)} modes x {len(variants)} indexes", flush=True)
    rows = []
    for ds_id in variants:
        for mode in modes:
            per_type = defaultdict(lambda: {"hits": 0, "rr": 0.0, "n": 0, "ms": []})
            for e in gold:
                t = time.perf_counter()
                try:
                    res = await GS.run_unified(e["query"], limit, mode, ds_id)
                    ids = GS._get_run(res["run_id"]).retrieved_ids
                except Exception as ex:  # noqa: BLE001
                    print(f"  ! {ds_id} {mode} {e['id']}: {ex!r}"[:160], flush=True)
                    ids = []
                ms = (time.perf_counter() - t) * 1000
                docs = distinct_docs(ids)
                rank = next((i for i, d in enumerate(docs, 1) if d in set(e["gold_ids"])), None)
                qt = "title" if e.get("query_type") == "title" else "generated"
                for key in (qt, "all"):
                    s = per_type[key]
                    s["n"] += 1
                    s["hits"] += bool(rank and rank <= k)
                    s["rr"] += 1 / rank if rank else 0
                    s["ms"].append(ms)
            for qt, s in per_type.items():
                rows.append({"index": ds_id, "mode": mode, "queries": qt, "n": s["n"],
                             f"hit@{k}": round(s["hits"] / s["n"], 3), "mrr": round(s["rr"] / s["n"], 3),
                             "ms": round(statistics.median(s["ms"]))})
            g = next(r for r in rows[::-1] if r["mode"] == mode and r["index"] == ds_id and r["queries"] == "generated") \
                if "generated" in per_type else rows[-1]
            print(f"{ds_id:28} {mode:12} generated hit@{k}={g[f'hit@{k}']:.2f} mrr={g['mrr']:.2f} {g['ms']}ms",
                  flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--modes", default="vector,bm25,bm25_vector,local,global,hybrid,mix")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = asyncio.run(run(a.dataset, a.modes.split(","), a.k, a.limit))
    key = f"hit@{a.k}"
    print(f"\n{'index':28} {'mode':12} {'queries':10} {'n':>4} {key:>7} {'mrr':>6} {'ms':>6}")
    for r in sorted(rows, key=lambda r: (r["queries"], -r[key], -r["mrr"])):
        print(f"{r['index']:28} {r['mode']:12} {r['queries']:10} {r['n']:4} {r[key]:7.3f} {r['mrr']:6.3f} {r['ms']:6}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
