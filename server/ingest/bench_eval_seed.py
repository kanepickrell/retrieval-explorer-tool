"""
Seed eval_set.json for the bench dataset: one entry per substantial doc, so
generate_eval_queries.py can write reworded questions whose answer is that doc.

  sessions  with at least MIN_REQUESTS user requests and some real content
            (not just "open Firefox" / "close that")
  notes / issues / measurements  every doc with enough text

query = the doc's title (its opening request), gold = [the doc]. The title query
appears verbatim in the doc, so it is a smoke test; the generated questions are
the benchmark.

Usage:
  python bench_eval_seed.py <dataset_dir>     (reads <dataset_dir>/docs, writes eval_set.json)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MIN_REQUESTS = 3
MIN_CHARS = 600
UI_ONLY = re.compile(r"^(open|close|minimi[sz]e|maximi[sz]e|make|go to|scroll|click|turn|show|leave|exit|switch)\b", re.I)


def main(ds: str) -> None:
    d = Path(ds)
    out = []
    for f in sorted((d / "docs").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        if len(text) < MIN_CHARS:
            continue
        title = text.splitlines()[0].lstrip("# ").strip()
        if f.name.startswith("session-"):
            asks = re.findall(r"^## User\n(.+)", text, re.M)
            real = [a for a in asks if not UI_ONLY.match(a) and len(a.split()) >= 4]
            if len(asks) < MIN_REQUESTS or len(real) < 2:
                continue
        out.append({"id": f.stem, "query": title, "gold_ids": [f.name],
                    "query_type": "title", "breadth": "focused"})
    (d / "eval_set.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    kinds = {}
    for e in out:
        k = e["id"].split("-", 1)[0]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"{len(out)} gold entries {kinds} -> {d / 'eval_set.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
