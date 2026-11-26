"""
One command for the whole ingest of an export folder (as written by deploy/export-local.sh
or the shared server's export_package.py):

  1. export_to_docs        records -> docs/*.md (+ catalogs)
  2. tag_docs              topics + document types from the semantic model (LLM, cached;
                           --incremental when an earlier run exists)
  3. build_passages        full text: passages + embeddings (no LLM; unchanged passages reused)
  4. build_metadata_graph  the graph: records, relations, properties, mentions links
  5. eval sets             eval_set.json (chat reviews) + eval_set_structural.json (graph facts);
                           --generated also writes LLM-written questions (slow)
  6. schema check          conformance of the new graph to schema.yaml (fails on violations
                           with --strict)

Stops at the first failing step. Each step prints its own summary; this prints the timings.

  python ingest/run_ingest.py <export_dir> [--generated] [--strict] [--full-retag]

teams.json (user-NN -> team) is not in an export: copy it into the export folder before
running if teams should be in the graph. Pseudonyms only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent
PY = sys.executable


def step(name: str, args: list) -> float:
    print(f"\n── {name} ──", flush=True)
    t = time.time()
    r = subprocess.run([PY, *map(str, args)], cwd=SERVER)
    if r.returncode:
        raise SystemExit(f"[ingest] FAILED at {name} (exit {r.returncode})")
    return time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--generated", action="store_true", help="also write LLM-generated eval questions")
    ap.add_argument("--strict", action="store_true", help="fail when the graph breaks the schema")
    ap.add_argument("--full-retag", action="store_true", help="re-tag every record (ignore the tag cache)")
    a = ap.parse_args()
    ex = Path(a.export_dir).resolve()
    if not (ex / "openwebui").is_dir():
        raise SystemExit(f"{ex} doesn't look like an export (no openwebui/)")
    if not (ex / "teams.json").exists():
        print("[ingest] note: no teams.json in the export; the graph will have no teams")
    t: dict = {}
    t["docs"] = step("1. export_to_docs", [HERE / "export_to_docs.py", ex, ex / "docs"])
    tag = [HERE / "tag_docs.py", ex] + (["--force"] if a.full_retag else
                                         ["--incremental"] if (ex / "tags.json").exists() else [])
    t["tags"] = step("2. tag_docs", tag)
    t["passages"] = step("3. build_passages", [HERE / "build_passages.py", ex / "docs", ex / "storage"])
    t["graph"] = step("4. build_metadata_graph", [HERE / "build_metadata_graph.py", ex])
    t["evals"] = step("5a. eval set (chat reviews)", [HERE / "build_eval_set.py", ex, ex / "docs", ex / "eval_set.json"])
    t["evals"] += step("5b. structural eval set", [HERE / "build_structural_gold.py", ex])
    if a.generated:
        t["evals"] += step("5c. generated eval questions", [HERE / "generate_eval_queries.py", ex])

    print("\n── 6. schema check ──", flush=True)
    sys.path.insert(0, str(SERVER))
    import schema_model as SM
    schema, _ = SM.load()
    m = SM.measure(schema, ex)
    bad = (m.get("violations") or 0) + (m.get("property_violations") or 0)
    unknown = {**(m.get("unknown_classes") or {}), **(m.get("unknown_relations") or {})}
    print(f"[ingest] schema v{schema.get('version')}: {m.get('nodes')} nodes, {m.get('edges')} edges; "
          f"{m.get('violations')} relation violation(s), {m.get('property_violations')} property violation(s)"
          + (f"; not in the schema: {json.dumps(unknown)}" if unknown else ""))
    print("[ingest] timings: " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items())
          + f"; total {sum(t.values()) / 60:.1f} min")
    if a.strict and (bad or unknown):
        raise SystemExit("[ingest] FAILED the schema check (--strict)")
    print(f"[ingest] done: {ex.name} (datasets '{ex.name}' and '{ex.name}~metadata')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
