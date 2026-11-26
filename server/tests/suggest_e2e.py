"""
End-to-end check of bottom-up vocabulary growth: new data the schema doesn't cover produces
term suggestions, the review actions change the schema, and `tag_docs.py --incremental`
applies an approved term without re-classifying everything.

  1. sandbox copy of an export (<export>-suggesttest; storage/ is not copied)
  2. new records the current schema has no term for: an after-action report file, a chat
     turning a hotwash into an AAR (topic gap), and two Sigma detection rules (doc type gap)
  3. tag_docs.py      -> the records still get existing terms, and the gaps come back as
                         suggestions built from their own words
  4. GET suggestions  -> grouped, checked against the schema, "ready" with enough support
  5. review via the API: approve the AAR topic and the detection-rule type, merge a name
     into a term as an alias, reject a Hub tag; a stale base_version gets 409
  6. tag_docs.py --incremental -> far fewer model calls than a full re-tag; the new terms
     land on the right records; no unrelated tag is removed; nothing is stale
  7. build_metadata_graph -> graph conforms and has the new Topic node
  8. restore schema.yaml and remove the test's history backups (always)

  .venv/Scripts/python.exe tests/suggest_e2e.py [--source <export>] [--keep] [--shots DIR]

--shots saves headless-Chrome screenshots of the suggestions tab before and after review.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema_e2e as E   # noqa: E402  (sandbox helpers and the shared new records)
from schema_e2e import API, CONTAINER, HISTORY, INCOMING, PY, SCHEMA, SERVER, check, metrics, results, run, rw  # noqa: E402

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

SIGMA2 = """title: Remote service creation via sc.exe (lateral movement)
id: 7c2d9e41-ledger-0002
status: experimental
description: sc.exe creating a service on a remote host, as seen in BROKEN LEDGER inject 6
logsource:
  product: windows
  category: process_creation
detection:
  selection:
    Image|endswith: '\\\\sc.exe'
    CommandLine|contains|all: ['\\\\\\\\', ' create ', 'binPath=']
  condition: selection
falsepositives: [software deployment by the range admins]
level: medium
tags: [attack.lateral_movement, attack.t1021.002, attack.t1543.003]
"""


def add_second_rule(ex: Path) -> str:
    fid = "e2e00000-0000-4000-8000-0000000aa003"
    rw(ex / "openwebui/file.json", lambda rows: rows + [{
        "id": fid, "user_id": "user-02", "filename": "remote_service_creation.yml", "hash": fid, "created_at": 1795600000,
        "meta": {"name": "remote_service_creation.yml", "content_type": "application/x-yaml", "size": len(SIGMA2), "data": {}},
        "data": {"status": "completed", "content": SIGMA2}}])
    return f"file-{fid}.md"


def shot(shots: Path | None, name: str, ds: str):
    if not shots or not CHROME.exists():
        return
    shots.mkdir(parents=True, exist_ok=True)
    out = shots / f"{name}.png"
    subprocess.run([str(CHROME), "--headless=new", "--window-size=1440,2200", "--virtual-time-budget=20000",
                    f"--screenshot={out}", f"http://localhost:5180/prospector?dataset={ds}~metadata&model=suggestions"],
                   capture_output=True, timeout=120)
    print(f"  screenshot {out}")


def calls(out: str) -> tuple:
    """(re-check calls of --incremental, model calls of the main pass)"""
    inc = re.search(r"--incremental: (\d+) re-check model calls", out)
    main = re.search(r"\((\d+) model calls to", out)
    return int(inc.group(1)) if inc else 0, int(main.group(1)) if main else 0


def find(groups, taxonomy, docs):
    """The suggestion group (in a taxonomy) whose evidence covers the most of these docs."""
    best = max((g for g in groups if g["taxonomy"] == taxonomy),
               key=lambda g: sum(e["doc"] in docs for e in g["evidence"]), default=None)
    return best if best and any(e["doc"] in docs for e in best["evidence"]) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="agenthub-export-2026-11-25-personas")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--shots", type=Path)
    a = ap.parse_args()

    src, ex = INCOMING / a.source, INCOMING / f"{a.source}-suggesttest"
    original = SCHEMA.read_text(encoding="utf-8")
    cand_path = SCHEMA.parent / "candidates.json"     # tag_docs.py's suggestion grouping
    cand_before = cand_path.read_bytes() if cand_path.exists() else None
    hist_before = set(HISTORY.glob("*.yaml")) if HISTORY.is_dir() else set()
    c = httpx.Client(timeout=120)
    try:
        print("1-2. sandbox + records the schema has no term for")
        if ex.exists():
            shutil.rmtree(ex)
        shutil.copytree(src, ex, ignore=shutil.ignore_patterns("storage", "schema_metrics.json"))
        ids = E.add_records(ex)
        ids["sigma2"] = add_second_rule(ex)
        run([PY, "ingest/export_to_docs.py", str(ex), str(ex / "docs")], cwd=SERVER)
        aar_docs, rule_docs = {ids["aar"], ids["chat"]}, {ids["sigma"], ids["sigma2"]}

        print("3. tag_docs (current schema)")
        out = run([PY, "ingest/tag_docs.py", str(ex)], cwd=SERVER)
        print("\n".join("    " + l for l in out.strip().splitlines()[-17:]))
        tags = json.loads((ex / "tags.json").read_text(encoding="utf-8"))
        new = list(ids.values())
        check("new records still get existing terms", all(tags.get(d) for d in new),
              "; ".join(f"{d[:13]}…: {','.join(tags.get(d, []))}" for d in new))
        cands = tags.get("_candidates", {})
        check("gaps recorded as candidates", any(d in cands for d in new),
              ", ".join(f"{c_['taxonomy']}:{c_['term']}" for d in new for c_ in cands.get(d, [])))

        print("4. suggestions")
        sg = c.get(f"{API}/suggestions", params={"dataset": ex.name}).json()
        for g in sg["groups"]:
            print(f"    {'READY' if g['ready'] else 'watch'} {g['taxonomy']:9} {g['term']:28} {g['records']} rec "
                  f"{g['users']} variants={g['variants']} nearest={g['nearest']}({g['nearest_score']}) parent={g['parent']}")
        schema = c.get(API).json()["schema"]
        known = {t: set(E_names(schema, t)) for t in schema["taxonomies"]}
        check("every suggestion is a new name", all(g["term"] not in known[g["taxonomy"]] for g in sg["groups"]))
        aar = find(sg["groups"], "topics", aar_docs)
        # the group holding the Office/PowerShell Sigma rule (the clearest detection rule)
        rule = find(sg["groups"], "doc_types", {ids["sigma"]})
        check("AAR topic gap suggested", aar is not None, aar and f"{aar['term']} ({aar['records']} records)")
        check("both AAR records grouped into one suggestion", bool(aar) and aar_docs <= {e["doc"] for e in aar["evidence"]},
              aar and f"{aar['term']} <- {', '.join([aar['term']] + aar['variants'])}")
        # soft: depends on the model's first label for each rule (gemma3:12b has called the
        # second one "yml-configuration"); the incremental re-check below can still reach it
        check("both Sigma rules grouped into one suggestion", bool(rule) and rule_docs <= {e["doc"] for e in rule["evidence"]},
              rule and f"{rule['term']} <- {', '.join([rule['term']] + rule['variants'])}", soft=True)
        check("detection-rule type gap suggested", rule is not None, rule and f"{rule['term']} ({rule['records']} records)")
        check("a suggestion is ready for review", any(g["ready"] for g in sg["groups"]))
        shot(a.shots, "1-suggestions-before", ex.name)

        print("5. review through the API")
        v = c.get(API).json()["version"]
        r = c.post(f"{API}/suggestions/approve", json={"taxonomy": "topics", "term": "x-stale", "base_version": v - 1})
        check("stale base_version rejected (409)", r.status_code == 409)
        approved = {}
        if aar:
            parent = aar["parent"] or "exercise-planning"
            r = c.post(f"{API}/suggestions/approve", json={
                "taxonomy": "topics", "term": aar["term"], "description": aar["description"], "parent": parent,
                "names": aar["variants"], "base_version": v})
            check("approve AAR topic", r.status_code == 200, f"{aar['term']} under {parent} -> v{r.json().get('version')}")
            v = r.json().get("version", v)
            approved["topics"] = aar["term"]
        if rule:
            r = c.post(f"{API}/suggestions/approve", json={
                "taxonomy": "doc_types", "term": rule["term"], "description": rule["description"],
                "names": rule["variants"], "base_version": v})
            check("approve detection-rule type", r.status_code == 200, f"{rule['term']} -> v{r.json().get('version')}")
            v = r.json().get("version", v)
            approved["doc_types"] = rule["term"]
        if aar:
            r = c.post(f"{API}/suggestions/merge", json={"taxonomy": "topics", "names": ["hotwash"],
                                                         "into": aar["term"], "base_version": v})
            v = r.json().get("version", v)
            doc = c.get(API).json()
            desc = E_leaf(doc["schema"], "topics").get(aar["term"], "")
            check("merge adds an alias the classifier sees", r.status_code == 200 and "hotwash" in desc, desc[-60:])
        hub = sg["hub_tags"].get("topics", {})
        if hub:
            tag = sorted(hub)[0]
            r = c.post(f"{API}/suggestions/reject", json={"taxonomy": "topics", "names": [tag], "base_version": v})
            v = r.json().get("version", v)
            after = c.get(f"{API}/suggestions", params={"dataset": ex.name}).json()
            check("rejected Hub tag hidden", r.status_code == 200 and tag not in after["hub_tags"].get("topics", {}), tag)
        after = c.get(f"{API}/suggestions", params={"dataset": ex.name}).json()
        check("approved suggestions leave the queue",
              not any(g["term"] in approved.values() for g in after["groups"]), f"{len(after['groups'])} left")
        m = metrics(c, ex.name)
        check("dataset flagged stale after the schema change",
              any(m["taxonomies"][t]["stale"] for t in approved) if approved else True)

        print("6. tag_docs --incremental")
        before = json.loads((ex / "tags.json").read_text(encoding="utf-8"))
        out = run([PY, "ingest/tag_docs.py", str(ex), "--incremental"], cwd=SERVER)
        print("\n".join("    " + l for l in out.strip().splitlines() if l.startswith(("[tags]", "    +", "    -", "    ")) and "  chat " not in l and "  file " not in l))
        inc, main_calls = calls(out)
        tags = json.loads((ex / "tags.json").read_text(encoding="utf-8"))
        records = [d for d in tags if not d.startswith("_")]
        full = len(records) + len(tags.get("_doc_types", {}))
        check("far fewer model calls than a full re-tag", inc + main_calls < full,
              f"{inc} re-checks + {main_calls} full asks vs {full} for a full re-tag")
        if "topics" in approved:
            got = [d for d in aar_docs if approved["topics"] in tags.get(d, [])]
            check("approved topic on the AAR records", len(got) >= 1, f"{len(got)}/{len(aar_docs)}")
        if "doc_types" in approved:
            got = [d for d in rule_docs if tags.get("_doc_types", {}).get(d) == approved["doc_types"]]
            check("approved type on the rule that proposed it", ids["sigma"] in got)
            check("re-check also finds the other Sigma rule", ids["sigma2"] in got,
                  f"{len(got)}/{len(rule_docs)} Sigma rules typed {approved['doc_types']}", soft=True)
        # precision: the source export has no after-action reports and no detection rules
        orig = [d for d in records if d not in set(ids.values())]
        if "topics" in approved:
            wrong = [d for d in orig if approved["topics"] in tags.get(d, [])]
            check("approved topic stays off unrelated records", len(wrong) <= 1,
                  f"{len(wrong)} of {len(orig)} original records got it")
        if "doc_types" in approved:
            wrong = [d for d in orig if tags.get("_doc_types", {}).get(d) == approved["doc_types"]]
            check("approved type stays off unrelated files", not wrong, f"{len(wrong)} original files retyped")
        # the one allowed change: a record that named the new subject itself gets it first and
        # loses its weakest tag when it already had the maximum
        own = {d for d in records if approved.get("topics") and tags.get(d, [None])[0] == approved["topics"]}
        lost = [(d, t) for d in records if d not in own for t in before.get(d, []) if t not in tags.get(d, [])]
        check("no unrelated tag removed", not lost, ", ".join(f"{d[:12]}:{t}" for d, t in lost)
              or f"{len(own)} record(s) now lead with the new term")
        m = metrics(c, ex.name)
        check("not stale after --incremental", not any(v_["stale"] for v_ in m["taxonomies"].values()))

        print("7. build_metadata_graph (engine container)")
        run(["wsl.exe", "-e", "bash", "-lc",
             f"docker exec {CONTAINER} python ingest/build_metadata_graph.py datasets/incoming/{ex.name} 2>&1 | tail -2"],
            env={**os.environ, "MSYS_NO_PATHCONV": "1"})
        m = metrics(c, ex.name)
        check("graph conforms", m["violations"] == 0, f"{m['nodes']} nodes, {m['edges']} edges")
        if "topics" in approved:
            check("new term counted in the graph", m["taxonomies"]["topics"]["terms"].get(approved["topics"], 0) >= 1)
        shot(a.shots, "2-suggestions-after", ex.name)
        return 0 if all(ok for _, ok in results) else 1
    finally:
        print("8. restore")
        SCHEMA.write_text(original, encoding="utf-8", newline="\n")
        if cand_before is None:
            cand_path.unlink(missing_ok=True)
        else:
            cand_path.write_bytes(cand_before)
        for p in (set(HISTORY.glob("*.yaml")) if HISTORY.is_dir() else set()) - hist_before:
            p.unlink()
        if not a.keep and ex.exists():
            shutil.rmtree(ex)
        n = sum(ok for _, ok in results)
        print(f"\n{n}/{len(results)} checks passed" + (f"; sandbox kept: {ex.name}" if a.keep and ex.exists() else ""))


def E_names(schema: dict, taxonomy: str):
    def walk(terms):
        for n, d in (terms or {}).items():
            yield n
            yield from walk((d or {}).get("children"))
    return walk(schema["taxonomies"][taxonomy]["terms"])


def E_leaf(schema: dict, taxonomy: str) -> dict:
    sys.path.insert(0, str(SERVER))
    import schema_model as SM
    return SM.leaf_terms(schema, taxonomy)


if __name__ == "__main__":
    sys.exit(main())
