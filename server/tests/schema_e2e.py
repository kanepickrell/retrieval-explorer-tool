"""
End-to-end check of the semantic model: a schema edit made through the API reaches the
LLM tagger, new data is classified with it, and the Prospector metrics follow along.

  1. copy an export to a sandbox (<export>-schematest; storage/ is not copied)
  2. add new records: an after-action report file, a Sigma rule file, and a chat about
     writing the hotwash report that contains the AAR file
  3. export_to_docs   -> the new records are counted but show as unclassified
  4. PUT /api/schema  -> adds topic "after-action-review" + doc type "after-action-report";
                         version bumps, a history backup appears, a stale base_version
                         gets 409, and every dataset is flagged stale
  5. tag_docs         -> the new records get the new terms
  6. build_metadata_graph (in the engine container) -> graph conforms, counts grew,
                         and a new ingest snapshot is recorded
  7. restore schema.yaml and remove the test's history backups (always, even on failure)

Runs on the host with the server venv; the engine must be up on :5180.
Calls the LLM host (~30 short calls) and the embed host once.

  .venv/Scripts/python.exe tests/schema_e2e.py [--source <export>] [--keep]

--keep leaves the sandbox export in place so it can be inspected in Prospector
(otherwise it is deleted at the end).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

SERVER = Path(__file__).resolve().parent.parent
INCOMING = SERVER / "datasets" / "incoming"
SCHEMA = SERVER / "datasets" / "schema" / "schema.yaml"
HISTORY = SCHEMA.parent / "history"
API = "http://localhost:5180/api/schema"
PY = sys.executable
CONTAINER = "oui-agents-retrieval-explorer-1"

AAR = """# After-Action Report: Exercise BROKEN LEDGER (Day 1-2)

Prepared by the white cell for the exercise director.

## Summary
The blue team detected the initial phishing foothold on WS-FIN-02 within 40 minutes
but did not identify the C2 domain until inject 6. Containment of FS-01 was late.

## What went well
- EDR alert triage started within 10 minutes of inject 1.
- The helpdesk interview (inject 2) found the INV-20931.docm email quickly.

## What needs improvement
- Mail log review was not in the team's playbook; the contingency hint was needed.
- No one owned DNS blocking, so the C2 domain stayed reachable for 3 hours.
- The leadership summary (inject 5) was late and lacked impact estimates.

## Recommendations
1. Add a mail-log triage step to the blue team runbook.
2. Assign a DNS/egress owner in the team roster before STARTEX.
3. Give the team a leadership-summary template.

## Lessons learned for the next exercise
Shorten the gap between injects 4 and 6, and brief the stop words again at STARTEX.
"""

SIGMA = """title: Encoded PowerShell launched by Office
id: 5b1f0c2e-ledger-0001
status: experimental
description: Office application spawning powershell.exe with -EncodedCommand (BROKEN LEDGER inject 1)
logsource:
  product: windows
  category: process_creation
detection:
  selection_parent:
    ParentImage|endswith: ['\\\\winword.exe', '\\\\excel.exe']
  selection_child:
    Image|endswith: '\\\\powershell.exe'
    CommandLine|contains: ['-enc', '-EncodedCommand']
  condition: selection_parent and selection_child
level: high
tags: [attack.execution, attack.t1059.001]
"""

CHAT_TURNS = [
    ("user", "Exercise BROKEN LEDGER wrapped yesterday. I attached our after-action report draft. "
             "Can you help me turn the hotwash notes into a cleaner AAR with findings, "
             "recommendations and owners for each lesson learned?"),
    ("assistant", "Sure. A common AAR structure is: summary, timeline of key events, what went well, "
                  "what needs improvement, root causes, and recommendations with an owner and due date. "
                  "From your draft, the strongest findings are the late C2 domain identification and "
                  "the missing DNS owner. I'd turn each into a recommendation with an owner."),
    ("user", "Good. Also add a section comparing the planned MSEL timing with what actually happened, "
             "so the exercise planners can tune the inject spacing for next time."),
    ("assistant", "Added a 'Planned vs actual' table: inject 4 was planned at D1 14:00 and answered at "
                  "D1 17:10; inject 6 at D2 10:00 was contained at D2 12:40. The recommendation for "
                  "planners is to move inject 6 earlier or add a hint after inject 4."),
]

TOPIC = "after-action-review"
DTYPE = "after-action-report"

results: list = []


def check(name: str, ok: bool, detail: str = "", soft: bool = False) -> bool:
    """soft: a model-quality observation worth seeing (WARN) that doesn't fail the run."""
    results.append((name, ok or soft))
    print(f"  {'PASS' if ok else 'WARN' if soft else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run(cmd: list, **kw) -> str:
    print("  $", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode:
        print(out[-3000:])
        raise SystemExit(f"command failed ({p.returncode})")
    return out


def metrics(c: httpx.Client, ds: str) -> dict:
    r = c.get(f"{API}/metrics", params={"dataset": ds})
    r.raise_for_status()
    return r.json()


def rw(p: Path, fn):
    data = json.loads(p.read_text(encoding="utf-8"))
    data = fn(data)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def add_records(ex: Path) -> dict:
    """New file + chat rows in the export tables, shaped like export_local.py's output."""
    now = int(time.time())
    aar_id, sigma_id = "e2e00000-0000-4000-8000-0000000aa001", "e2e00000-0000-4000-8000-0000000aa002"
    chat_id = "e2e00000-0000-4000-8000-0000000cc001"

    def file_row(fid, user, name, ctype, text):
        return {"id": fid, "user_id": user, "filename": name, "hash": fid, "created_at": now,
                "meta": {"name": name, "content_type": ctype, "size": len(text), "data": {}},
                "data": {"status": "completed", "content": text}}

    rw(ex / "openwebui/file.json", lambda rows: rows + [
        file_row(aar_id, "user-03", "broken_ledger_aar.md", "text/markdown", AAR),
        file_row(sigma_id, "user-04", "office_encoded_powershell.yml", "application/x-yaml", SIGMA)])

    msgs, parent = [], None
    for k, (role, text) in enumerate(CHAT_TURNS):
        mid = f"m{k:03d}"
        msgs.append({"id": f"{chat_id}-{mid}", "chat_id": chat_id, "user_id": "user-03", "role": role,
                     "parent_id": parent, "content": text, "done": 1,
                     "model_id": "gpt-oss:20b" if role == "assistant" else None,
                     "created_at": now + k, "updated_at": now + k})
        parent = mid
    rw(ex / "openwebui/chat_message.json", lambda rows: rows + msgs)
    rw(ex / "openwebui/chat.json", lambda rows: rows + [{
        "id": chat_id, "user_id": "user-03", "title": "Hotwash to AAR", "created_at": now, "updated_at": now,
        "archived": 0, "pinned": 0, "meta": {"tags": ["exercise", "reporting"]}, "current_message_id": parent}])
    rw(ex / "openwebui/chat_file.json", lambda rows: rows + [{
        "id": "e2e00000-0000-4000-8000-0000000cf001", "user_id": "user-03", "chat_id": chat_id,
        "file_id": aar_id, "message_id": "m000", "created_at": now, "updated_at": now}])
    return {"aar": f"file-{aar_id}.md", "sigma": f"file-{sigma_id}.md", "chat": f"chat-{chat_id}.md"}


def edited_schema(schema: dict) -> dict:
    s = json.loads(json.dumps(schema))
    s["taxonomies"]["topics"]["terms"]["exercise-planning"]["children"][TOPIC] = {
        "description": "after-action reports, hotwash, lessons learned, findings and recommendations after an exercise"}
    terms = s["taxonomies"]["doc_types"]["terms"]
    other = terms.pop("other")
    terms[DTYPE] = {"description": "after-action report or hotwash write-up: findings, lessons learned, recommendations"}
    terms["other"] = other
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="agenthub-export-2026-11-25-personas")
    ap.add_argument("--keep", action="store_true", help="leave the sandbox export for inspection")
    a = ap.parse_args()

    src, ex = INCOMING / a.source, INCOMING / f"{a.source}-schematest"
    original = SCHEMA.read_text(encoding="utf-8")
    cand_path = SCHEMA.parent / "candidates.json"     # tag_docs.py's suggestion grouping
    cand_before = cand_path.read_bytes() if cand_path.exists() else None
    hist_before = set(HISTORY.glob("*.yaml")) if HISTORY.is_dir() else set()
    c = httpx.Client(timeout=120)
    try:
        print("1. sandbox")
        if ex.exists():
            shutil.rmtree(ex)
        shutil.copytree(src, ex, ignore=shutil.ignore_patterns("storage", "schema_metrics.json"))
        m0 = metrics(c, ex.name)
        check("sandbox measured like its source", m0["violations"] == 0 and not m0["taxonomies"]["topics"]["stale"],
              f"{m0['nodes']} nodes, topics {m0['taxonomies']['topics']['assigned']}/{m0['taxonomies']['topics']['records']}")

        print("2-3. new records + export_to_docs")
        ids = add_records(ex)
        run([PY, "ingest/export_to_docs.py", str(ex), str(ex / "docs")], cwd=SERVER)
        check("new docs written", all((ex / "docs" / d).exists() for d in ids.values()))
        m1 = metrics(c, ex.name)
        t1 = m1["taxonomies"]["topics"]
        check("new records counted, not yet classified",
              t1["records"] == m0["taxonomies"]["topics"]["records"] + 3 and t1["assigned"] == t1["records"] - 3,
              f"topics {t1['assigned']}/{t1['records']}, doc_types "
              f"{m1['taxonomies']['doc_types']['assigned']}/{m1['taxonomies']['doc_types']['records']}")

        print("4. schema edit through the API")
        g = c.get(API).json()
        v = g["version"]
        new = edited_schema(g["schema"])
        bad = c.put(API, json={"schema": {**new, "relations": {**new["relations"], "x": {"from": ["Nope"], "to": ["Topic"]}}},
                               "base_version": v})
        check("invalid schema rejected (400)", bad.status_code == 400, bad.json().get("detail", "")[:80])
        r = c.put(API, json={"schema": new, "base_version": v})
        check("valid edit saved", r.status_code == 200 and r.json()["version"] == v + 1, f"v{v} -> v{r.json().get('version')}")
        stale_put = c.put(API, json={"schema": new, "base_version": v})
        check("stale base_version rejected (409)", stale_put.status_code == 409)
        new_hist = (set(HISTORY.glob("*.yaml")) if HISTORY.is_dir() else set()) - hist_before
        check("previous version backed up", len(new_hist) == 1, ", ".join(p.name for p in new_hist))
        m2 = metrics(c, ex.name)
        check("sandbox flagged stale (topics + doc_types)",
              m2["taxonomies"]["topics"]["stale"] and m2["taxonomies"]["doc_types"]["stale"])
        check("new terms listed with 0 uses", m2["taxonomies"]["topics"]["terms"].get(TOPIC) == 0
              and m2["taxonomies"]["doc_types"]["terms"].get(DTYPE) == 0)
        ms = metrics(c, a.source)
        check("other datasets flagged stale too", ms["taxonomies"]["topics"]["stale"])

        print("5. tag_docs with the new vocabulary")
        out = run([PY, "ingest/tag_docs.py", str(ex)], cwd=SERVER)
        print("\n".join("    " + l for l in out.strip().splitlines()[-18:]))
        tags = json.loads((ex / "tags.json").read_text(encoding="utf-8"))
        dt = tags.get("_doc_types", {})
        check(f"AAR file typed {DTYPE}", dt.get(ids["aar"]) == DTYPE, f"got {dt.get(ids['aar'])}")
        check(f"AAR file tagged {TOPIC}", TOPIC in tags.get(ids["aar"], []), ", ".join(tags.get(ids["aar"], [])))
        check(f"AAR chat tagged {TOPIC}", TOPIC in tags.get(ids["chat"], []), ", ".join(tags.get(ids["chat"], [])))
        check("Sigma rule tagged detection", "detection" in tags.get(ids["sigma"], []),
              f"{', '.join(tags.get(ids['sigma'], []))} [{dt.get(ids['sigma'])}]")
        check("no off-vocabulary tags", all(t in tags["_vocabulary"] for k, v in tags.items()
                                           if not k.startswith("_") for t in v))
        m3 = metrics(c, ex.name)
        check("not stale after re-tagging", not m3["taxonomies"]["topics"]["stale"]
              and not m3["taxonomies"]["doc_types"]["stale"])

        print("6. build_metadata_graph (engine container)")
        rel = f"datasets/incoming/{ex.name}"
        run(["wsl.exe", "-e", "bash", "-lc", f"docker exec {CONTAINER} python ingest/build_metadata_graph.py {rel} 2>&1 | tail -3"],
            env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"})
        m4 = metrics(c, ex.name)
        cl, rl = m4["classes"], m4["relations"]
        check("graph conforms (0 violations)", m4["violations"] == 0, f"{m4['nodes']} nodes, {m4['edges']} edges")
        check("Chat +1, File +2", cl["Chat"]["count"] == m0["classes"]["Chat"]["count"] + 1
              and cl["File"]["count"] == m0["classes"]["File"]["count"] + 2,
              f"Chat {cl['Chat']['count']}, File {cl['File']['count']}")
        check("contains +1", rl["contains"]["count"] == m0["relations"]["contains"]["count"] + 1)
        check(f"graph has the {TOPIC} topic node", m4["taxonomies"]["topics"]["terms"].get(TOPIC, 0) >= 2,
              f"{m4['taxonomies']['topics']['terms'].get(TOPIC)} records")
        check(f"{DTYPE} counted", m4["taxonomies"]["doc_types"]["terms"].get(DTYPE, 0) >= 1)
        cards = (ex / "storage-metadata/kv_store_full_docs.json").read_text(encoding="utf-8")
        check("AAR card says its document type",        # cards spell terms with spaces
              f"Document type: {DTYPE.replace('-', ' ')}" in cards)
        hist = m4["history"]
        check("ingest snapshots recorded", len(hist) >= 2, f"{len(hist)} snapshots")
        return 0 if all(ok for _, ok in results) else 1
    finally:
        print("7. restore")
        SCHEMA.write_text(original, encoding="utf-8", newline="\n")
        if cand_before is None:
            cand_path.unlink(missing_ok=True)
        else:
            cand_path.write_bytes(cand_before)
        for p in (set(HISTORY.glob("*.yaml")) if HISTORY.is_dir() else set()) - hist_before:
            p.unlink()
        if not a.keep and ex.exists():
            shutil.rmtree(ex)
        try:
            ms = metrics(c, a.source)
            check("schema restored, source dataset not stale", not ms["taxonomies"]["topics"]["stale"]
                  and not ms["taxonomies"]["doc_types"]["stale"], f"schema v{ms['schema_version']}")
        except Exception as e:   # noqa: BLE001
            check("schema restored", False, str(e))
        n = sum(ok for _, ok in results)
        print(f"\n{n}/{len(results)} checks passed" + (f"; sandbox kept: {ex.name}" if a.keep and ex.exists() else ""))


if __name__ == "__main__":
    sys.exit(main())
