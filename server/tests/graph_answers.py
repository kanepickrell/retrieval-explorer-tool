"""
Edge-case questions for factual answers (false premises, counts, multi-step, ambiguous names,
missing data), each with the facts a correct answer must contain. Runs graph_query.answer
directly (no retrieval), so it tests the lookup path: does the question become the right
lookup, and does the worded answer keep the exact facts?

  .venv/Scripts/python.exe tests/graph_answers.py [--dataset <id>] [--only N,N]

Expectations are for agenthub-export-2026-11-25-personas (re-derive them if the data changes):
  route  "graph" = should be answered by a lookup; "text" = should fall through to retrieval
  all    every one of these must appear in the answer (case-insensitive)
  any    at least one of these must appear
  none   none of these may appear
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER))
import graph_query  # noqa: E402
from retrieval import ollama_client as ollama  # noqa: E402

CASES = [
    # false premises
    {"id": "fp1", "q": "Which PDF did user-05 upload?", "route": "graph",
     "all": ["attachments.md", "gitlab-ci", "healthcheck"], "any": ["no pdf", "did not upload", "didn't upload", "none", "not upload any pdf", "no files in pdf", "not a pdf", "none of them", "no pdfs", "assumes", "has not", "any pdf"]},
    {"id": "fp2", "q": "What has the Blue Team been working on?", "route": "graph",
     "any": ["no blue team", "no team called", "not exist", "doesn't exist", "does not exist", "there is no"],
     "all": ["Red Team"]},
    {"id": "fp3", "q": "Which agent did user-02 create?", "route": "graph",
     "any": ["none", "no agent", "did not create", "didn't create", "has not created", "hasn't created", "0"]},
    {"id": "fp4", "q": "How many HTML files are there?", "route": "graph",
     "any": ["no html", "0", "zero", "none"]},
    {"id": "fp5", "q": "we have five pdf files in this data right?", "route": "graph",
     "any": ["2 pdf", "two", "only 2", "are 2", "is 2", "exactly 2"]},
    {"id": "fp6", "q": "which user is on the planning team?", "route": "graph",
     "any": ["no planning", "no team called", "not exist", "doesn't exist", "does not exist", "there is no"],
     "all": ["Scenario Development"], "none": ["any of these teams", "no users on any of", "none of these teams"]},
    # counting and comparing
    {"id": "c1", "q": "How many files did user-05 upload, and in what formats?", "route": "graph",
     "all": ["markdown", "yaml", "python"], "any": ["3", "three"]},
    {"id": "c2", "q": "Which user opened the most chats?", "route": "graph", "all": ["user-03"]},
    {"id": "c3", "q": "Which agents have skills?", "route": "graph",
     "all": ["Code Reviewer", "Editor", "Technical Writer"]},
    {"id": "c4", "q": "What's the largest uploaded file?", "route": "graph", "all": ["range_revert_runbook"]},
    {"id": "c5", "q": "How many chats are about troubleshooting?", "route": "graph", "all": ["7"]},
    # multi-step
    {"id": "m1", "q": "Which team uploaded the rules of engagement?", "route": "graph", "all": ["Red Team"],
     "none": ["user-01", "user-02", "user-04", "user-05"]},
    {"id": "m2", "q": "Which uploaded files were never attached to a chat?", "route": "graph",
     "all": ["attachments.md", "gitlab-ci"], "none": ["healthcheck", "msel"]},
    {"id": "m3", "q": "Who is on the Range Engineering team?", "route": "graph", "all": ["user-02"]},
    # ambiguous names
    {"id": "a1", "q": "Who opened the Greeting chat?", "route": "graph", "all": ["user-01", "user-04"]},
    # missing data
    {"id": "n1", "q": "What team is user-01 on?", "route": "graph",
     "any": ["no team", "not on any team", "isn't on", "is not on", "not a member", "none", "no matching", "not assigned"]},
    {"id": "n2", "q": "Which member of the Automation team built a Hub tool?", "route": "graph",
     "any": ["none", "no one", "nobody", "no member", "did not", "didn't", "built-in", "0", "zero"]},
    # mentions links (ATT&CK ids and file names found in the text)
    {"id": "x1", "q": "Which chats mention T1059.001?", "route": "graph", "all": ["Greeting Exchange"]},
    {"id": "x2", "q": "How many ATT&CK techniques are mentioned in the data?", "route": "graph", "any": ["18", "eighteen"]},
    {"id": "x3", "q": "Which of the mentioned techniques are retired?", "route": "graph",
     "all": ["T1028", "T1060", "T1075", "T1076", "T1086"], "none": ["T1059.001", "T1027"]},
    {"id": "x4", "q": "Which chat mentions the healthcheck script without attaching it?", "route": "graph",
     "all": ["Snapshot Revert Speed"]},
    # content questions: must NOT become lookups
    {"id": "t1", "q": "What's the stop word for the red team exercise?", "route": "text"},
    {"id": "t2", "q": "How do I fix Kerberos errors after a snapshot revert?", "route": "text"},
]


def grade(case: dict, g) -> tuple:
    route = "graph" if g and g.get("answer") else "text"
    ans = (g or {}).get("answer", "") or ""
    low = ans.lower()
    problems = []
    if route != case["route"]:
        problems.append(f"route {route}, expected {case['route']}" + (f" ({g.get('skipped')})" if g and g.get("skipped") else ""))
    if route == "graph" and case["route"] == "graph":
        miss = [w for w in case.get("all", []) if w.lower() not in low]
        if miss:
            problems.append(f"missing {miss}")
        if case.get("any") and not any(w.lower() in low for w in case["any"]):
            problems.append(f"none of {case['any'][:3]}…")
        bad = [w for w in case.get("none", []) if w.lower() in low]
        if bad:
            problems.append(f"should not mention {bad}")
    return route, ans, problems


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agenthub-export-2026-11-25-personas~metadata")
    ap.add_argument("--only")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    ok = 0
    rows = []
    for case in CASES:
        if only and case["id"] not in only:
            continue
        g = await graph_query.answer(case["q"], a.dataset, lambda p, s: ollama.llm(p, system_prompt=s))
        route, ans, problems = grade(case, g)
        ok += not problems
        print(f"{'PASS' if not problems else 'FAIL'} {case['id']:4} {case['q']}")
        if g and g.get("query"):
            print(f"       lookup: {g['query']}")
        if ans:
            print("       answer: " + " ".join(ans.split())[:300])
        for p in problems:
            print(f"       !! {p}")
        rows.append({"id": case["id"], "q": case["q"], "route": route, "problems": problems,
                     "lookup": (g or {}).get("query"), "answer": ans})
    n = len(rows)
    print(f"\n{ok}/{n} passed")
    (Path(__file__).with_suffix(".last.json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    await ollama.aclose()
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
