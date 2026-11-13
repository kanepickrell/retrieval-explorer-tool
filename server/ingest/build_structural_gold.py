"""
Structural gold questions: questions about how the records relate (whose chats,
which files, who used an agent, what was attached where), with answers computed
exactly from the metadata graph rather than written by an LLM.

The existing gold sets only ask "find the doc that says X", which chunk search is
built for. These test what the graph layer is for. Gold ids are source doc files,
so every index of the export (chunks, structured, metadata, agent) is scored on
the same questions.

Reads <export>/storage-metadata/graph_chunk_entity_relation.graphml (run
build_metadata_graph.py first) and writes <export>/eval_set_structural.json;
datasets.gold_entries() picks it up next to eval_set.json. Team questions are
added only when <export>/teams.json exists.

Usage:
  python build_structural_gold.py <export_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

MAX_GOLD = 8          # keep answer sets small enough to judge at k=10
PER_TYPE = 6          # questions per template
GENERIC_TAGS = {"general", "technology", "programming", "software development", "science", "education"}


def pick(variants: list, key: str) -> str:
    return variants[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(variants)]


def main(export_dir: str) -> None:
    ex = Path(export_dir)
    G = nx.read_graphml(ex / "storage-metadata" / "graph_chunk_entity_relation.graphml")
    status = json.loads((ex / "storage-metadata" / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    indexed = {v["file_path"] for v in status.values() if v.get("status") == "processed"}
    T = lambda n: G.nodes[n].get("entity_type")
    doc = lambda n: G.nodes[n].get("file_path") or ""
    of = lambda n, t: sorted(m for m in G.neighbors(n) if T(m) == t)
    title = lambda n: n.split(": ", 1)[-1].rsplit(" [", 1)[0]
    out, counts = [], defaultdict(int)

    def add(qtype: str, query: str, nodes: list, reference: str):
        gold = sorted({doc(n) for n in nodes if doc(n) in indexed})
        if not gold or len(gold) > MAX_GOLD or counts[qtype] >= PER_TYPE:
            return
        counts[qtype] += 1
        out.append({"id": f"s-{qtype}-{hashlib.md5(query.encode()).hexdigest()[:6]}", "query": query,
                    "gold_ids": gold, "reference": reference, "query_type": f"structural_{qtype}",
                    "generated_by": "build_structural_gold.py (exact, from the metadata graph)"})

    users = sorted(n for n in G if T(n) == "User")
    for u in users:
        chats = of(u, "Chat")
        if 2 <= len(chats) <= MAX_GOLD:
            add("user_activity", pick([f"What has {u} been working on?", f"Show me {u}'s recent chats.",
                                       f"What kinds of things has {u} asked about?"], u),
                chats, f"{u} has {len(chats)} chats: " + "; ".join(title(c) for c in chats))
        files = of(u, "File")
        if files:
            add("user_files", pick([f"Which files has {u} uploaded?", f"What did {u} upload?"], u),
                files, f"{u} uploaded: " + "; ".join(title(f) for f in files))
    for u in users[::3]:
        chats = of(u, "Chat")
        if chats and f"person-{u}.md" in indexed:
            add("user_count", f"How many chats has {u} had?", [u], f"{u} has {len(chats)} indexed chats.")

    names = defaultdict(list)
    for f in (n for n in G if T(n) == "File"):
        names[title(f)].append(f)
    asked = set()
    for name, fs in sorted(names.items()):
        owners = sorted({u for f in fs for u in of(f, "User")})
        if len(fs) == 1 and len(owners) == 1 and owners[0] not in asked:   # one question per owner
            asked.add(owners[0])
            add("file_owner", pick([f"Who uploaded {name}?", f"Whose file is {name}?"], name),
                owners, f"{name} was uploaded by {owners[0]}.")

    for a in sorted(n for n in G if T(n) == "Agent"):
        chats = of(a, "Chat")
        if chats:
            add("agent_usage", pick([f"Which chats used the {a} agent?", f"Where has {a} been used?"], a),
                chats, f"{a} was used in: " + "; ".join(title(c) for c in chats))
        people = of(a, "User") + [u for c in chats for u in of(c, "User")]
        if people:
            ppl = sorted(set(people))
            add("agent_users", pick([f"Who has used {a}?", f"Which users have run {a}?"], a),
                ppl, f"{a} has been used or installed by: " + ", ".join(ppl))
        skills = of(a, "Skill")
        if skills:
            add("agent_skills", f"What skills does the {a} agent have?", [a],
                f"{a} has: " + ", ".join(title(s) for s in skills))
    with_skills = sorted(a for a in G if T(a) == "Agent" and of(a, "Skill"))
    if with_skills:
        add("agents_with_skills", "Which agents come with skills attached?", with_skills,
            "Agents with skills: " + ", ".join(with_skills))

    for c in sorted(n for n in G if T(n) == "Chat"):
        files = of(c, "File")
        if len(files) >= 2:
            add("chat_files", f"What files were attached in the \"{title(c)}\" chat?", files,
                "; ".join(title(f) for f in files))

    for t in sorted((n for n in G if T(n) == "Topic"), key=lambda n: -G.degree(n)):
        tag = t[len("Topic: "):]
        chats = of(t, "Chat")
        if tag not in GENERIC_TAGS and 2 <= len(chats) <= 6:
            add("topic_chats", pick([f"Find the chats about {tag}.", f"Which conversations dealt with {tag}?"], tag),
                chats, f"Chats tagged {tag}: " + "; ".join(title(c) for c in chats))

    for kind, plural in (("Step", "workflow steps"), ("Tool", "tools")):
        items = sorted(n for n in G if T(n) == kind)
        cat = f"Catalog: {plural.split()[-1]}"
        if items:
            add("catalog", f"Which {plural} are available in Agent Hub?", items + ([cat] if cat in G else []),
                ", ".join(items))

    for team in sorted(n for n in G if T(n) == "Team"):
        members = of(team, "User")
        chats = [c for u in members for c in of(u, "Chat")]
        name = team[len("Team: "):]
        add("team_members", f"Who is on the {name} team?", members, ", ".join(members))
        add("team_activity", f"What has the {name} team been working on?", chats,
            "; ".join(title(c) for c in chats))

    f = ex / "eval_set_structural.json"
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} structural questions -> {f}")
    print(dict(counts))


if __name__ == "__main__":
    main(sys.argv[1])
