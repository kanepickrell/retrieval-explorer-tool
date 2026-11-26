"""
Metadata graph: a routing layer over the export instead of a graph of every
entity mentioned in the text.

Every node is a metadata record built straight from the export tables (no LLM
extraction). Source docs (chats, files, agents, steps, tools, people, catalogs)
each get one "card": a short text holding the record's metadata, its summary
and the identifiers found in its full text (TIDs, file names, CVEs). Retrieval
searches the cards; a hit links to the full source doc (docs/<id>.md), which an
agent or the user then reads with full context.

  nodes   User, Team, Chat, File, Agent, Skill, Step, Tool, Model, Topic, Catalog
  edges   member_of   user -> team         (teams.json, optional)
          opened      user -> chat         (chat.user_id)
          contains    chat -> file         (chat_file)
          uploaded    user -> file         (file.user_id)
          created     user -> agent/step/tool (owner_id; built-ins have owner "system")
          installed   user -> agent/step/tool (hub_install)
          used        chat -> agent/model  (assistant messages' model_id)
          has_skill   agent -> skill       (manifest skills)
          uses        agent -> tool/step   (manifest tools)
          forked_from item -> item
          (files also carry doc_type, tags.json's _doc_types, on the node and card)
          about       chat/file -> topic (tags.json: tag_docs.py's fixed vocabulary;
                      without it, Open WebUI's own chat tags), agent/step/tool -> topic
          lists       catalog -> item (agents, tools, steps, files, teams)
  "produced" (chat/agent -> artifact) has no source table in the export yet.

Node properties (the semantic model's classes[...].properties, measured in Prospector):
  User     team, chats, files            Chat   title, started, last_active, user_turns
  Team     name, members                 File   filename, format, content_type, size_bytes,
  Agent/Step/Tool  name, slug, created_on,      uploaded_on, doc_type
           category, version             Model  model_id    Skill / Topic  name
  Catalog  kind, count
Dates are YYYY-MM-DD strings; counts and sizes are integers. A property is only written
when the export has a value for it, so coverage shows what the data really holds.

Output is a LightRAG-shaped storage dir (graphml, vdb_chunks, kv stores), so
datasets.py registers it as <export>~metadata and the vector / bm25 /
bm25_vector modes and the Prospector work on it unchanged. Card ids are the
source doc file names, so the export's gold set scores it as-is. LightRAG's
own graph modes (local/global/...) do not apply: there is no entity vdb.

Teams: <export>/teams.json, {"user-NN": "Team"}. Pseudonyms only, never real
names (the export is pseudonymized and the repo is public). Users without an
entry get no team.

Usage:
  python build_metadata_graph.py <export_dir> [--out storage-metadata]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_to_docs import day, file_format, load, loose, msg_text, slug  # noqa: E402

EMBED_HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
EMBED_MODEL = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")

TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
# ATT&CK technique ids -> name / retired, extracted from attack-mcp's enterprise-attack bundle
ATTACK = Path(os.getenv("PROTOGRAPH_ATTACK_TECHNIQUES",
                        Path(__file__).resolve().parent.parent / "datasets" / "reference" / "attack-techniques.json"))
CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
FILE = re.compile(r"\b[\w-]+\.(?:py|ya?ml|json|ps1|psm1|sh|bat|exe|dll|csv|txt|md|xml|html|js|ts|go|rs|c|cpp|java|sql|conf|ini|log)\b")
MAX_IDS = 30          # identifiers kept per card
REQ_CHARS = 400       # opening user request kept on a chat card
PROMPT_CHARS = 400    # system prompt excerpt kept on an agent card
TEXT_CHARS = 300      # opening text kept on a file card


def identifiers(text: str) -> list:
    """TIDs, CVEs and file names in the full text, most frequent first."""
    c = Counter()
    for rx in (TID, CVE, FILE):
        c.update(m.group(0) for m in rx.finditer(text or ""))
    return [k for k, _ in c.most_common(MAX_IDS)]


def one_line(s, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


class Builder:
    def __init__(self):
        self.G = nx.Graph()
        self.cards: dict = {}            # doc file -> card text
        self.card_nodes = defaultdict(set)   # doc file -> node ids the card is about

    def node(self, nid: str, etype: str, desc: str = "", doc: str = "") -> str:
        if nid not in self.G:
            self.G.add_node(nid, entity_id=nid, entity_type=etype, description=desc,
                            source_id="", file_path=doc)
        elif desc and not self.G.nodes[nid].get("description"):
            self.G.nodes[nid]["description"] = desc
        if doc:
            self.G.nodes[nid]["file_path"] = doc
            self.card_nodes[doc].add(nid)
        return nid

    def prop(self, nid: str, **kv):
        """Set structured properties on a node, skipping empty values (graphml has no null)."""
        for k, v in kv.items():
            if v is None or v == "" or nid not in self.G:
                continue
            self.G.nodes[nid][k] = v if isinstance(v, (int, float, bool)) else str(v)

    def edge(self, u: str, v: str, rel: str, desc: str = ""):
        if u == v or u not in self.G or v not in self.G:
            return
        if self.G.has_edge(u, v):
            kw = self.G.edges[u, v]["keywords"]
            if rel not in kw.split(","):
                self.G.edges[u, v]["keywords"] = kw + "," + rel
            self.G.edges[u, v]["weight"] += 1.0
        else:
            self.G.add_edge(u, v, keywords=rel, description=desc or rel, weight=1.0, source_id="", file_path="")


def build(export_dir: Path, out_dir: Path) -> None:
    ex, docs = export_dir, export_dir / "docs"
    doc_files = {p.name for p in docs.glob("*.md")}
    teams = {}
    if (ex / "teams.json").exists():
        teams = {k: v for k, v in load(ex / "teams.json").items() if re.fullmatch(r"user-\d+", k)}
    # tag_docs.py output: fixed-vocabulary tags per doc. When present it replaces Open
    # WebUI's free-form chat tags as the Topic source (those stay on the card as text).
    vocab_tags, doc_types = {}, {}
    if (ex / "tags.json").exists():
        tj = load(ex / "tags.json")
        vocab_tags = {k: v for k, v in tj.items() if not k.startswith("_")}
        doc_types = tj.get("_doc_types") or {}   # taxonomies.doc_types in the semantic model
    B = Builder()

    def user(u: str) -> str:
        if not u or u == "system":
            return ""
        doc = f"person-{u}.md" if f"person-{u}.md" in doc_files else ""
        nid = B.node(u, "User", f"Pseudonymous Agent Hub user {u}", doc)
        if teams.get(u):
            t = B.node(f"Team: {teams[u]}", "Team", f"The {teams[u]} team")
            B.prop(t, name=teams[u])
            B.prop(nid, team=teams[u])
            B.edge(nid, t, "member_of")
        return nid

    def topic(tag: str) -> str:
        tag = str(tag).strip().replace("_", " ").lower()
        if not tag:
            return ""
        nid = B.node(f"Topic: {tag}", "Topic", f"Items tagged '{tag}'")
        B.prop(nid, name=tag)
        return nid

    # ── hub items: agents / steps / tools, latest version manifest ──
    def latest(rows, key):
        best = {}
        for r in rows:
            if r[key] not in best or r["version"] > best[r[key]]["version"]:
                best[r[key]] = r
        return {k: v for k, v in best.items()}

    item_rows = {"agent": {a["id"]: a for a in load(ex / "hub/agent.json")},
                 "step": {s["id"]: s for s in load(ex / "hub/step.json")},
                 "tool": {t["id"]: t for t in load(ex / "hub/hub_tool.json")}}
    item_node = {}       # (kind, slug) -> node id
    hub_cards = {}       # doc -> (lines, kind, slug, row, manifest)
    for kind, path, key in (("agent", "hub/agent_version.json", "agent_id"),
                            ("step", "hub/step_version.json", "step_id"),
                            ("tool", "hub/hub_tool_version.json", "tool_id")):
        for item_id, v in latest(load(ex / path), key).items():
            row = item_rows[kind].get(item_id, {})
            if row.get("deleted_at"):
                continue
            man = loose(v.get("manifest")) or {}
            name = man.get("name") or row.get("name") or man.get("slug") or item_id
            s = man.get("slug") or row.get("slug") or slug(name)
            doc = f"{kind}-{slug(s)}.md"
            nid = B.node(name, kind.title(), one_line(man.get("description") or row.get("description"), 300),
                         doc if doc in doc_files else "")
            item_node[(kind, s)] = nid
            hub_cards[doc] = (kind, s, row, man, v)
            created = day(row.get("created_at"))
            B.prop(nid, name=name, slug=s, created_on=created if created != "unknown date" else "",
                   category=row.get("category"), version=int(v["version"]) if str(v.get("version") or "").isdigit() else None)

    for doc, (kind, s, row, man, v) in hub_cards.items():
        nid = item_node[(kind, s)]
        owner = row.get("owner_id") or "system"
        if user(owner):
            B.edge(owner, nid, "created")
        for tag in man.get("tags") or row.get("tags") or []:
            B.edge(nid, topic(tag), "about")
        if row.get("forked_from_slug"):
            parent = item_node.get((kind, row["forked_from_slug"]))
            if parent:
                B.edge(nid, parent, "forked_from")
        skills = [sk for sk in (man.get("skills") or []) if isinstance(sk, dict) and sk.get("name")]
        for sk in skills:
            sid = B.node(f"Skill: {sk['name']}", "Skill", one_line(sk.get("description"), 300))
            B.prop(sid, name=sk["name"])
            B.edge(nid, sid, "has_skill")
        tools = []
        for t in man.get("tools") or []:
            tslug = t.get("slug") or t.get("name") if isinstance(t, dict) else str(t)
            tools.append(tslug)
            target = item_node.get(("tool", tslug)) or item_node.get(("step", tslug))
            if target:
                B.edge(nid, target, "uses")
        if doc not in doc_files:
            continue
        lines = [f"# {B.G.nodes[nid]['entity_id']} ({kind})", "",
                 f"Type: Agent Hub {kind}. Created {day(row.get('created_at'))} by "
                 f"{'built-in (system)' if owner == 'system' else owner}"
                 + (f" ({teams[owner]} team)" if teams.get(owner) else "") + "."]
        if man.get("description") or row.get("description"):
            lines.append(f"Purpose: {one_line(man.get('description') or row.get('description'), 400)}")
        tags = man.get("tags") or row.get("tags") or []
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        if row.get("category"):
            lines.append(f"Category: {row['category']}")
        if man.get("manual_time_minutes") or row.get("manual_time_minutes"):
            lines.append(f"Automates a process that takes about {man.get('manual_time_minutes') or row.get('manual_time_minutes')} minutes by hand.")
        if skills:
            lines.append("Skills: " + "; ".join(f"{sk['name']} ({one_line(sk.get('description'), 120)})" for sk in skills))
        if tools:
            lines.append("Tools: " + ", ".join(tools))
        if man.get("base_model_id"):
            lines.append(f"Base model: {man['base_model_id']}")
        if row.get("forked_from_slug"):
            lines.append(f"Forked from: {row['forked_from_slug']}")
        if man.get("system"):
            lines.append(f"System prompt (opening): {one_line(man['system'], PROMPT_CHARS)}")
        full = (docs / doc).read_text(encoding="utf-8")
        ids = identifiers(full)
        if ids:
            lines.append("Identifiers: " + ", ".join(ids))
        B.cards[doc] = "\n".join(lines)

    for i in load(ex / "hub/hub_install.json"):
        target = item_node.get((i.get("kind"), i.get("item_slug")))
        if target and user(i.get("user_id")):
            B.edge(i["user_id"], target, "installed")

    # ── files ──
    files = {f["id"]: f for f in load(ex / "openwebui/file.json")}
    file_chats = defaultdict(set)
    for cf in load(ex / "openwebui/chat_file.json"):
        file_chats[cf["file_id"]].add(cf["chat_id"])

    # ── chats ──
    msgs = defaultdict(list)
    for m in load(ex / "openwebui/chat_message.json"):
        msgs[m["chat_id"]].append(m)
    reviews = {}
    for r in load(ex / "openwebui/chat_review.json"):
        if r["chat_id"] not in reviews or str(r.get("summary_at") or "") > str(reviews[r["chat_id"]].get("summary_at") or ""):
            reviews[r["chat_id"]] = r
    chats = {c["id"]: c for c in load(ex / "openwebui/chat.json")}
    agent_slugs = {s: n for (k, s), n in item_node.items() if k == "agent"}
    chat_node = {}
    for cid, c in chats.items():
        doc = f"chat-{cid}.md"
        if doc not in doc_files:
            continue      # small talk, skipped by the converter
        title = c.get("title") or "Chat"
        nid = B.node(f"Chat: {title} [{cid[:8]}]", "Chat", "", doc)
        chat_node[cid] = nid
    for cid, nid in chat_node.items():
        c, doc = chats[cid], f"chat-{cid}.md"
        turns = sorted(msgs.get(cid, []), key=lambda m: int(m.get("created_at") or 0))
        u = c.get("user_id") or ""
        if user(u):
            B.edge(u, nid, "opened")
        meta = loose(c.get("meta")) or {}
        auto_tags = meta.get("tags", []) if isinstance(meta, dict) else []
        tags = vocab_tags.get(doc, []) if vocab_tags else auto_tags
        for tag in tags:
            B.edge(nid, topic(tag), "about")
        used = Counter(m.get("model_id") for m in turns if m.get("role") == "assistant" and m.get("model_id"))
        agents_used, models_used = [], []
        for mid in used:
            if mid in agent_slugs:
                B.edge(nid, agent_slugs[mid], "used")
                agents_used.append(B.G.nodes[agent_slugs[mid]]["entity_id"])
                if u:
                    B.edge(u, agent_slugs[mid], "installed")
            else:
                mn = B.node(f"Model: {mid}", "Model", f"Chat model {mid}")
                B.prop(mn, model_id=mid)
                B.edge(nid, mn, "used")
                models_used.append(mid)
        attached = []
        for fid, cids in file_chats.items():
            if cid in cids and fid in files:
                fdoc = f"file-{fid}.md"
                fname = files[fid].get("filename") or fid
                fn = B.node(f"File: {fname} [{fid[:8]}]", "File", "", fdoc if fdoc in doc_files else "")
                B.edge(nid, fn, "contains")
                attached.append(fname)
        started, last = day(c.get("created_at")), day(c.get("updated_at"))
        B.prop(nid, title=c.get("title") or "Chat", started=started if started != "unknown date" else "",
               last_active=last if last != "unknown date" else "",
               user_turns=sum(1 for m in turns if m.get("role") == "user"))
        rv = reviews.get(cid) or {}
        first_req = next((msg_text(m) for m in turns if m.get("role") == "user" and msg_text(m).strip()), "")
        full = (docs / doc).read_text(encoding="utf-8")
        lines = [f"# {c.get('title') or 'Chat'}", "",
                 f"Type: chat session. Started {day(c.get('created_at'))}, last active {day(c.get('updated_at'))}, "
                 f"{sum(1 for m in turns if m.get('role') == 'user')} user turns. Opened by {u or 'unknown'}"
                 + (f" ({teams[u]} team)" if teams.get(u) else "") + "."]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        if vocab_tags and auto_tags:
            lines.append(f"User tags: {', '.join(auto_tags)}")
        if rv.get("summary"):
            lines.append(f"Summary: {one_line(rv['summary'], 700)}")
        pains = loose(rv.get("pain_points")) or []
        if pains:
            lines.append("Pain points: " + "; ".join(one_line(p if isinstance(p, str) else json.dumps(p), 150) for p in pains[:5]))
        if first_req:
            lines.append(f"Opening request: {one_line(first_req, REQ_CHARS)}")
        if agents_used:
            lines.append(f"Agents used: {', '.join(agents_used)}")
        if models_used:
            lines.append(f"Models: {', '.join(models_used)}")
        if attached:
            lines.append(f"Attached files: {', '.join(attached)}")
        ids = identifiers(full)
        if ids:
            lines.append("Identifiers: " + ", ".join(ids))
        B.G.nodes[nid]["description"] = one_line(rv.get("summary") or first_req, 300)
        B.cards[doc] = "\n".join(lines)

    for fid, f in files.items():
        doc = f"file-{fid}.md"
        if doc not in doc_files:
            continue
        fname = f.get("filename") or fid
        nid = B.node(f"File: {fname} [{fid[:8]}]", "File", "", doc)
        u = f.get("user_id") or ""
        if user(u):
            B.edge(u, nid, "uploaded")
        meta = loose(f.get("meta")) or {}
        up = day(f.get("created_at"))
        size = meta.get("size")
        B.prop(nid, filename=fname, format=file_format(fname, meta.get("content_type")),
               content_type=meta.get("content_type"), size_bytes=int(size) if str(size or "").isdigit() else None,
               uploaded_on=up if up != "unknown date" else "")
        full = (docs / doc).read_text(encoding="utf-8")
        body = full.split("\n", 2)[-1] if full.startswith("#") else full
        in_chats = [chats[c].get("title") or c for c in file_chats.get(fid, []) if c in chats]
        ftags = vocab_tags.get(doc) or []
        for tag in ftags:
            B.edge(nid, topic(tag), "about")
        lines = [f"# {fname}", "",
                 f"Type: file, format {file_format(fname, meta.get('content_type'))} "
                 f"({meta.get('content_type') or 'unknown type'}, {meta.get('size') or '?'} bytes). "
                 f"Uploaded {day(f.get('created_at'))} by {u or 'unknown'}"
                 + (f" ({teams[u]} team)" if teams.get(u) else "") + "."]
        if meta.get("collection_name"):
            lines.append(f"Knowledge collection: {meta['collection_name']}")
        if in_chats:
            lines.append("Attached in chats: " + "; ".join(in_chats))
        if ftags:
            lines.append(f"Tags: {', '.join(ftags)}")
        if doc_types.get(doc):
            B.G.nodes[nid]["doc_type"] = doc_types[doc]
            lines.append(f"Document type: {doc_types[doc].replace('-', ' ')}")
        lines.append(f"Opening text: {one_line(body, TEXT_CHARS)}")
        ids = identifiers(full)
        if ids:
            lines.append("Identifiers: " + ", ".join(ids))
        B.G.nodes[nid]["description"] = one_line(body, 300)
        B.cards[doc] = "\n".join(lines)

    # ── people ──
    for doc in sorted(d for d in doc_files if d.startswith("person-")):
        u = doc[len("person-"):-3]
        nid = user(u)
        if not nid:
            continue
        nb = list(B.G.neighbors(nid))
        my_chats = [n for n in nb if B.G.nodes[n]["entity_type"] == "Chat"]
        topics = Counter(t for ch in my_chats for t in B.G.neighbors(ch) if B.G.nodes[t]["entity_type"] == "Topic")
        items = [n for n in nb if B.G.nodes[n]["entity_type"] in ("Agent", "Step", "Tool")]
        lines = [f"# {u}", "",
                 f"Type: user (pseudonymous). Team: {teams.get(u, 'unknown')}. "
                 f"{len(my_chats)} indexed chats, {sum(1 for n in nb if B.G.nodes[n]['entity_type'] == 'File')} files."]
        if topics:
            lines.append("Main topics: " + ", ".join(f"{t[len('Topic: '):]} ({k})" for t, k in topics.most_common(8)))
        if items:
            lines.append("Agents and tools created, installed or used: " + ", ".join(items))
        if my_chats:
            lines.append("Chats: " + "; ".join(n[len("Chat: "):].rsplit(" [", 1)[0] for n in my_chats[:25]))
        B.G.nodes[nid]["description"] = lines[2]
        B.prop(nid, chats=len(my_chats), files=sum(1 for n in nb if B.G.nodes[n]["entity_type"] == "File"))
        B.cards[doc] = "\n".join(lines)

    # ── catalogs ──
    for kind, plural in (("agent", "agents"), ("tool", "tools"), ("step", "steps")):
        doc = f"catalog-{plural}.md"
        if doc not in doc_files:
            continue
        members = sorted(n for (k, _), n in item_node.items() if k == kind)
        nid = B.node(f"Catalog: {plural}", "Catalog", f"All {len(members)} Agent Hub {plural}", doc)
        B.prop(nid, kind=plural, count=len(members))
        for m in members:
            B.edge(nid, m, "lists")
        B.cards[doc] = "\n".join([f"# Agent Hub catalog: all {plural}", "",
                                  f"Type: catalog. Agent Hub has {len(members)} {plural}.",
                                  "Members: " + "; ".join(f"{m} ({one_line(B.G.nodes[m]['description'], 90)})" for m in members)])

    # every uploaded file / every team, as one card each (complete lists for counting questions)
    T = lambda n: B.G.nodes[n]["entity_type"]
    if "catalog-files.md" in doc_files:
        fnodes = sorted(n for n in B.G if T(n) == "File")
        nid = B.node("Catalog: files", "Catalog", f"All {len(fnodes)} uploaded files", "catalog-files.md")
        B.prop(nid, kind="files", count=len(fnodes))
        fmts = Counter(B.G.nodes[n].get("format", "unknown") for n in fnodes)
        rows = []
        for n in fnodes:
            d = B.G.nodes[n]
            up = next((m for m in B.G.neighbors(n) if T(m) == "User"), None)
            B.edge(nid, n, "lists")
            rows.append(f"{d.get('filename', n)} (format {d.get('format', '?')}"
                        + (f", {d['doc_type'].replace('-', ' ')}" if d.get("doc_type") else "")
                        + (f", {d['size_bytes']} bytes" if d.get("size_bytes") else "")
                        + (f", uploaded by {up}" if up else "") + ")")
        B.cards["catalog-files.md"] = "\n".join([
            "# Uploaded files catalog: all files", "",
            f"Type: catalog. There are {len(fnodes)} uploaded files: "
            + ", ".join(f"{k} {f}" for f, k in fmts.most_common()) + ". Only these are files; "
            "names mentioned inside a file's text are not separate files.",
            "Files: " + "; ".join(rows)])
    if "catalog-teams.md" in doc_files:
        tnodes = sorted(n for n in B.G if T(n) == "Team")
        nid = B.node("Catalog: teams", "Catalog", f"All {len(tnodes)} teams", "catalog-teams.md")
        B.prop(nid, kind="teams", count=len(tnodes))
        rows = []
        for t in tnodes:
            mem = sorted(m for m in B.G.neighbors(t) if T(m) == "User")
            B.prop(t, members=len(mem))
            B.edge(nid, t, "lists")
            # what the team works on: topics of its members' chats and files (the works_on rule)
            work = Counter(tp for u in mem for x in B.G.neighbors(u) if T(x) in ("Chat", "File")
                           for tp in B.G.neighbors(x) if T(tp) == "Topic")
            rows.append(f"{B.G.nodes[t].get('name', t)}: members {', '.join(mem)}"
                        + (f"; works on {', '.join(w[len('Topic: '):] for w, _ in work.most_common(4))}" if work else ""))
        no_team = sorted(n for n in B.G if T(n) == "User" and not any(T(m) == "Team" for m in B.G.neighbors(n)))
        B.cards["catalog-teams.md"] = "\n".join([
            "# Teams catalog: all teams", "",
            f"Type: catalog. There are {len(tnodes)} teams: "
            + ", ".join(B.G.nodes[t].get("name", t) for t in tnodes) + ". No other teams exist.",
            "Teams: " + "; ".join(rows) + (f". Not on any team: {', '.join(no_team)}" if no_team else "")])

    mentions(B, docs)
    write_storage(B, out_dir, ex)


def mentions(B: "Builder", docs: Path) -> None:
    """`mentions` edges, found deterministically in each chat's and file's full text and
    checked before they enter the graph (schema: Chat|File -mentions-> Technique|File):
      Technique  an ATT&CK id (T1059.001) that exists in the ATT&CK bundle; unknown ids are
                 dropped, retired ones kept but flagged (retired=true) so answers can say so
      File       an uploaded file's exact name (with or without its user-NN_ prefix) in a
                 CHAT's text, unless the chat already attaches it (contains). Not file -> file:
                 the graph is undirected, so "A mentions B" and "B mentions A" would look the same
    No LLM: every edge is a literal match that can be shown in the text."""
    T = lambda n: B.G.nodes[n].get("entity_type")
    try:
        ref = json.loads(ATTACK.read_text(encoding="utf-8"))
        known, version = ref["techniques"], ref.get("version", "")
    except Exception as e:
        known, version = None, ""
        print(f"[meta] mentions: no ATT&CK reference ({e}); technique links skipped")
    names = {}
    for n in B.G:
        if T(n) == "File" and B.G.nodes[n].get("filename"):
            fn = B.G.nodes[n]["filename"]
            for k in {fn, re.sub(r"^user-\d+_", "", fn)}:
                if len(k) >= 6 and "." in k:
                    names[k.lower()] = n
    name_rx = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(k) for k in sorted(names, key=len, reverse=True))
                         + r")(?![\w-])", re.I) if names else None
    stats = Counter()
    for n in [x for x in B.G if T(x) in ("Chat", "File")]:
        doc = B.G.nodes[n].get("file_path")
        if not doc or not (docs / doc).exists():
            continue
        full = (docs / doc).read_text(encoding="utf-8")
        found = []
        if known is not None:
            for tid in sorted(set(TID.findall(full))):
                t = known.get(tid)
                if not t:
                    stats["unknown_tid"] += 1
                    continue
                tn = B.node(f"Technique: {tid}", "Technique",
                            f"ATT&CK {tid} {t['name']}" + (" (retired id)" if t["retired"] else ""))
                B.prop(tn, technique_id=tid, name=t["name"], retired=bool(t["retired"]), attack_version=version)
                B.edge(n, tn, "mentions", f"the text names {tid}")
                found.append(f"{tid} {t['name']}" + (" (retired id)" if t["retired"] else ""))
                stats["technique"] += 1
        if name_rx and T(n) == "Chat":
            for m in {m.group(1).lower() for m in name_rx.finditer(full)}:
                f = names[m]
                if f == n or (B.G.has_edge(n, f) and "contains" in B.G.edges[n, f]["keywords"].split(",")):
                    continue
                B.edge(n, f, "mentions", f"the text names {B.G.nodes[f]['filename']}")
                stats["file"] += 1
        if found and doc in B.cards:
            B.cards[doc] += "\nATT&CK techniques named: " + "; ".join(found)
    print(f"[meta] mentions: {stats['technique']} technique link(s) "
          f"({sum(1 for x in B.G if T(x) == 'Technique')} techniques, ATT&CK {version or '?'}), "
          f"{stats['file']} file link(s), {stats['unknown_tid']} unknown id(s) dropped")


def embed(texts: list) -> list:
    out = []
    with httpx.Client(timeout=120) as c:
        for i in range(0, len(texts), 16):
            r = c.post(f"{EMBED_HOST}/api/embed",
                       json={"model": EMBED_MODEL, "input": ["search_document: " + t for t in texts[i:i + 16]]})
            r.raise_for_status()
            for v in r.json()["embeddings"]:
                n = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / n for x in v])
    return out


def write_storage(B: Builder, out: Path, ex: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    iso = datetime.now(timezone.utc).isoformat()
    # content duplicates: carry LightRAG's records over from the export's main index
    # (and give them no card), so gold labels on a duplicate upload map to the
    # original as they do there
    dups = {}
    base = ex / "storage" / "kv_store_doc_status.json"
    if base.exists():
        for k, v in json.loads(base.read_text(encoding="utf-8")).items():
            if v.get("status") == "failed" and "Identical content already exists" in (v.get("error_msg") or ""):
                dups[k] = v
    dup_files = {v.get("file_path") for v in dups.values()}
    docs = sorted(d for d in B.cards if d not in dup_files)
    cid = {d: f"{d[:-3]}-card" for d in docs}
    for d in docs:      # each node points at the card(s) it appears on
        for n in B.card_nodes[d]:
            B.G.nodes[n]["source_id"] = cid[d]
    t0 = time.time()
    vecs = embed([B.cards[d] for d in docs])
    print(f"[meta] embedded {len(docs)} cards in {time.time() - t0:.1f}s")

    def dump(name, obj):
        tmp = out / (name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, out / name)

    tokens = lambda s: max(1, len(s) // 4)
    status, full_docs, chunks, full_ents, ent_chunks = {}, {}, {}, {}, defaultdict(list)
    for d in docs:
        did, text = d[:-3], B.cards[d]
        h = hashlib.md5(text.encode()).hexdigest()
        status[did] = {"status": "processed", "content_summary": text[:100], "content_length": len(text),
                       "created_at": iso, "updated_at": iso, "file_path": d, "content_hash": h,
                       "chunks_count": 1, "chunks_list": [cid[d]]}
        full_docs[did] = {"content": text, "file_path": d, "content_hash": h, "create_time": now, "update_time": now, "_id": did}
        chunks[cid[d]] = {"tokens": tokens(text), "content": text, "chunk_order_index": 0, "full_doc_id": did,
                          "file_path": d, "create_time": now, "update_time": now, "_id": cid[d]}
        names = sorted(B.card_nodes[d] | {n for x in B.card_nodes[d] for n in B.G.neighbors(x)})
        full_ents[did] = {"entity_names": names, "count": len(names), "create_time": now, "update_time": now, "_id": did}
        for n in names:
            ent_chunks[n].append(cid[d])
    status.update(dups)
    dump("kv_store_doc_status.json", status)
    dump("kv_store_full_docs.json", full_docs)
    dump("kv_store_text_chunks.json", chunks)
    dump("kv_store_full_entities.json", full_ents)
    dump("kv_store_entity_chunks.json", {n: {"chunk_ids": c, "count": len(c), "update_time": now, "_id": n}
                                         for n, c in ent_chunks.items()})
    flat = struct.pack(f"<{len(vecs) * 768}f", *[x for v in vecs for x in v])
    dump("vdb_chunks.json", {"embedding_dim": 768, "matrix": base64.b64encode(flat).decode(),
                             "data": [{"__id__": cid[d], "__created_at__": now, "content": B.cards[d],
                                       "full_doc_id": d[:-3], "file_path": d} for d in docs]})
    tmp = out / "graph_chunk_entity_relation.graphml.tmp"
    nx.write_graphml(B.G, tmp)
    os.replace(tmp, out / "graph_chunk_entity_relation.graphml")
    types = Counter(d["entity_type"] for _, d in B.G.nodes(data=True))
    rels = Counter(r for _, _, d in B.G.edges(data=True) for r in d["keywords"].split(","))
    print(f"[meta] {B.G.number_of_nodes()} nodes {dict(types)}")
    print(f"[meta] {B.G.number_of_edges()} edges {dict(rels)}")
    print(f"[meta] {len(docs)} cards, avg {sum(len(B.cards[d]) for d in docs) // max(1, len(docs))} chars -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--out", default="storage-metadata")
    a = ap.parse_args()
    ex = Path(a.export_dir).resolve()
    build(ex, ex / a.out)
