"""
Answer factual questions (counts, lists, largest/smallest, who-belongs-where) straight from the
metadata graph, so the answer is exact and complete instead of pieced together from the few
records text retrieval returns.

  1. plan     the LLM turns the question into a small structured lookup, choosing only from
              what the semantic model defines (classes, their properties, relations), or says
              it isn't a lookup question (then the normal answer over retrieved text runs)
  2. check    the plan is validated against the schema: unknown classes, properties,
              relations or enum values are refused, so the model can't invent fields
  3. run      executed deterministically on the graph (networkx): filters on properties,
              filters through relations (nested, e.g. a team whose member uploaded a file,
              or "never attached to any chat"), then count / list / max / min / exists
  4. word     the LLM phrases the answer from that exact result only. Names that don't exist
              ("the Blue Team") come back as missing, with the names that do exist, so the
              answer can correct the question instead of playing along

  plan = {"lookup": true, "action": "count"|"list"|"max"|"min"|"exists",
          "class": "File",
          "where":   [{"property": "format", "op": "=", "value": "pdf"}],
          "related": [{"relation": "uploaded", "class": "User", "name": "user-05",
                       "none": false, "where": [...], "related": [...]}],
          "group_by": "format", "by": "size_bytes", "show": ["filename", "format"]}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import schema_model as SM

INCOMING = Path(__file__).resolve().parent / "datasets" / "incoming"
ACTIONS = ("count", "list", "max", "min", "exists")
OPS = ("=", "!=", ">", "<", ">=", "<=", "contains")
MAX_ITEMS = 60            # items written into the answer prompt (the count is always exact)
_GRAPHS: dict = {}        # export dir -> (mtime, graph)


# ───────────────────────── graph + naming ─────────────────────────

def _graph(dataset: str):
    base = (dataset or "").split("~")[0]
    ex = INCOMING / base
    p = ex / "storage-metadata" / SM.GRAPH_FILE
    if not base or not p.exists():
        return None
    mt = p.stat().st_mtime
    if _GRAPHS.get(str(ex), (None,))[0] != mt:
        G, _ = SM._graph(ex)
        _GRAPHS[str(ex)] = (mt, G)
    return _GRAPHS[str(ex)][1]


def label(G, n) -> str:
    d = G.nodes[n]
    if d.get("technique_id"):   # "T1086 PowerShell": the id is what people search and cite
        return f"{d['technique_id']} {d.get('name', '')}".strip()
    for k in ("name", "filename", "title", "model_id"):
        if d.get(k):
            return str(d[k]) + (f" [{str(n).rsplit('[', 1)[-1]}" if k == "title" and "[" in str(n) else "")
    return str(n)


def _norm(s: str, cls: str = "") -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
    if cls:   # "the planning team" -> "planning" when looking for a Team
        s = re.sub(rf"\b(the|a|an|{cls.lower()}s?)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _names(G, n) -> set:
    d = G.nodes[n]
    out = {_norm(n), _norm(label(G, n))}
    for k in ("name", "filename", "title", "slug", "model_id"):
        if d.get(k):
            out.add(_norm(d[k]))
            if k == "filename" and "." in str(d[k]):
                out.add(_norm(str(d[k]).rsplit(".", 1)[0]))
    return {x for x in out if x}


def match_name(G, nodes: list, name: str, cls: str) -> list:
    """Nodes a name refers to: exact name first, else nodes whose names contain all its words."""
    want = _norm(name, cls) or _norm(name)
    if not want:
        return list(nodes)
    exact = [n for n in nodes if want in _names(G, n)]
    if exact:
        return exact
    words = want.split()
    hit = [n for n in nodes if any(all(w in x for w in words) for x in _names(G, n))]
    if hit:
        return hit
    # "the healthcheck script" -> user-05_healthcheck.py: drop words that describe the kind of
    # record rather than name it, and match on what is left
    core = [w for w in words if w not in KIND_WORDS]
    if not core or core == words:
        return []
    return [n for n in nodes if any(all(w in x for w in core) for x in _names(G, n))]


KIND_WORDS = {"script", "scripts", "file", "files", "document", "doc", "docs", "chat", "chats", "conversation",
              "record", "spreadsheet", "sheet", "report", "upload", "uploaded", "attachment", "my", "our", "their",
              "his", "her", "this", "that", "of", "for"}


# ───────────────────────── plan: the menu and the prompt ─────────────────────────

def menu(schema: dict) -> str:
    """What a lookup may use: implemented, non-abstract classes with their properties, and
    relations with their from/to classes."""
    classes = schema.get("classes") or {}
    lines = ["CLASSES (property: type):"]
    for c, d in classes.items():
        d = d or {}
        if d.get("abstract") or d.get("status") == "proposed":
            continue
        props = []
        for pn, pd in SM.class_properties(schema, c).items():
            t = pd.get("type", "string")
            if t == "enum":
                vals = list(SM.leaf_terms(schema, pd["taxonomy"])) if pd.get("taxonomy") else SM._list(pd.get("values"))
                t = "one of " + "|".join(vals)
            props.append(f"{pn}: {t}")
        lines.append(f"- {c}: {d.get('description', '')[:90]}" + (f"\n    properties: {', '.join(props)}" if props else ""))
    lines.append("RELATIONS (name: from -> to):")
    for r, d in (schema.get("relations") or {}).items():
        d = d or {}
        if d.get("status") == "proposed":
            continue
        lines.append(f"- {r}: {'/'.join(SM._list(d.get('from')))} -> {'/'.join(SM._list(d.get('to')))}")
    return "\n".join(lines)


PLAN_SYS = "You turn questions about a team's records into structured lookups. Reply with JSON only."
PLAN_PROMPT = """Decide whether this question can be answered by looking up records (counting, listing, finding the largest/smallest, checking who/what is linked to what). It is NOT a lookup when the answer is a value written INSIDE a document or chat (a stop word, a port number, a date or time from a schedule, a step in a procedure, a setting), or when it asks what something says, how to do something, why, or for advice. Record properties are only the ones listed below; anything else lives inside the documents.

Use ONLY the class, property and relation names below. Put names of specific things (a team, user, file, chat, agent) in "name" exactly as the question says them, even if you think they may not exist.

{menu}

JSON fields:
- "lookup": true or false
- "action": count | list | max | min | exists
- "class": the class being counted or listed
- "where": property filters [{{"property": ..., "op": "=|!=|>|<|>=|<=|contains", "value": ...}}]
- "related": filters through a relation [{{"relation": ..., "class": <the class at the other end>, "name": <optional>, "none": <true = has NO such link>, "where": [...], "related": [...]}}]
- "group_by": optional property to break a count down by
- "by": the property for max/min
- "show": properties to show for listed items

Examples:
Q: how many pdf files are there?
{{"lookup": true, "action": "count", "class": "File", "where": [{{"property": "format", "op": "=", "value": "pdf"}}]}}
Q: who is on the Red Team?
{{"lookup": true, "action": "list", "class": "User", "related": [{{"relation": "member_of", "class": "Team", "name": "Red Team"}}]}}
Q: which chats never had a file attached?
{{"lookup": true, "action": "list", "class": "Chat", "related": [{{"relation": "contains", "class": "File", "none": true}}]}}
Q: which team does the person who uploaded the runbook belong to?
{{"lookup": true, "action": "list", "class": "Team", "related": [{{"relation": "member_of", "class": "User", "related": [{{"relation": "uploaded", "class": "File", "name": "runbook"}}]}}]}}
Q: how many files did user-07 upload?
{{"lookup": true, "action": "count", "class": "File", "related": [{{"relation": "uploaded", "class": "User", "name": "user-07"}}], "group_by": "format"}}
Q: what has the Red Team been working on?
{{"lookup": true, "action": "count", "class": "Topic", "related": [{{"relation": "about", "class": "Chat", "related": [{{"relation": "opened", "class": "User", "related": [{{"relation": "member_of", "class": "Team", "name": "Red Team"}}]}}]}}]}}
Q: which chats mention the runbook but don't attach it?
{{"lookup": true, "action": "list", "class": "Chat", "related": [{{"relation": "mentions", "class": "File", "name": "runbook"}}, {{"relation": "contains", "class": "File", "name": "runbook", "none": true}}]}}
Q: which of the mentioned techniques are retired?
{{"lookup": true, "action": "list", "class": "Technique", "where": [{{"property": "retired", "op": "=", "value": true}}], "show": ["technique_id"]}}
Q: what does the rules of engagement say about stop words?
{{"lookup": false}}
Q: what's the stop word for the exercise?
{{"lookup": false}}
Q: what port does the health check use?
{{"lookup": false}}
Q: when is the maintenance window?
{{"lookup": false}}

Q: {question}
"""


def parse_plan(reply: str) -> Optional[dict]:
    a, b = reply.find("{"), reply.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(reply[a:b + 1])
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


# ───────────────────────── check the plan against the schema ─────────────────────────

def validate(schema: dict, plan: dict) -> list:
    errs = []
    classes = schema.get("classes") or {}
    rels = schema.get("relations") or {}

    def props_of(c):
        return SM.class_properties(schema, c)

    def check_where(c, where, path):
        for w in where or []:
            pn = (w or {}).get("property")
            if pn not in props_of(c):
                errs.append(f"{path}: {c} has no property '{pn}'")
            elif (w.get("op") or "=") not in OPS:
                errs.append(f"{path}: unknown operator {w.get('op')}")
            elif w.get("value") in (None, ""):
                errs.append(f"{path}: filter on {pn} has no value")

    def check_related(c, related, path):
        for r in related or []:
            r = r or {}
            rn, rc = r.get("relation"), r.get("class")
            if rn not in rels:
                errs.append(f"{path}: no relation '{rn}'")
                continue
            if rc not in classes:
                errs.append(f"{path}: no class '{rc}'")
                continue
            if not SM._relation_ok(schema, rels[rn] or {}, c, rc):
                errs.append(f"{path}: {rn} doesn't connect {c} and {rc}")
            check_where(rc, r.get("where"), f"{path}.{rn}")
            check_related(rc, r.get("related"), f"{path}.{rn}")

    c = plan.get("class")
    if c not in classes:
        return [f"no class '{c}'"]
    if plan.get("action") not in ACTIONS:
        errs.append(f"unknown action {plan.get('action')}")
    check_where(c, plan.get("where"), c)
    check_related(c, plan.get("related"), c)
    for k in ("group_by", "by"):
        if plan.get(k) and plan[k] not in props_of(c):
            errs.append(f"{c} has no property '{plan[k]}' to {k}")
    if plan.get("action") in ("max", "min") and not plan.get("by"):
        errs.append(f"{plan['action']} needs 'by'")
    return errs


# ───────────────────────── run it on the graph ─────────────────────────

def _cmp(v, op: str, want) -> bool:
    if v is None:
        return op == "!="
    a, b = v, want
    try:
        if isinstance(v, (int, float)) or str(v).lstrip("-").replace(".", "", 1).isdigit():
            a, b = float(v), float(want)
    except (TypeError, ValueError):
        a, b = str(v).lower(), str(want).lower()
    if isinstance(a, str) or isinstance(b, str):
        a, b = str(a).lower(), str(b).lower()
    if op == "contains":
        return str(b) in str(a)
    return {"=": a == b, "!=": a != b, ">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}.get(op, False)


class Runner:
    def __init__(self, schema: dict, G):
        self.s, self.G = schema, G
        self.missing, self.ambiguous = [], []

    def of(self, cls: str) -> list:
        kinds = SM.descendants(self.s, cls)
        return [n for n, d in self.G.nodes(data=True) if d.get("entity_type") in kinds]

    def where_ok(self, n, where) -> bool:
        d = self.G.nodes[n]
        return all(_cmp(d.get(w["property"]), w.get("op") or "=", w.get("value")) for w in where or [])

    def targets(self, r: dict) -> Optional[set]:
        """The nodes a related-filter may link to (None: no restriction)."""
        cls = r["class"]
        pool = [n for n in self.of(cls) if self.where_ok(n, r.get("where"))
                and all(self.related_ok(n, x) for x in r.get("related") or [])]
        if r.get("name"):
            hit = match_name(self.G, self.of(cls), r["name"], cls)
            if not hit:
                self.missing.append({"class": cls, "name": r["name"],
                                     "existing": sorted(label(self.G, n) for n in self.of(cls))[:40]})
            elif len(hit) > 1:
                self.ambiguous.append({"class": cls, "name": r["name"], "matches": sorted(label(self.G, n) for n in hit)})
            pool = [n for n in pool if n in set(hit)]
        return set(pool)

    def related_ok(self, n, r: dict) -> bool:
        tg = r.get("_targets")
        if tg is None:
            tg = r["_targets"] = self.targets(r)
        linked = any(m in tg and r["relation"] in (self.G.edges[n, m].get("keywords") or "").split(",")
                     for m in self.G.neighbors(n))
        return not linked if r.get("none") else linked

    def run(self, plan: dict) -> dict:
        c, action = plan["class"], plan["action"]
        nodes = [n for n in self.of(c) if self.where_ok(n, plan.get("where"))
                 and all(self.related_ok(n, r) for r in plan.get("related") or [])]
        out = {"action": action, "class": c, "count": len(nodes), "missing": self.missing,
               "ambiguous": self.ambiguous, "total_of_class": len(self.of(c))}
        if action in ("max", "min") and nodes:
            key = plan["by"]
            vals = [n for n in nodes if self.G.nodes[n].get(key) not in (None, "")]
            if vals:
                pick = (max if action == "max" else min)(vals, key=lambda n: self.G.nodes[n][key])
                best = self.G.nodes[pick][key]
                nodes = [n for n in vals if self.G.nodes[n][key] == best]
        show = [p for p in (plan.get("show") or []) if p in SM.class_properties(self.s, c)]
        if plan.get("by") and plan["by"] not in show:
            show.append(plan["by"])
        if plan.get("group_by"):
            g = {}
            for n in nodes:
                k = str(self.G.nodes[n].get(plan["group_by"], "(none)"))
                g[k] = g.get(k, 0) + 1
            out["groups"] = dict(sorted(g.items(), key=lambda x: (-x[1], x[0])))
        props = SM.class_properties(self.s, c)
        out["items"] = [{"name": label(self.G, n),
                         **{p: self.G.nodes[n][p] for p in (show or list(props)[:4]) if p in self.G.nodes[n]}}
                        for n in sorted(nodes, key=lambda n: label(self.G, n).lower())]
        return out


def describe(plan: dict) -> str:
    """The plan as one readable line, e.g. 'count File where format = pdf, linked by uploaded to User "user-05"'."""
    def where(ws):
        return " and ".join(f"{w['property']} {w.get('op') or '='} {w.get('value')}" for w in ws or [])

    def rel(rs):
        parts = []
        for r in rs or []:
            s = f"{'with no ' if r.get('none') else ''}{r['relation']} link to {r['class']}"
            if r.get("name"):
                s += f' "{r["name"]}"'
            if r.get("where"):
                s += f" where {where(r['where'])}"
            if r.get("related"):
                s += f" ({rel(r['related'])})"
            parts.append(s)
        return ", ".join(parts)

    s = f"{plan['action']} {plan['class']}"
    if plan.get("action") in ("max", "min"):
        s += f" by {plan.get('by')}"
    if plan.get("where"):
        s += f" where {where(plan['where'])}"
    if plan.get("related"):
        s += f", {rel(plan['related'])}"
    if plan.get("group_by"):
        s += f", grouped by {plan['group_by']}"
    return s


def result_text(plan: dict, res: dict, relaxed: Optional[dict]) -> str:
    """The exact lookup result as plain text for the answer prompt."""
    c = res["class"]
    lines = [f"Lookup: {describe(plan)}"]
    for m in res["missing"]:
        lines.append(f'NOT FOUND: there is no {m["class"]} called "{m["name"]}". '
                     f'The {m["class"]} records that exist: {", ".join(m["existing"]) or "none"}.')
    for a in res["ambiguous"]:
        lines.append(f'AMBIGUOUS: "{a["name"]}" matches several {a["class"]} records: {", ".join(a["matches"])}.')
    if res["action"] == "exists":
        lines.append(f"Result: {'yes' if res['count'] else 'no'} ({res['count']} matching {c}).")
    elif res["action"] in ("max", "min"):
        lines.append(f"Result: {'the largest' if res['action'] == 'max' else 'the smallest'} by {plan.get('by')}:")
    elif res["action"] == "count":
        lines.append(f"Result: exactly {res['count']} {c} record(s) match (out of {res['total_of_class']} {c} in total).")
    else:
        lines.append(f"Result: exactly {res['count']} {c} record(s) match.")
    if res.get("groups"):
        lines.append("Breakdown by " + plan["group_by"] + ": " + ", ".join(f"{k}: {v}" for k, v in res["groups"].items()))
    for it in res["items"][:MAX_ITEMS]:
        extra = ", ".join(f"{k} {v}" for k, v in it.items() if k != "name")
        lines.append(f"- {it['name']}" + (f" ({extra})" if extra else ""))
    if len(res["items"]) > MAX_ITEMS:
        lines.append(f"... and {len(res['items']) - MAX_ITEMS} more (the count above is exact)")
    if relaxed is not None:
        lines.append(f"Without the property filters ({', '.join(relaxed['dropped'])}), {relaxed['count']} {c} record(s) match: "
                     + "; ".join(f"{i['name']} (" + ", ".join(f"{k} {v}" for k, v in i.items() if k != 'name') + ")"
                                 for i in relaxed["items"][:20]))
    return "\n".join(lines)


ANSWER_SYS = (
    "You answer questions about a team's records from an exact database lookup. The lookup result is "
    "complete and authoritative: if it says N records match, there are exactly N, and anything not in it "
    "does not exist in the data. Answer the question directly with the facts from the result. ONLY when the "
    "question states or assumes something the result contradicts (a name marked NOT FOUND, a wrong number, "
    "something that has zero matches) start by correcting it plainly, e.g. \"There is no Blue Team.\" or "
    "\"There are 2 PDFs, not 5.\"; never invent an assumption the question didn't make. After a correction "
    "ALWAYS give what the result shows instead: the records that do exist (e.g. the teams listed after NOT "
    "FOUND), or what matches without the filter (e.g. the files that person did upload). When nothing "
    "matches, say so in a full sentence about the subject (\"user-01 is not on any team.\", \"user-02 has not "
    "created any agents.\"). Use only facts from the result; do not add any, and do not add details the "
    "question didn't ask for (totals, dates, other members): every extra sentence is a chance to be wrong. "
    "Usually one or two sentences.")


async def answer(question: str, dataset: str, llm) -> Optional[dict]:
    """The graph answer for a question, or None when it isn't a lookup question (or the dataset
    has no metadata graph, or the plan doesn't fit the schema). `llm(prompt, system_prompt)` is
    the chat call. Returns {answer, plan, query, result, text, ms}."""
    G = _graph(dataset)
    if G is None:
        return None
    schema, _ = SM.load()
    t0 = time.time()
    reply = await llm(PLAN_PROMPT.format(menu=menu(schema), question=question), PLAN_SYS)
    plan = parse_plan(reply or "")
    if not plan or not plan.get("lookup"):
        return {"skipped": "not a lookup question", "plan": plan, "ms": round((time.time() - t0) * 1000)}
    errs = validate(schema, plan)
    if errs:
        return {"skipped": "plan doesn't fit the schema: " + "; ".join(errs), "plan": plan,
                "ms": round((time.time() - t0) * 1000)}
    res = Runner(schema, G).run(json.loads(json.dumps(plan)))   # the runner caches into its copy
    relaxed = None
    if res["count"] == 0 and plan.get("where") and not res["missing"]:
        # e.g. "which PDF did user-05 upload?" -> none; say what user-05 did upload
        loose = {**plan, "where": [], "action": "list"}
        r2 = Runner(schema, G).run(json.loads(json.dumps(loose)))
        relaxed = {"count": r2["count"], "items": r2["items"],
                   "dropped": [f"{w['property']} {w.get('op') or '='} {w.get('value')}" for w in plan["where"]]}
    text = result_text(plan, res, relaxed)
    ans = await llm(f"Question: {question}\n\nLOOKUP RESULT:\n{text}\n\nAnswer:", ANSWER_SYS)
    return {"answer": (ans or "").strip(), "plan": {k: v for k, v in plan.items()}, "query": describe(plan),
            "result": res, "text": text, "ms": round((time.time() - t0) * 1000)}
