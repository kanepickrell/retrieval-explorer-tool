"""
The semantic model (datasets/schema/schema.yaml): loading, validating, saving, and measuring a
dataset's metadata graph against it.

Pure functions, no web framework, so the ingest scripts (tag_docs.py) read the same
vocabularies the Prospector page edits. schema_api.py serves these over HTTP.

  load()            -> (schema dict, yaml text)
  validate(schema)  -> [error, ...]
  save(schema)      -> new version (the old one is kept under schema/history/)
  leaf_terms(schema, "topics") -> {term: description}  (what the LLM may assign)
  vocab_hash(terms) -> the hash tags.json records, to tell when a dataset was tagged with
                       an older vocabulary
  measure(schema, export_dir) -> conformance + coverage metrics for one dataset (classes,
                       relations, class properties, taxonomies, glossary, rules)
  class_properties(schema, cls) -> a class's properties, inherited ones included
  suggestions(schema, exports, export_dir) -> bottom-up vocabulary proposals: terms the LLM
                       tagger suggested when nothing fitted, grouped and checked against the
                       schema, plus term-health advisories (overloaded / unused terms)
  approve_term / merge_terms / reject_terms -> apply a reviewed suggestion to a schema dict
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
SCHEMA_PATH = Path(os.getenv("PROTOGRAPH_SCHEMA") or _HERE / "datasets" / "schema" / "schema.yaml")
GRAPH_FILE = "graph_chunk_entity_relation.graphml"
# tag_docs.py's LLM grouping of open suggestion labels: {taxonomy: {label: {group, description, same_as}}}
CANDIDATES_PATH = SCHEMA_PATH.parent / "candidates.json"
# which docs/<prefix>*.md files hold records of a class (taxonomy coverage counts these)
DOC_PREFIX = {"Chat": "chat-", "File": "file-", "Agent": "agent-", "Step": "step-", "Tool": "tool-",
              "User": "person-", "Catalog": "catalog-"}
SECTIONS = ("classes", "relations", "taxonomies", "glossary", "rules")
NAME = re.compile(r"^[A-Za-z][\w\- ]{0,63}$")
# class properties: structured facts build_metadata_graph.py writes on nodes
PROP_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
PROP_TYPES = ("string", "integer", "number", "date", "boolean", "enum")
SYSTEM_ATTRS = {"entity_id", "entity_type", "description", "source_id", "file_path"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# ───────────────────────── load / save ─────────────────────────

def load(path: Path = SCHEMA_PATH) -> Tuple[dict, str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    data = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError("schema.yaml must hold a mapping at the top level")
    for k in SECTIONS:
        data.setdefault(k, {})
        if data[k] is None:
            data[k] = {}
    data.setdefault("version", 1)
    return data, text


def parse(text: str) -> dict:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("the YAML must hold a mapping at the top level")
    for k in SECTIONS:
        data[k] = data.get(k) or {}
    return data


def _header(text: str) -> str:
    """The leading comment block of the file, kept across saves from the UI."""
    out = []
    for line in text.splitlines():
        if line.startswith("#") or (not line.strip() and out):
            out.append(line)
        else:
            break
    return "\n".join(out).rstrip() + "\n\n" if out else ""


def dump(schema: dict) -> str:
    return yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False)


def save(schema: dict, path: Path = SCHEMA_PATH) -> int:
    """Write a validated schema as the next version; the current file goes to history/."""
    errors = validate(schema)
    if errors:
        raise ValueError("; ".join(errors))
    old, old_text = load(path) if path.exists() else ({"version": 0}, "")
    if old_text:
        hist = path.parent / "history"
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"schema-v{old.get('version', 0)}-{time.strftime('%Y%m%d-%H%M%S')}.yaml").write_text(old_text, encoding="utf-8")
    schema = {"version": int(old.get("version") or 0) + 1, **{k: v for k, v in schema.items() if k != "version"}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_header(old_text) + dump(schema), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return schema["version"]


def history(path: Path = SCHEMA_PATH) -> List[dict]:
    hist = path.parent / "history"
    out = []
    for p in sorted(hist.glob("schema-v*.yaml")) if hist.is_dir() else []:
        m = re.match(r"schema-v(\d+)-(\d{8}-\d{6})", p.stem)
        if m:
            out.append({"version": int(m.group(1)), "replaced_at": m.group(2), "file": p.name})
    return sorted(out, key=lambda h: -h["version"])


# ───────────────────────── structure helpers ─────────────────────────

def descendants(schema: dict, name: str) -> set:
    """A class and every class under it (a relation on Capability accepts an Agent)."""
    classes = schema.get("classes") or {}
    out, frontier = {name}, {name}
    while frontier:
        frontier = {c for c, d in classes.items() if (d or {}).get("parent") in frontier} - out
        out |= frontier
    return out


def expand(schema: dict, names) -> set:
    out = set()
    for n in names or []:
        out |= descendants(schema, n)
    return out


def _walk_terms(terms: dict, parent: Optional[str] = None):
    for name, d in (terms or {}).items():
        d = d or {}
        yield name, d, parent
        yield from _walk_terms(d.get("children") or {}, name)


def _described(d: dict) -> str:
    """A term's description as the LLM sees it; merged suggestions ride along as aliases."""
    desc = str(d.get("description") or "")
    aliases = [str(a) for a in _list(d.get("aliases"))]
    return f"{desc} (also: {', '.join(aliases)})" if aliases else desc


def leaf_terms(schema: dict, taxonomy: str) -> Dict[str, str]:
    """{term: description} of the terms an LLM may assign (leaves: group terms only organise)."""
    tax = (schema.get("taxonomies") or {}).get(taxonomy) or {}
    return {n: _described(d) for n, d, _ in _walk_terms(tax.get("terms") or {})
            if not d.get("children")}


def term_parents(schema: dict, taxonomy: str) -> Dict[str, Optional[str]]:
    tax = (schema.get("taxonomies") or {}).get(taxonomy) or {}
    return {n: p for n, _, p in _walk_terms(tax.get("terms") or {})}


def vocab_hash(terms: Dict[str, str]) -> str:
    """Must match tag_docs.py's hash of its vocabulary dict."""
    return hashlib.md5(json.dumps(terms, sort_keys=True).encode()).hexdigest()[:8]


def validate(schema: dict) -> List[str]:
    errs = []
    classes = schema.get("classes") or {}
    if not isinstance(classes, dict) or not classes:
        return ["classes: at least one class is required"]
    for c, d in classes.items():
        d = d or {}
        if not NAME.match(str(c)):
            errs.append(f"class '{c}': names start with a letter and use letters, digits, - or _")
        p = d.get("parent")
        if p and p not in classes:
            errs.append(f"class {c}: parent '{p}' is not a class")
        seen, cur = {c}, p
        while cur:
            if cur in seen:
                errs.append(f"class {c}: its parents form a cycle")
                break
            seen.add(cur)
            cur = (classes.get(cur) or {}).get("parent")
    for c, d in classes.items():
        props = (d or {}).get("properties") or {}
        if not isinstance(props, dict):
            errs.append(f"class {c}: properties must be a mapping of name -> definition")
            continue
        for pn, pd in props.items():
            pd = pd or {}
            if not PROP_NAME.match(str(pn)):
                errs.append(f"class {c}: property '{pn}': use lowercase letters, digits and _")
            if pn in SYSTEM_ATTRS:
                errs.append(f"class {c}: property '{pn}' is a built-in field of every node")
            if pn in (schema.get("relations") or {}):
                # a lookup planner (graph_query.py) confuses the two: "uploaded" the date vs the link
                errs.append(f"class {c}: property '{pn}' has the same name as a relation; give the property its own name")
            if pd.get("type", "string") not in PROP_TYPES:
                errs.append(f"class {c}: property {pn}: type must be one of {', '.join(PROP_TYPES)}")
            if pd.get("type") == "enum" and not (pd.get("values") or pd.get("taxonomy")):
                errs.append(f"class {c}: property {pn}: an enum needs values (or a taxonomy)")
            if pd.get("taxonomy") and pd["taxonomy"] not in (schema.get("taxonomies") or {}):
                errs.append(f"class {c}: property {pn}: taxonomy '{pd['taxonomy']}' doesn't exist")
    for r, d in (schema.get("relations") or {}).items():
        d = d or {}
        if not NAME.match(str(r)):
            errs.append(f"relation '{r}': bad name")
        for side in ("from", "to"):
            v = d.get(side)
            if not v:
                errs.append(f"relation {r}: '{side}' needs at least one class")
            for c in (v if isinstance(v, list) else [v]) if v else []:
                if c not in classes:
                    errs.append(f"relation {r}: {side} class '{c}' is not a class")
    for t, d in (schema.get("taxonomies") or {}).items():
        d = d or {}
        if not NAME.match(str(t)) or " " in str(t):
            errs.append(f"taxonomy '{t}': names start with a letter and use letters, digits, - or _")
        if d.get("assigned_by") not in (None, "", "llm", "author"):
            errs.append(f"taxonomy {t}: assigned_by must be llm or author")
        if not d.get("applies_to"):
            errs.append(f"taxonomy {t}: applies_to needs at least one class")
        mpr = d.get("max_per_record")
        if mpr not in (None, "") and (not isinstance(mpr, int) or isinstance(mpr, bool) or mpr < 1):
            errs.append(f"taxonomy {t}: max_per_record must be a whole number of at least 1")
        # what tag_docs.py can actually do with the two taxonomies it runs
        if t == "doc_types" and mpr not in (None, "", 1):
            errs.append("taxonomy doc_types: the classifier picks exactly one type, so 'up to' must be 1")
        if t in ("topics", "doc_types") and (d.get("assigned_by") or "llm") == "llm" and d.get("applies_to") \
                and not (set(expand(schema, d["applies_to"])) & {"Chat", "File"}):
            errs.append(f"taxonomy {t}: the classifier only reads chats and files, so applies_to must include Chat or File")
        if d.get("assigned_by") == "author" and not d.get("class"):
            errs.append(f"taxonomy {t}: an author-assigned taxonomy needs the class its tags are (e.g. Topic)")
        if d.get("class") and d["class"] not in classes:
            errs.append(f"taxonomy {t}: class '{d['class']}' is not a class")
        for c in d.get("applies_to") or []:
            if c not in classes:
                errs.append(f"taxonomy {t}: applies_to '{c}' is not a class")
        names = [n for n, _, _ in _walk_terms(d.get("terms") or {})]
        dup = [n for n, k in Counter(names).items() if k > 1]
        if dup:
            errs.append(f"taxonomy {t}: duplicate terms {', '.join(dup)}")
        if not names:
            errs.append(f"taxonomy {t}: no terms")
        if not isinstance(d.get("rejected") or [], list):
            errs.append(f"taxonomy {t}: rejected must be a list of names")
        for n, td, _ in _walk_terms(d.get("terms") or {}):
            if not isinstance(td.get("aliases") or [], list):
                errs.append(f"taxonomy {t}: term {n}: aliases must be a list")
    rels = set((schema.get("relations") or {}).keys())
    for r, d in (schema.get("rules") or {}).items():
        d = d or {}
        for side in ("from", "to"):
            if d.get(side) and d[side] not in classes:
                errs.append(f"rule {r}: {side} '{d[side]}' is not a class")
        for path in d.get("via") or []:
            for rel in path or []:
                if rel not in rels:
                    errs.append(f"rule {r}: relation '{rel}' is not defined")
    return errs


# ───────────────────────── measuring a dataset ─────────────────────────

def _graph(export_dir: Path):
    import networkx as nx
    p = export_dir / "storage-metadata" / GRAPH_FILE
    return (nx.read_graphml(p), p.stat().st_mtime) if p.exists() else (None, None)


def _relation_ok(schema: dict, rel: dict, ta: str, tb: str) -> bool:
    """The graph is undirected: an edge conforms if it fits the relation either way round."""
    f, t = expand(schema, _list(rel.get("from"))), expand(schema, _list(rel.get("to")))
    return (ta in f and tb in t) or (tb in f and ta in t)


def _list(v) -> list:
    return v if isinstance(v, list) else ([v] if v else [])


def _rule_pairs(schema: dict, G, rule: dict) -> set:
    T = lambda n: G.nodes[n].get("entity_type")
    starts = [n for n in G if T(n) in descendants(schema, rule.get("from", ""))]
    goal = descendants(schema, rule.get("to", ""))
    pairs = set()
    for path in rule.get("via") or []:
        for s in starts:
            frontier = {s}
            for rel in path:
                frontier = {m for x in frontier for m in G.neighbors(x)
                            if rel in (G.edges[x, m].get("keywords") or "").split(",")}
            pairs |= {(s, e) for e in frontier if T(e) in goal}
    return pairs


def class_properties(schema: dict, cls: str) -> Dict[str, dict]:
    """A class's properties including those inherited from its parents (Actor -> User)."""
    classes, chain, cur = schema.get("classes") or {}, [], cls
    while cur and cur in classes and cur not in chain:
        chain.append(cur)
        cur = (classes[cur] or {}).get("parent")
    out: Dict[str, dict] = {}
    for c in reversed(chain):            # the class's own definition wins over a parent's
        for pn, pd in ((classes[c] or {}).get("properties") or {}).items():
            out[pn] = {**(pd or {}), "defined_on": c}
    return out


def _prop_ok(schema: dict, pd: dict, v) -> bool:
    t = pd.get("type", "string")
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool) or (isinstance(v, str) and v.lstrip("-").isdigit())
    if t == "number":
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False
    if t == "date":
        return bool(DATE.match(str(v)))
    if t == "boolean":
        return isinstance(v, bool) or str(v).lower() in ("true", "false")
    if t == "enum":
        allowed = set(_list(pd.get("values")))
        if pd.get("taxonomy"):
            allowed |= set(leaf_terms(schema, pd["taxonomy"]))
        return str(v) in allowed
    return True


def _measure_properties(schema: dict, G) -> dict:
    """Per class: for each declared property, how many of the class's nodes carry it and how
    many values have the wrong type; plus attributes nodes carry that the schema doesn't declare."""
    out = {}
    if G is None:
        return out
    by_type = defaultdict(list)
    for n, d in G.nodes(data=True):
        by_type[d.get("entity_type")].append(d)
    for c in schema.get("classes") or {}:
        nodes = by_type.get(c, [])
        props = class_properties(schema, c)
        if not nodes and not props:
            continue
        res, undeclared = {}, Counter()
        for pn, pd in props.items():
            have = [d[pn] for d in nodes if pn in d and d[pn] not in (None, "")]
            bad = [v for v in have if not _prop_ok(schema, pd, v)]
            res[pn] = {"count": len(have), "total": len(nodes), "bad": len(bad),
                       "bad_examples": [str(v) for v in bad[:3]],
                       "examples": sorted({str(v) for v in have}, key=str)[:6],
                       "defined_on": pd["defined_on"]}
        for d in nodes:
            for k in d:
                if k not in SYSTEM_ATTRS and k not in props:
                    undeclared[k] += 1
        out[c] = {"properties": res, "undeclared": dict(undeclared), "nodes": len(nodes)}
    return out


def covered_names(schema: dict, cls: str) -> set:
    """Every term name and alias (lowercase) of the taxonomies whose terms are `cls` nodes."""
    out = set()
    for t, d in (schema.get("taxonomies") or {}).items():
        if (d or {}).get("class") == cls:
            out |= set(all_term_names(schema, t))
    return out


def _measure_author(schema: dict, name: str, tax: dict, G) -> dict:
    """Coverage of a taxonomy whose terms are set by item authors (assigned_by: author), not by
    the LLM: each applies_to item's tags in the graph (its `about` edges to `class` nodes,
    e.g. Hub manifest tags) are matched to terms by name or alias. Nothing to re-run, so
    never stale; tags no term or alias covers are listed as not_in_schema."""
    terms, parents = leaf_terms(schema, name), term_parents(schema, name)
    resolve = all_term_names(schema, name)
    cls, kinds = tax.get("class") or "Topic", expand(schema, tax.get("applies_to"))
    items = [n for n, d in G.nodes(data=True) if d.get("entity_type") in kinds] if G is not None else []
    # tags rejected in any taxonomy on the same class ("never a term") aren't listed as uncovered
    rejected = {slug(r) for t in (schema.get("taxonomies") or {}).values() if (t or {}).get("class") == cls
                for r in _list((t or {}).get("rejected"))}
    counts, unmatched, assigned = Counter(), Counter(), 0
    for n in items:
        got = set()
        for m in G.neighbors(n):
            if G.nodes[m].get("entity_type") != cls or "about" not in (G.edges[n, m].get("keywords") or "").split(","):
                continue
            tag = str(m).split(": ", 1)[-1]
            term = resolve.get(tag.lower()) or resolve.get(slug(tag))
            if term:
                got.add(term)
            elif slug(tag) not in rejected:
                unmatched[tag] += 1
        counts.update(got)
        assigned += bool(got)
    group_counts = Counter()
    for t, k in counts.items():
        p = parents.get(t)
        while p:
            group_counts[p] += k
            p = parents.get(p)
    return {"records": len(items), "assigned": assigned, "assigned_by": "author",
            "terms": {t: counts.get(t, 0) for t in parents if t in terms},
            "groups": {g: group_counts.get(g, 0) for g in parents if g not in terms},
            "not_in_schema": dict(unmatched), "graph_terms_not_in_schema": {},
            "current_vocab": vocab_hash(terms), "tagged_vocab": None, "suggested": 0, "stale": False}


def measure(schema: dict, export_dir: Path) -> dict:
    """How a dataset's metadata graph and tags fit the schema: instance counts per class and
    relation, edges that break a relation's from/to, node and edge types the schema doesn't
    define, taxonomy coverage (and whether the dataset was tagged with the current terms),
    glossary terms' doc counts, and how many edges each rule would derive."""
    classes, relations = schema.get("classes") or {}, schema.get("relations") or {}
    G, built = _graph(export_dir)
    tags_path = export_dir / "tags.json"
    tags = json.loads(tags_path.read_text(encoding="utf-8")) if tags_path.exists() else {}
    docs_dir = export_dir / "docs"
    doc_files = sorted(p.name for p in docs_dir.glob("*.md")) if docs_dir.is_dir() else []
    out: dict = {"graph_built_at": built, "tagged_at": tags_path.stat().st_mtime if tags_path.exists() else None,
                 "has_graph": G is not None, "docs": len(doc_files)}

    # classes
    by_type = Counter(d.get("entity_type") for _, d in G.nodes(data=True)) if G is not None else Counter()
    cls = {}
    for c in classes:
        direct = by_type.get(c, 0)
        cls[c] = {"count": direct, "total": sum(by_type.get(x, 0) for x in descendants(schema, c))}
    out["classes"] = cls
    out["unknown_classes"] = {t: n for t, n in by_type.items() if t not in classes}

    # relations
    rel = {r: {"count": 0, "violations": 0, "examples": []} for r in relations}
    unknown_rel = Counter()
    pairs = defaultdict(Counter)        # relation -> "TypeA→TypeB" counts (for the map)
    if G is not None:
        for a, b, d in G.edges(data=True):
            ta, tb = G.nodes[a].get("entity_type"), G.nodes[b].get("entity_type")
            for r in filter(None, (d.get("keywords") or "").split(",")):
                if r not in relations:
                    unknown_rel[r] += 1
                    continue
                rel[r]["count"] += 1
                spec = relations[r] or {}
                fr = expand(schema, _list(spec.get("from")))
                pairs[r][f"{ta}→{tb}" if ta in fr else f"{tb}→{ta}"] += 1
                if not _relation_ok(schema, spec, ta, tb):
                    rel[r]["violations"] += 1
                    if len(rel[r]["examples"]) < 3:
                        rel[r]["examples"].append(f"{a} ({ta}) — {b} ({tb})")
        for r in rel:
            rel[r]["pairs"] = dict(pairs[r])
    out["relations"] = rel
    out["unknown_relations"] = dict(unknown_rel)
    out["nodes"], out["edges"] = (G.number_of_nodes(), G.number_of_edges()) if G is not None else (0, 0)

    # taxonomies
    tax_out = {}
    for name, tax in (schema.get("taxonomies") or {}).items():
        tax = tax or {}
        terms = leaf_terms(schema, name)
        parents = term_parents(schema, name)
        if (tax.get("assigned_by") or "llm") == "author":
            tax_out[name] = _measure_author(schema, name, tax, G)
            continue
        prefixes = [DOC_PREFIX[c] for c in expand(schema, tax.get("applies_to")) if c in DOC_PREFIX]
        records = [d for d in doc_files if any(d.startswith(p) for p in prefixes)]
        if name == "topics":
            assigned = {d: tags.get(d) or [] for d in records}
            recorded = tags.get("_vocab")
        else:
            assigned = {d: _list((tags.get(f"_{name}") or {}).get(d)) for d in records}
            recorded = tags.get(f"_{name}_vocab")
        counts = Counter(t for v in assigned.values() for t in v)
        # Topic nodes in the graph that are not schema terms (e.g. Hub items' free-form tags)
        off = {}
        if tax.get("class") and G is not None:
            # a tag counts as known once ANY taxonomy on the same class names it (as a term
            # or an alias), e.g. Hub tags covered by an author-assigned "capabilities"
            known = covered_names(schema, tax["class"])
            for n, d in G.nodes(data=True):
                if d.get("entity_type") == tax["class"]:
                    term = str(n).split(": ", 1)[-1]
                    if term.lower() not in known and slug(term) not in known:
                        off[term] = G.degree(n)
        group_counts = Counter()
        for t, k in counts.items():
            p = parents.get(t)
            while p:
                group_counts[p] += k
                p = parents.get(p)
        tax_out[name] = {
            "records": len(records), "assigned": sum(1 for v in assigned.values() if v),
            "terms": {t: counts.get(t, 0) for t in parents if t in terms},
            "groups": {g: group_counts.get(g, 0) for g in parents if g not in terms},
            "not_in_schema": {t: k for t, k in counts.items() if t not in terms},
            "graph_terms_not_in_schema": off,
            "current_vocab": vocab_hash(terms), "tagged_vocab": recorded,
            # records whose classifier reply said "nothing fits well" and proposed a term
            "suggested": sum(1 for c in (tags.get("_candidates") or {}).values()
                             if any(x.get("taxonomy") == name for x in c)),
            "stale": bool(records) and recorded != vocab_hash(terms),
        }
    out["taxonomies"] = tax_out

    # glossary: how many docs use each term (or a synonym)
    texts = {}
    for d in doc_files:
        try:
            texts[d] = (docs_dir / d).read_text(encoding="utf-8").lower()
        except OSError:
            pass
    gl = {}
    for term, d in (schema.get("glossary") or {}).items():
        words = [str(term)] + [str(s) for s in _list((d or {}).get("synonyms"))]
        rx = re.compile("|".join(r"(?<![\w-])" + re.escape(w.lower()) + r"(?![\w-])" for w in words if w))
        gl[term] = sum(1 for t in texts.values() if rx.search(t))
    out["glossary"] = gl

    # rules
    rules = {}
    for r, d in (schema.get("rules") or {}).items():
        p = _rule_pairs(schema, G, d or {}) if G is not None else set()
        rules[r] = {"edges": len(p), "sample": sorted(f"{a} → {str(b).split(': ', 1)[-1]}" for a, b in p)[:8]}
    out["rules"] = rules
    out["violations"] = sum(v["violations"] for v in rel.values())
    out["properties"] = _measure_properties(schema, G)
    out["property_violations"] = sum(p["bad"] for c in out["properties"].values() for p in c["properties"].values())
    return out


def snapshot(m: dict) -> dict:
    """The parts of a measurement worth tracking over time (one row per ingest)."""
    return {"graph_built_at": m.get("graph_built_at"), "tagged_at": m.get("tagged_at"),
            "measured_at": time.time(), "docs": m.get("docs"), "nodes": m.get("nodes"), "edges": m.get("edges"),
            "violations": m.get("violations"),
            "classes": {c: v["count"] for c, v in (m.get("classes") or {}).items() if v["count"]},
            "relations": {r: v["count"] for r, v in (m.get("relations") or {}).items() if v["count"]},
            "taxonomies": {t: {"assigned": v["assigned"], "records": v["records"], "suggested": v.get("suggested", 0),
                               "terms": {k: n for k, n in v["terms"].items() if n}}
                           for t, v in (m.get("taxonomies") or {}).items()}}


def track(export_dir: Path, m: dict) -> List[dict]:
    """Append a snapshot when the dataset was rebuilt or re-tagged since the last one
    (so each ingest is one point), and return the history, oldest first."""
    p = export_dir / "schema_metrics.json"
    hist = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    last = hist[-1] if hist else {}
    if m.get("has_graph") and (last.get("graph_built_at") != m.get("graph_built_at")
                               or last.get("tagged_at") != m.get("tagged_at")):
        hist.append(snapshot(m))
        p.write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return hist


# ───────────────────────── bottom-up vocabulary growth ─────────────────────────
#
# tag_docs.py lets the LLM say "no allowed term covers this record's main subject" and propose
# one, built from words in the record. Those proposals land in tags.json "_candidates". Here
# they are grouped across every export, checked against the schema (already a term or alias?
# rejected before? which existing term is nearest?) and given a support count. A person
# approves (new term), merges (alias of an existing term) or rejects (never suggested again);
# tag_docs.py --incremental then re-checks only the records the change can affect.

MIN_RECORDS = 2       # distinct records before a suggestion is "ready" for review
MIN_USERS = 1         # distinct authors (raise once datasets have many users)
OVERLOADED = 0.4      # a term on more than this share of records may need splitting
_STOP = {"and", "the", "for", "with", "from", "into", "of", "a", "an", "or", "to", "in", "on", "by"}


def slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")[:48]


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "s", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _tokens(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2 and w not in _STOP}


def _similar(a: str, b: str) -> bool:
    """Two suggested names that mean the same term (detection-rule / detection-rules,
    after-action-report / after-action-review)."""
    ta, tb = _tokens(a.replace("-", " ")), _tokens(b.replace("-", " "))
    if not ta or not tb:
        return False
    shared = ta & tb
    return ta == tb or (len(shared) >= 2 and len(shared) / len(ta | tb) >= 0.5)


def all_term_names(schema: dict, taxonomy: str) -> Dict[str, str]:
    """{name or alias (lowercase): the term it resolves to}, group terms included."""
    tax = (schema.get("taxonomies") or {}).get(taxonomy) or {}
    out = {}
    for n, d, _ in _walk_terms(tax.get("terms") or {}):
        out[str(n).lower()] = n
        for a in _list(d.get("aliases")):
            out[slug(a)] = n
    return out


def _doc_title(export_dir: Path, doc: str) -> str:
    try:
        with (export_dir / "docs" / doc).open(encoding="utf-8") as f:
            return re.sub(r"^#\s*", "", f.readline()).strip()[:80] or doc
    except OSError:
        return doc


def suggestions(schema: dict, exports: List[Path], export_dir: Optional[Path] = None) -> dict:
    """Grouped, schema-checked term proposals from every export's tags.json, plus term-health
    advisories and the Hub's free-form tags for the selected export."""
    taxes = schema.get("taxonomies") or {}
    try:
        cmap = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8")) if CANDIDATES_PATH.exists() else {}
    except ValueError:
        cmap = {}
    evidence = defaultdict(dict)          # (taxonomy, slug) -> {doc: evidence}; export copies dedupe by doc id
    descs = defaultdict(Counter)
    for ex in exports:
        p = ex / "tags.json"
        if not p.exists():
            continue
        try:
            tags = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for doc, cands in (tags.get("_candidates") or {}).items():
            for c in cands or []:
                t, name = c.get("taxonomy"), slug(c.get("term"))
                if t not in taxes or not name:
                    continue
                descs[(t, name)][str(c.get("description") or "").strip()] += 1
                evidence[(t, name)].setdefault(doc, {
                    "doc": doc, "dataset": ex.name, "title": _doc_title(ex, doc),
                    "author": c.get("author"), "picked": c.get("picked") or []})

    groups = []
    for t in taxes:
        names = all_term_names(schema, t)
        rejected = {slug(r) for r in _list(taxes[t].get("rejected"))}
        leaves = leaf_terms(schema, t)
        parents = term_parents(schema, t)
        tmap = cmap.get(t) or {}
        pending = sorted((n for (tt, n) in evidence if tt == t), key=lambda n: -len(evidence[(t, n)]))
        # the LLM grouping first (labels it saw); shared words for labels it hasn't seen yet
        by_group: Dict[str, List[str]] = defaultdict(list)
        for n in pending:
            if n in tmap:
                by_group[tmap[n]["group"]].append(n)
        clusters: List[List[str]] = list(by_group.values())
        for n in pending:
            if n in tmap:
                continue
            for cl in clusters:
                if any(_similar(n, m) for m in cl):
                    cl.append(n)
                    break
            else:
                clusters.append([n])
        for cl in clusters:
            # already resolved: approved as a term, merged as an alias, or rejected (the group's
            # canonical name counts too: approving it resolves every member)
            gname = next((tmap[n]["group"] for n in cl if n in tmap), None)
            if any(n in names or n in rejected for n in cl) or (gname and (gname in names or gname in rejected)):
                continue
            ev = {}
            for n in cl:
                ev.update(evidence[(t, n)])
            g0 = next((tmap[n] for n in cl if n in tmap), None)
            if g0 and g0["group"] not in names and g0["group"] not in rejected:
                label, desc = g0["group"], g0.get("description") or ""
            else:
                label = max(cl, key=lambda n: (len(evidence[(t, n)]), -len(n)))
                desc = (descs[(t, label)].most_common(1) or [("", 0)])[0][0]
            same_as = g0.get("same_as") if g0 and g0.get("same_as") in leaves else None
            # nearest existing term by shared words (name + description + aliases)
            want = _tokens(" ".join(cl).replace("-", " ") + " " + desc)
            near, score = None, 0.0
            for term, d in leaves.items():
                have = _tokens(term.replace("-", " ") + " " + d)
                sc = len(want & have) / max(1, len(want | have))
                if sc > score:
                    near, score = term, sc
            if same_as:                   # the LLM says it is an existing term: recommend merging
                near, score = same_as, 1.0
            users = {e["author"] for e in ev.values() if e.get("author")}
            groups.append({
                "taxonomy": t, "term": label, "variants": sorted(set(cl) - {label}), "description": desc,
                "same_as": same_as,
                "records": len(ev), "users": sorted(users), "evidence": sorted(ev.values(), key=lambda e: e["doc"]),
                "nearest": near, "nearest_score": round(score, 2),
                # the grouping step's placement first, else the nearest term's group
                "parent": (g0.get("parent") if g0 and g0.get("parent") in parents
                           else parents.get(near) if near and score >= 0.08 else None),
                "ready": len(ev) >= MIN_RECORDS and len(users) >= MIN_USERS,
            })
    groups.sort(key=lambda g: (not g["ready"], -g["records"], g["term"]))

    health, hub_tags = [], {}
    if export_dir is not None:
        m = measure(schema, export_dir)
        for t, v in m["taxonomies"].items():
            n = v["records"]
            if v["graph_terms_not_in_schema"]:
                rejected = {slug(r) for r in _list(taxes[t].get("rejected"))}
                hub_tags[t] = {k: c for k, c in v["graph_terms_not_in_schema"].items() if slug(k) not in rejected}
            if not n or v["stale"]:
                continue
            for term, k in v["terms"].items():
                if term in ("off-topic", "other"):
                    continue
                if n >= 8 and k / n > OVERLOADED:
                    health.append({"taxonomy": t, "term": term, "kind": "overloaded", "count": k, "records": n,
                                   "note": f"on {k} of {n} records: consider splitting it into narrower terms"})
                elif k == 0:
                    health.append({"taxonomy": t, "term": term, "kind": "unused", "count": 0, "records": n,
                                   "note": "no record in this dataset uses it: keep it, or retire it"})
            if v.get("suggested"):
                health.append({"taxonomy": t, "term": None, "kind": "gaps", "count": v["suggested"], "records": n,
                               "note": f"{v['suggested']} of {n} records said no {t} term covers their main subject"})
    return {"thresholds": {"min_records": MIN_RECORDS, "min_users": MIN_USERS, "overloaded": OVERLOADED},
            "groups": groups, "health": health, "hub_tags": hub_tags}


def _find_term(terms: dict, name: str) -> Optional[dict]:
    for n, d, _ in _walk_terms(terms):
        if n == name:
            return d
    return None


def approve_term(schema: dict, taxonomy: str, term: str, description: str = "",
                 parent: Optional[str] = None, aliases: Optional[list] = None) -> dict:
    """The schema with a reviewed suggestion added as a term (under a group term, or top level)."""
    s = json.loads(json.dumps(schema))
    tax = (s.get("taxonomies") or {}).get(taxonomy)
    if tax is None:
        raise ValueError(f"no taxonomy {taxonomy}")
    term = slug(term)
    if not term:
        raise ValueError("empty term name")
    if term in all_term_names(s, taxonomy):
        raise ValueError(f"{term} is already a term or alias in {taxonomy}")
    entry: dict = {"description": (description or "").strip()}
    al = sorted({slug(a) for a in aliases or []} - {"", term})
    if al:
        entry["aliases"] = al
    terms = tax.setdefault("terms", {})
    if parent:
        p = _find_term(terms, parent)
        if p is None:
            raise ValueError(f"no term {parent} in {taxonomy}")
        if not p.get("children"):
            raise ValueError(f"{parent} is a leaf term; pick a group term as the parent")
        p["children"][term] = entry
    else:
        # keep the catch-all terms last
        tail = {k: terms.pop(k) for k in [k for k in list(terms) if k in ("other", "off-topic")]}
        terms[term] = entry
        terms.update(tail)
    return s


def merge_terms(schema: dict, taxonomy: str, names: list, into: str) -> dict:
    """The schema with suggested names added as aliases of an existing term."""
    s = json.loads(json.dumps(schema))
    tax = (s.get("taxonomies") or {}).get(taxonomy) or {}
    d = _find_term(tax.get("terms") or {}, into)
    if d is None:
        raise ValueError(f"no term {into} in {taxonomy}")
    known = all_term_names(s, taxonomy)
    new = {slug(n) for n in names} - set(known) - {""}
    if not new:
        raise ValueError("nothing to merge: every name is already a term or alias")
    d["aliases"] = sorted(set(_list(d.get("aliases"))) | new)
    return s


def reject_terms(schema: dict, taxonomy: str, names: list) -> dict:
    """The schema with names added to the taxonomy's rejected list (never suggested again)."""
    s = json.loads(json.dumps(schema))
    tax = (s.get("taxonomies") or {}).get(taxonomy)
    if tax is None:
        raise ValueError(f"no taxonomy {taxonomy}")
    tax["rejected"] = sorted(set(_list(tax.get("rejected"))) | {slug(n) for n in names if slug(n)})
    return s
