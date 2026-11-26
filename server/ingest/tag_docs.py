"""
Tag chats and files from a FIXED vocabulary with an LLM, so the metadata graph's
Topic nodes link related work across users and teams.

Open WebUI's own auto-tags are free-form ("technology", "general", one-off
labels per chat), so they rarely connect two people's chats. Here the model may
only pick from the agreed list, which makes topics comparable everywhere. Users
do nothing; change the list and re-run to retag.

  reads   <export>/docs/chat-*.md, file-*.md  (export_to_docs.py output)
          <export>/openwebui/chat_review.json  (summaries, when present)
          the semantic model (server/datasets/schema/schema.yaml, edited in Prospector):
            taxonomies.topics     -> the tags (leaf terms; group terms only organise)
            taxonomies.doc_types  -> one document type per record it applies_to (files)
          <export>/tag_vocab.json              (optional topic override; else the schema,
                                                else the default below)
  writes  <export>/tags.json                   {"chat-<id>.md": ["detection", ...], ...,
                                                "_doc_types": {"file-<id>.md": "runbook"},
                                                "_candidates": {doc: [suggested term, ...]}}

Bottom-up growth: before picking from the list, the model names the record's subject (and,
for files, its document type) in its own words. Asking it to judge its own fit doesn't work
(it rates every pick "good"), so the gap test is deterministic: a topic label that shares no
meaningful word with ANY topic term, or a type label that shares none with the type it was
given, becomes a candidate term. The record still gets the closest existing terms. Labels
are cached with the tags, so candidates are recomputed against the current schema on every
run at no model cost (an approved term stops being a candidate by itself).

Labels are free text, so the same gap comes back worded differently ("exercise-reporting",
"exercise-after-action"); neither shared words nor label embeddings group those reliably.
So after a run that produced new labels, the model groups ALL open labels across every
export (one call per taxonomy), naming each group, describing it, and saying when a group
just means an existing term. That map is datasets/schema/candidates.json; Prospector >
Semantic model > suggestions reads it to build the review queue (approve / merge / reject).

build_metadata_graph.py turns tags.json into `about` edges to Topic nodes.
Results are cached by doc content hash in tags.json, so a re-run only asks the
model about new or changed docs (and everything when the vocabulary changes).

--incremental: after a vocabulary change, compare the terms the dataset was tagged with
against the schema's. Removed terms are dropped from records (no model call). Added terms,
and terms whose description changed, are re-checked on the likely records only (the records
whose own label became the term, plus the records whose text is nearest its description):
the record is classified again with the full new list, and only that answer's decision about
the changed term is applied, so the new term has to beat the existing ones (asked alone,
"is this about X?" gets a yes for nearly anything). Every other tag is kept as it is. Falls
back to a full re-tag for a taxonomy whose previous vocabulary was not recorded.

Usage:
  python tag_docs.py <export_dir> [--model gemma3:12b] [--force | --incremental]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_to_docs import load  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import schema_model as SM   # the shared semantic model (needs PyYAML)
except Exception:               # noqa: BLE001 - tagging still works with the defaults
    SM = None

HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
MODEL = os.getenv("PROTOGRAPH_LLM", "gemma3:12b")
EMBED_MODEL = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")
DOC_CHARS = 6000      # of each doc shown to the model
MAX_TAGS = 4

DEFAULT_VOCAB = {
    "range-infrastructure": "Proxmox, hypervisors, snapshots, revert, storage, VM/enclave performance",
    "active-directory": "domains, trust relationships, Kerberos, time sync, machine accounts, GPO",
    "automation": "Ansible, Terraform, Packer, CI/CD pipelines, scripted deployment, WinRM tooling",
    "detection": "Sysmon, Windows event IDs, SIEM, EDR, blue-team visibility and alerting",
    "adversary-emulation": "ATT&CK techniques, lateral movement, phishing, C2, red-team tradecraft",
    "scenario-design": "exercise scenarios, MSELs, injects, storylines, exercise timelines",
    "exercise-rules": "rules of engagement, scope, stop words, maintenance windows, exercise policy",
    "content-seeding": "planting emails, files and artifacts into enclaves for an exercise",
    "troubleshooting": "something was broken and the conversation works out why or how to fix it",
    "off-topic": "greetings, small talk, requests unrelated to the team's work",
}

PROMPT = """You label workplace documents for a search index. Pick the tags that describe what this {kind} is substantially about.

ALLOWED TAGS (use only these exact names):
{vocab}

Rules:
- Choose 1 to {max_tags} tags, most relevant first. Only tag a topic the {kind} actually spends real content on, not one it mentions in passing.
- "off-topic" only when there is no work content at all; never combine it with other tags.
- "troubleshooting" only when a concrete problem is being diagnosed or fixed.
First, before choosing tags, name in your own words what this {kind} is mainly about, as the topic a team would file it under: 1-3 lowercase-hyphenated words, e.g. "phishing-campaign-planning", "vpn-troubleshooting", "firewall-change-review". Not a specific exercise, host, person or product name, and not limited to the tag list.

{kind_upper}:
{text}

Reply with JSON only: {{"subject": "own-words-topic", "tags": ["tag", ...]}}"""


TYPE_PROMPT = """You classify workplace documents for a search index. Which ONE type is this {kind}?

TYPES (use only these exact names):
{vocab}

First, name in your own words what kind of document this is: 1-3 lowercase-hyphenated words, e.g. "meeting-minutes", "network-diagram", "ansible-playbook". Not limited to the list. Then pick the closest TYPE from the list.

{kind_upper}:
{text}

Reply with JSON only: {{"own_type": "own-words-type", "type": "name"}}"""



def schema_vocabs():
    """(topics, doc_types, kinds doc types apply to) from the semantic model; empty without it."""
    if SM is None:
        return {}, {}, set()
    try:
        schema, _ = SM.load()
    except Exception as e:   # noqa: BLE001
        print(f"[tags] schema.yaml unreadable ({e}); using the default vocabulary")
        return {}, {}, set()
    dt = (schema.get("taxonomies") or {}).get("doc_types") or {}
    kinds = {c.lower() for c in SM.expand(schema, dt.get("applies_to") or [])}
    return SM.leaf_terms(schema, "topics"), SM.leaf_terms(schema, "doc_types"), kinds


def topic_rules():
    """(most topics per record, record kinds that get topics) from taxonomies.topics:
    max_per_record and applies_to. Without the schema: 4, chats and files."""
    if SM is None:
        return 4, {"chat", "file"}
    try:
        schema, _ = SM.load()
    except Exception:   # noqa: BLE001
        return 4, {"chat", "file"}
    tp = (schema.get("taxonomies") or {}).get("topics") or {}
    n = tp.get("max_per_record")
    kinds = {c.lower() for c in SM.expand(schema, tp.get("applies_to") or ["Chat", "File"])} & {"chat", "file"}
    return (n if isinstance(n, int) and n >= 1 else 4), kinds


def _post(client: httpx.Client, model: str, prompt: str, num_predict: int):
    """The model's replies, up to 3 attempts (the caller stops at the first usable one)."""
    for _ in range(3):
        try:   # the model host may be busy (an index build shares it): wait, then retry once more
            r = client.post(f"{HOST}/api/chat", json={
                # no format="json": gemma3 can loop forever under Ollama's JSON grammar,
                # hanging the model for every other client. Plain text, capped, parsed by the caller.
                "model": model, "stream": False,
                "options": {"temperature": 0, "num_ctx": 16384, "num_predict": num_predict},
                "messages": [{"role": "user", "content": prompt}]})
        except httpx.TimeoutException:
            continue
        r.raise_for_status()
        yield r.json()["message"]["content"]


def _named(vocab: dict, reply: str) -> list:
    """Allowed names the reply spelled out, in vocabulary order."""
    return [t for t in vocab if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", reply.lower())]


def _json(reply: str) -> dict:
    """The outermost {...} in a reply (suggestions nest an object, so not the first '}')."""
    a, b = reply.find("{"), reply.rfind("}")
    if a < 0 or b <= a:
        return {}
    try:
        v = json.loads(reply[a:b + 1])
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


# words too common in this domain's term descriptions to show that a label is covered
GENERIC = {"exercise", "team", "work", "range", "training", "general", "document", "file", "chat",
           "management", "process", "task", "data", "information", "content"}


def _label(raw) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(raw or "").lower()).strip("-")[:48]


def gap(label: str, covering: dict, taken: set):
    """A candidate term when the model's own label for a record shares no meaningful word with
    the covering terms (name + description + aliases); None when covered or unusable."""
    if SM is None or not label or label in taken or label in ("none", "null", "other", "off-topic", "misc"):
        return None
    words = SM._tokens(label.replace("-", " ")) - GENERIC
    if not words:
        return None
    have = set()
    for t, d in covering.items():
        have |= SM._tokens(t.replace("-", " ") + " " + d)
    if words & have:
        return None
    return {"term": label, "description": label.replace("-", " ")}


def ask_type(client: httpx.Client, model: str, kind: str, text: str, vocab: dict):
    """(type, the model's own label for the type)"""
    prompt = TYPE_PROMPT.format(kind=kind, kind_upper=kind.upper(), text=text,
                                vocab="\n".join(f"- {k}: {v}" for k, v in vocab.items()))
    for reply in _post(client, model, prompt, 120):
        j = _json(reply)
        t = str(j.get("type") or "").strip().lower()
        if t not in vocab:
            t = next(iter(_named(vocab, reply.split('"type"', 1)[-1])), "")
        if t:
            return t, _label(j.get("own_type"))
    return "", ""


def summaries(ex: Path) -> dict:
    p = ex / "openwebui" / "chat_review.json"
    out = {}
    for r in (load(p) if p.exists() else []):
        if r.get("summary"):
            out[f"chat-{r['chat_id']}.md"] = r["summary"]
    return out


def ask(client: httpx.Client, model: str, kind: str, text: str, vocab: dict):
    """(tags, the model's own label for the subject)"""
    prompt = PROMPT.format(kind=kind, kind_upper=kind.upper(), max_tags=MAX_TAGS, text=text,
                           vocab="\n".join(f"- {k}: {v}" for k, v in vocab.items()))
    for reply in _post(client, model, prompt, 200):
        j = _json(reply)
        tags = j.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        if not tags:   # fall back to any allowed tag names the reply spelled out
            tags = _named(vocab, reply.split('"tags"', 1)[-1])
        if not tags:
            continue
        tags = [t for t in dict.fromkeys(str(t).strip().lower() for t in tags) if t in vocab][:MAX_TAGS]
        if "off-topic" in tags and len(tags) > 1:
            tags.remove("off-topic")
        if tags:
            return tags, _label(j.get("subject"))
    return [], ""


GROUP_PROMPT = """A document classifier proposed these new {taxonomy} names for records that no existing term covered.
Group the proposed names that mean the same thing. Give each group a short canonical lowercase-hyphenated name (general enough to cover all its members; avoid names of specific exercises, hosts, people or products) and a one-line description of which records it should cover.
If a group means the same as one of the EXISTING terms, set "same_as" to that term; otherwise null.{parent_rule}

EXISTING TERMS:
{vocab}

PROPOSED NAMES (with the start of an example record each; group by what the records are, not only by how the name is worded):
{labels}

Every proposed name must appear in exactly one group. Reply with JSON only:
{{"groups": [{{"name": "canonical-name", "description": "...", "members": ["proposed-name", ...], "same_as": null{parent_field}}}]}}"""


def consolidate(client: httpx.Client, model: str, ex: Path, vocabs: dict) -> None:
    """Group every export's open candidate labels per taxonomy (LLM, one call per taxonomy
    with new labels) into datasets/schema/candidates.json."""
    if SM is None:
        return
    path = SM.CANDIDATES_PATH
    try:
        cmap = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        cmap = {}
    labels: dict = {}                     # taxonomy -> {label: example title}
    for tj in sorted(ex.parent.glob("*/tags.json")):
        try:
            cands = json.loads(tj.read_text(encoding="utf-8")).get("_candidates") or {}
        except ValueError:
            continue
        for doc, cs in cands.items():
            for c in cs:
                if c.get("taxonomy") in vocabs and c.get("term"):
                    try:
                        body = (tj.parent / "docs" / doc).read_text(encoding="utf-8")
                        head, _, rest = body.partition("\n")
                        # title + a short excerpt, so records are grouped by what they are, not only by label
                        title = head.lstrip("# ")[:60] + " | " + re.sub(r"\s+", " ", rest).strip()[:180]
                    except OSError:
                        title = doc
                    labels.setdefault(c["taxonomy"], {}).setdefault(c["term"], title)
    for tax, found in labels.items():
        known = cmap.get(tax) or {}
        if set(found) <= set(known):
            continue                      # nothing new since the last grouping
        vocab = vocabs[tax]
        # group terms of the taxonomy (topics: range-operations, ...) so a new term gets a place
        try:
            schema_now, _ = SM.load()
            tdef = (schema_now.get("taxonomies") or {}).get(tax) or {}
            gterms = {n: str(d.get("description") or "") for n, d, _ in SM._walk_terms(tdef.get("terms") or {})
                      if d.get("children")}
        except Exception:   # noqa: BLE001
            gterms = {}
        parent_rule = ('\nAlso set "parent" to the GROUP it belongs under, or null if none fits.\nGROUPS:\n'
                       + "\n".join(f"- {k}: {v}" for k, v in gterms.items())) if gterms else ""
        prompt = GROUP_PROMPT.format(taxonomy=tax, vocab="\n".join(f"- {k}: {v}" for k, v in vocab.items()),
                                     parent_rule=parent_rule, parent_field=', "parent": null' if gterms else "",
                                     labels="\n".join(f'- {k}: "{t}"' for k, t in sorted(found.items())))
        out = {}
        for reply in _post(client, model, prompt, 900):
            for g in (_json(reply).get("groups") or []):
                if not isinstance(g, dict):
                    continue
                name = _label(g.get("name"))
                same = g.get("same_as") if g.get("same_as") in vocab else None
                parent = g.get("parent") if g.get("parent") in gterms else None
                for m_ in g.get("members") or []:
                    m_ = _label(m_)
                    if m_ in found and m_ not in out and name:
                        out[m_] = {"group": name, "description": str(g.get("description") or "").strip()[:200],
                                   "same_as": same, "parent": parent}
            if out:
                break
        for m_ in found:                  # anything the model left out stands alone
            out.setdefault(m_, {"group": m_, "description": m_.replace("-", " "), "same_as": None})
        cmap[tax] = out
        groups = {}
        for m_, g in out.items():
            groups.setdefault(g["group"], []).append(m_)
        print(f"[tags] grouped {len(found)} open {tax} labels into {len(groups)}: "
              + "; ".join(f"{g} <- {', '.join(ms)}" for g, ms in groups.items()))
    if labels:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cmap, indent=1), encoding="utf-8")


def owners(ex: Path) -> dict:
    """{doc name: user-NN} from the export tables (chats' and files' user_id; pseudonyms only),
    so suggestion support can count distinct people."""
    out = {}
    for table, prefix in (("chat", "chat-"), ("file", "file-")):
        p = ex / "openwebui" / f"{table}.json"
        for r in (load(p) if p.exists() else []):
            if r.get("id") and str(r.get("user_id") or "").startswith("user-"):
                out[f"{prefix}{r['id']}.md"] = r["user_id"]
    return out


def _embed(client: httpx.Client, texts: list):
    """Embeddings for the incremental candidate prefilter (None when the host can't)."""
    try:
        r = client.post(f"{HOST}/api/embed", json={"model": EMBED_MODEL, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:   # noqa: BLE001
        print(f"[tags] embeddings unavailable ({e}); checking every record")
        return None


def _cos(a, b) -> float:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    ap.add_argument("--incremental", action="store_true",
                    help="after a vocabulary change, re-check only the records added/changed terms can affect")
    a = ap.parse_args()
    ex = Path(a.export_dir).resolve()
    s_topics, type_vocab, type_kinds = schema_vocabs()
    # the schema's "up to N per record" and "applies to" for topics (doc types: always exactly one)
    global MAX_TAGS
    MAX_TAGS, topic_kinds = topic_rules()
    print(f"[tags] topics: up to {MAX_TAGS} per record, on {', '.join(sorted(topic_kinds)) or 'nothing'}")
    vocab = load(ex / "tag_vocab.json") if (ex / "tag_vocab.json").exists() else (s_topics or DEFAULT_VOCAB)
    type_hash = hashlib.md5(json.dumps(type_vocab, sort_keys=True).encode()).hexdigest()[:8]
    vocab_hash = hashlib.md5(json.dumps(vocab, sort_keys=True).encode()).hexdigest()[:8]
    out_path = ex / "tags.json"
    old = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and not a.force else {}
    if a.incremental:
        with httpx.Client(timeout=180) as client:
            old = incremental(client, a.model, ex, old, vocab, vocab_hash, type_vocab, type_hash, type_kinds)
    cache = old.get("_cache", {}) if old.get("_vocab") == vocab_hash else {}
    type_cache = old.get("_doc_types_cache", {}) if old.get("_doc_types_vocab") == type_hash else {}
    summ = summaries(ex)
    owner = owners(ex)

    docs = sorted((ex / "docs").glob("chat-*.md")) + sorted((ex / "docs").glob("file-*.md"))
    result, cache_out, asked = {}, {}, 0
    types, type_cache_out, candidates = {}, {}, {}
    taken_topics, taken_types = set(vocab), set(type_vocab)
    if SM is not None:   # group terms and aliases count as taken too
        try:
            schema_now, _ = SM.load()
            taken_topics |= set(SM.all_term_names(schema_now, "topics"))
            taken_types |= set(SM.all_term_names(schema_now, "doc_types"))
        except Exception:   # noqa: BLE001
            pass
    with httpx.Client(timeout=180) as client:
        for p in docs:
            kind = "chat" if p.name.startswith("chat-") else "file"
            body = p.read_text(encoding="utf-8")
            text = (f"Summary: {summ[p.name]}\n\n" if p.name in summ else "") + body[:DOC_CHARS]
            h = hashlib.md5(text.encode()).hexdigest()
            # The cache keeps the model's full answer (most relevant first); the schema's limit and
            # applies_to only shape the output, so changing either back and forth costs no calls.
            # (Raising the limit above what the model was asked for needs --force.)
            cached = cache[p.name] if cache.get(p.name, {}).get("hash") == h else None
            if kind not in topic_kinds:       # topics don't apply to this kind of record
                full, label = (cached["tags"], cached.get("label", "")) if cached else (None, "")
                tags = []
            elif cached:
                full, label = cached["tags"], cached.get("label", "")
                tags = full[:MAX_TAGS]
            else:
                full, label = ask(client, a.model, kind, text, vocab)
                tags = full
                asked += 1
            dtype, tlabel = "", ""
            if type_vocab and kind in type_kinds:
                if type_cache.get(p.name, {}).get("hash") == h:
                    dtype, tlabel = type_cache[p.name]["type"], type_cache[p.name].get("label", "")
                else:
                    dtype, tlabel = ask_type(client, a.model, kind, text, type_vocab)
                    asked += 1
                if dtype:
                    types[p.name] = dtype
                type_cache_out[p.name] = {"hash": h, "type": dtype, "label": tlabel}
            # topics: a gap when no term in the whole vocabulary covers the subject;
            # doc types: when the assigned type doesn't describe what the model called it
            sug = None if "off-topic" in tags or kind not in topic_kinds else gap(label, vocab, taken_topics)
            tsug = gap(tlabel, {dtype: type_vocab.get(dtype, "")}, taken_types) if dtype else None
            who = owner.get(p.name)
            cands = ([{"taxonomy": "topics", **sug, "picked": tags, "author": who}] if sug else []) + \
                    ([{"taxonomy": "doc_types", **tsug, "picked": [dtype], "author": who}] if tsug else [])
            if cands:
                candidates[p.name] = cands
            title = re.sub(r"^#\s*", "", body.split("\n", 1)[0]).strip()[:60]
            print(f"  {kind:4} {title:60} {', '.join(tags) or '(none)'}" + (f"  [{dtype}]" if dtype else "")
                  + "".join(f"  ?{c['taxonomy']}:{c['term']}" for c in cands))
            result[p.name] = tags
            if full is not None:              # never cache "not asked" as an empty answer
                cache_out[p.name] = {"hash": h, "tags": full, "label": label}

    extra = ({"_doc_types": types, "_doc_types_vocab": type_hash, "_doc_types_cache": type_cache_out,
              "_doc_types_vocabulary": type_vocab} if type_vocab else {})
    out_path.write_text(json.dumps({**result, "_vocab": vocab_hash, "_cache": cache_out,
                                    "_vocabulary": vocab, **extra, "_candidates": candidates}, indent=1),
                        encoding="utf-8")
    print(f"[tags] {len(result)} docs tagged, {len(types)} typed, {len(candidates)} with suggestions "
          f"({asked} model calls to {a.model}) -> {out_path}")
    if candidates:
        with httpx.Client(timeout=300) as client:
            consolidate(client, a.model, ex, {"topics": vocab, "doc_types": type_vocab})


def incremental(client, model, ex: Path, old: dict, vocab: dict, vocab_hash: str,
                type_vocab: dict, type_hash: str, type_kinds: set) -> dict:
    """Bring tags.json up to the current vocabulary without re-asking everything: drop removed
    terms, ask yes/no for added or re-described terms on the records they can affect, and
    rewrite the caches under the new vocabulary hash so the main loop reuses them."""
    if not old:
        print("[tags] --incremental: no previous tags.json; tagging everything")
        return old
    summ = summaries(ex)
    docs = {p.name: p for p in sorted((ex / "docs").glob("chat-*.md")) + sorted((ex / "docs").glob("file-*.md"))}
    texts = {}
    for name, p in docs.items():
        body = p.read_text(encoding="utf-8")
        texts[name] = (f"Summary: {summ[name]}\n\n" if name in summ else "") + body[:DOC_CHARS]
    cands = old.get("_candidates") or {}
    vecs = {}
    asked = 0

    def suggesters(term: str, desc: str, pool: list) -> set:
        """Records whose own label became this term, or one of its aliases ("(also: ...)")."""
        m = re.search(r"\(also: ([^)]*)\)$", desc)
        names = {term} | ({a.strip() for a in m.group(1).split(",")} if m else set())

        def mine(c):
            t = c.get("term") or ""
            return t in names or (SM is not None and any(SM._similar(t, n) for n in names))
        return {d for d in pool for c in cands.get(d, []) if mine(c)}

    def likely(term: str, desc: str, pool: list, holders: set) -> list:
        """Records to ask about a term: holders, records that suggested it, and the nearest
        quarter of the rest by embedding (all of them when there are few, or no embeddings)."""
        sug = suggesters(term, desc, pool)
        rest = [d for d in pool if d not in sug and d not in holders]
        k = max(8, len(pool) // 4)
        if len(rest) > k:
            if not vecs:
                got = _embed(client, [texts[d][:2000] for d in pool])
                if got:
                    vecs.update(zip(pool, got))
            q = _embed(client, [f"{term}: {desc}"]) if vecs else None
            if q:
                rest = sorted(rest, key=lambda d: -_cos(vecs[d], q[0]))[:k]
        return sorted(holders | sug | set(rest))

    # ── topics
    prev = old.get("_vocabulary")
    if old.get("_vocab") != vocab_hash:
        if not isinstance(prev, dict):
            print("[tags] --incremental: topics were tagged without a recorded vocabulary; re-tagging them all")
        else:
            removed = set(prev) - set(vocab)
            check = {t: d for t, d in vocab.items() if prev.get(t) != d}
            print(f"[tags] topics: -{len(removed)} removed {sorted(removed)}, ~{len(check)} to check {sorted(check)}")
            cache = old.get("_cache", {})
            pool = [d for d in texts if d in cache]
            for name in pool:
                cache[name]["tags"] = [t for t in cache[name]["tags"] if t not in removed]
            for term, desc in check.items():
                holders = {d for d in pool if term in cache[d]["tags"]}
                own = suggesters(term, desc, pool)
                for d in likely(term, desc, pool, holders):
                    kind = "chat" if d.startswith("chat-") else "file"
                    # comparative re-pick with the whole new list (asked alone, "is this about X?"
                    # gets a yes for nearly anything); only its decision about this term is used
                    picked, _ = ask(client, model, kind, texts[d], vocab)
                    yes = term in picked
                    asked += 1
                    tags = cache[d]["tags"]
                    if yes and term not in tags:
                        tags = [t for t in tags if t != "off-topic"]
                        if d in own:     # it named this subject itself: the term leads, the weakest tag goes
                            tags = [term] + tags[:MAX_TAGS - 1]
                            print(f"    + {term:24} {d}  (its own subject)")
                        elif len(tags) < MAX_TAGS:
                            tags.append(term)
                            print(f"    + {term:24} {d}")
                    elif not yes and term in tags and term in prev:
                        tags.remove(term)
                        print(f"    - {term:24} {d}")
                    cache[d]["tags"] = tags
            for d in pool:   # a record left with nothing falls back to a full ask
                if not cache[d]["tags"]:
                    cache.pop(d)
            old["_vocab"], old["_vocabulary"] = vocab_hash, vocab

    # ── doc types (single-valued: a yes replaces the current type)
    tprev = old.get("_doc_types_vocabulary")
    if type_vocab and old.get("_doc_types_vocab") != type_hash:
        if not isinstance(tprev, dict):
            print("[tags] --incremental: doc types were tagged without a recorded vocabulary; re-typing them all")
        else:
            removed = set(tprev) - set(type_vocab)
            check = {t: d for t, d in type_vocab.items() if tprev.get(t) != d}
            print(f"[tags] doc_types: -{len(removed)} removed {sorted(removed)}, ~{len(check)} to check {sorted(check)}")
            tc = old.get("_doc_types_cache", {})
            pool = [d for d in texts if d in tc and ("chat" if d.startswith("chat-") else "file") in type_kinds]
            for d in pool:
                if tc[d]["type"] in removed:
                    tc.pop(d)   # re-asked in full by the main loop
            pool = [d for d in pool if d in tc]
            for term, desc in check.items():
                holders = {d for d in pool if tc[d]["type"] == term}
                for d in likely(term, desc, pool, holders):
                    cur = tc[d]["type"]
                    kind = "chat" if d.startswith("chat-") else "file"
                    # comparative re-pick with the whole new list: the new type has to beat the others
                    picked, _ = ask_type(client, model, kind, texts[d], type_vocab)
                    asked += 1
                    if picked == term and cur != term:
                        print(f"    {cur} -> {term:16} {d}")
                        tc[d]["type"] = term
                    elif cur == term and picked and picked != term:
                        print(f"    {term} -> {picked:16} {d}")
                        tc[d]["type"] = picked
            old["_doc_types_vocab"], old["_doc_types_vocabulary"] = type_hash, type_vocab
    print(f"[tags] --incremental: {asked} re-check model calls")
    return old


if __name__ == "__main__":
    main()
