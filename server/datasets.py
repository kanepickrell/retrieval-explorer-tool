"""
Dataset registry: which corpora the backend can serve, and where each one's
LightRAG storage and gold eval set live.

  agenthub-synthetic   the synthetic Agent Hub corpus (PROTOGRAPH_CORPUS_STORAGE)
                       + datasets/agenthub_focused.json gold
  <export folder>      every datasets/incoming/<name>/ that has a storage/ dir
                       (built by ingest/build_index.py); gold = eval_set.json,
                       docs = docs/
  <export>~<variant>   each storage-<variant>/ dir in the same export: an alternative
                       index of the same docs (e.g. storage-structured = a different
                       chunker), sharing the export's gold set, for side-by-side tests.
                       Underscore names (storage_v2, storage_backup...) are ignored.

Status comes from LightRAG's own doc-status store, so a dataset whose index is
still building reports `indexing` with progress and is not queryable yet.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
DEFAULT_ID = "agenthub-synthetic"


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    kind: str                 # "chat" | "document" | "user" (frontend colour key)
    storage: Path
    gold: Optional[Path]
    docs_dir: Optional[Path]  # source markdown, used to know the expected doc count while indexing


def _registry() -> Dict[str, Dataset]:
    out: Dict[str, Dataset] = {}
    syn = Path(os.getenv("PROTOGRAPH_CORPUS_STORAGE", r"C:\repos\lightrag-explainer\agenthub_storage"))
    out[DEFAULT_ID] = Dataset(DEFAULT_ID, "Agent Hub (synthetic)", "chat", syn,
                              Path(os.getenv("PROTOGRAPH_GOLD", _HERE / "datasets" / "agenthub_focused.json")),
                              None)
    incoming = _HERE / "datasets" / "incoming"
    if incoming.is_dir():
        for d in sorted(incoming.iterdir()):
            pretty = d.name.replace("agenthub-export-", "Agent Hub export ")
            if (d / "storage").is_dir():
                out[d.name] = Dataset(d.name, pretty, "chat", d / "storage",
                                      d / "eval_set.json", d / "docs")
            for v in sorted(d.glob("storage-*")):
                if v.is_dir():
                    variant = v.name[len("storage-"):]
                    out[f"{d.name}~{variant}"] = Dataset(f"{d.name}~{variant}", f"{pretty} · {variant}", "chat",
                                                         v, d / "eval_set.json", d / "docs")
    return out


def get(dataset_id: Optional[str]) -> Dataset:
    reg = _registry()
    return reg.get(dataset_id or DEFAULT_ID) or reg[DEFAULT_ID]


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_content_dup(v: dict) -> bool:
    return "Identical content already exists" in (v.get("error_msg") or "")


def duplicate_aliases(ds: Dataset) -> Dict[str, str]:
    """{duplicate doc file -> original doc file} for docs LightRAG skipped because
    the same content was already indexed under another id. Retrieval can only ever
    return the original, so gold labels must point at it too."""
    status = _read(ds.storage / "kv_store_doc_status.json") or {}
    out = {}
    for v in status.values():
        if isinstance(v, dict) and _is_content_dup(v):
            m = re.search(r"Original doc_id: ([^,]+)", v.get("error_msg") or "")
            if m and v.get("file_path"):
                orig = status.get(m.group(1), {})
                out[v["file_path"]] = orig.get("file_path") or f"{m.group(1)}.md"
    return out


def gold_entries(ds: Dataset) -> List[dict]:
    """The dataset's gold set: eval_set.json (real user requests) plus, when
    present next to it, eval_set_generated.json (LLM-written questions, tagged
    query_type generated_*) and eval_set_structural.json (questions about how the
    records relate, answered exactly from the metadata graph: structural_*)."""
    g = _read(ds.gold) if ds.gold and ds.gold.exists() else None
    if not isinstance(g, list):
        return []
    for name in ("eval_set_generated.json", "eval_set_structural.json"):
        f = ds.gold.parent / name
        extra = _read(f) if ds.gold.name == "eval_set.json" and f.exists() else None
        if isinstance(extra, list):
            g = g + extra
    alias = duplicate_aliases(ds)
    if alias:
        for e in g:
            ids = [alias.get(x, x) for x in e.get("gold_ids", [])]
            e["gold_ids"] = list(dict.fromkeys(ids))  # map dups -> original, dedupe, keep order
    return g


def describe(ds: Dataset) -> dict:
    status = _read(ds.storage / "kv_store_doc_status.json") or {}
    rows = [v for v in status.values() if isinstance(v, dict)]
    processed = sum(1 for v in rows if v.get("status") == "processed")
    duplicates = sum(1 for v in rows if v.get("status") == "failed" and _is_content_dup(v))
    failed = sum(1 for v in rows if v.get("status") == "failed" and not _is_content_dup(v))
    source_docs = len(list(ds.docs_dir.glob("*.md"))) if ds.docs_dir and ds.docs_dir.is_dir() else len(status)
    expected = source_docs - duplicates   # duplicates are indexed once, under the original
    has_index = (ds.storage / "graph_chunk_entity_relation.graphml").exists() and (ds.storage / "vdb_chunks.json").exists()
    if not has_index:
        state = "missing" if not status else "indexing"
    elif processed + failed < expected:
        state = "indexing"
    else:
        state = "ready"
    gold = gold_entries(ds)
    mtime = max((p.stat().st_mtime for p in ds.storage.glob("*.json")), default=0) if ds.storage.is_dir() else 0
    # short, readable gold queries make good "try" chips
    samples = [g["query"] for g in gold if 12 <= len(g.get("query", "")) <= 90 and "\n" not in g["query"]][:4]
    return {
        "id": ds.id, "name": ds.name, "kind": ds.kind,
        "rows": processed, "expected_rows": expected, "failed_rows": failed,
        "duplicate_rows": duplicates,
        "updated": datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d") if mtime else None,
        "labeled": bool(gold), "gold_queries": len(gold),
        "generated_queries": sum(1 for x in gold if str(x.get("query_type", "")).startswith("generated")),
        "has_reference_answers": any(g.get("reference") for g in gold),
        "status": state, "samples": samples,
    }


def list_all() -> List[dict]:
    return [describe(d) for d in _registry().values()]


def _as_list(v) -> list:
    """LightRAG kv fields are lists, or (in some stores) their string repr."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        import ast
        try:
            return list(ast.literal_eval(v))
        except Exception:
            return []
    return []


_TAGS = re.compile(r"^Tags: (.+)$", re.M)


def categorize(doc: str, content: str) -> tuple:
    """(category, subcategory) for browsing, from the doc's file name and text."""
    title = (content or "").lstrip().split("\n", 1)[0].lstrip("# ").strip()
    if doc.startswith("chat-"):
        m = re.match(r"Chat session chat-\d+: (.+)", title)       # synthetic corpus: topic in the title
        if m:
            return "Chats", m.group(1).strip()
        tags = _TAGS.search(content or "")
        first = tags.group(1).split(",")[0].strip() if tags else ""
        return "Chats", (first.replace("_", " ") or "untagged")
    if doc.startswith("file-user-guide"):
        return "Files", "User guide"
    if doc.startswith("file-"):
        return "Files", "Uploaded files"
    for prefix, sub in (("catalog-", "Catalogs"), ("agent-", "Agents"), ("tool-", "Tools"), ("step-", "Steps")):
        if doc.startswith(prefix):
            return "Agent Hub", sub
    if doc.startswith("person-"):
        return "People", "Pseudonymous users"
    return "Other", "Other"


def documents(ds: Dataset) -> List[dict]:
    """Indexed documents with category, chunk + entity counts (entity names = graph
    node ids) and the gold queries that point at each doc."""
    status = _read(ds.storage / "kv_store_doc_status.json") or {}
    ents = _read(ds.storage / "kv_store_full_entities.json") or {}
    full = _read(ds.storage / "kv_store_full_docs.json") or {}
    gold_by_doc: Dict[str, list] = {}
    for g in gold_entries(ds):
        for d in g.get("gold_ids", []):
            gold_by_doc.setdefault(d, []).append({
                "id": g.get("id"), "query": g.get("query", "")[:300],
                "type": g.get("query_type", "")})
    out = []
    for doc_id, s in status.items():
        if not isinstance(s, dict) or s.get("status") != "processed":
            continue
        names = _as_list((ents.get(doc_id) or {}).get("entity_names"))
        fp = s.get("file_path") or doc_id
        content = (full.get(doc_id) or {}).get("content") or s.get("content_summary") or ""
        title = content.lstrip().split("\n", 1)[0].lstrip("# ").strip()
        cat, sub = categorize(fp, content)
        out.append({"id": fp, "title": title or fp, "category": cat, "subcategory": sub,
                    "chunks": int(s.get("chunks_count") or 0), "entities": list(names),
                    "gold": gold_by_doc.get(fp, [])})
    return sorted(out, key=lambda d: (d["category"], d["subcategory"], d["title"].lower()))


def chunks_of(ds: Dataset, doc: str) -> List[dict]:
    """A document's chunks in order, with the entities extracted from each."""
    chunks = _read(ds.storage / "kv_store_text_chunks.json") or {}
    mine = {k: v for k, v in chunks.items() if isinstance(v, dict) and v.get("file_path") == doc}
    ents = _read(ds.storage / "kv_store_entity_chunks.json") or {}
    by_chunk: Dict[str, list] = {}
    for name, v in ents.items():
        for cid in _as_list((v or {}).get("chunk_ids")):
            if cid in mine:
                by_chunk.setdefault(cid, []).append(name)
    return [{"id": k, "order": int(v.get("chunk_order_index") or 0), "tokens": int(v.get("tokens") or 0),
             "content": v.get("content") or "", "entities": sorted(by_chunk.get(k, []))}
            for k, v in sorted(mine.items(), key=lambda kv: int(kv[1].get("chunk_order_index") or 0))]
