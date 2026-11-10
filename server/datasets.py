"""
Dataset registry: which corpora the backend can serve, and where each one's
LightRAG storage and gold eval set live.

  agenthub-synthetic   the synthetic Agent Hub corpus (PROTOGRAPH_CORPUS_STORAGE)
                       + datasets/agenthub_focused.json gold
  <export folder>      every datasets/incoming/<name>/ that has a storage/ dir
                       (built by ingest/build_index.py); gold = eval_set.json,
                       docs = docs/

Status comes from LightRAG's own doc-status store, so a dataset whose index is
still building reports `indexing` with progress and is not queryable yet.
"""
from __future__ import annotations

import json
import os
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
            if (d / "storage").is_dir():
                pretty = d.name.replace("agenthub-export-", "Agent Hub export ")
                out[d.name] = Dataset(d.name, pretty, "chat", d / "storage",
                                      d / "eval_set.json", d / "docs")
    return out


def get(dataset_id: Optional[str]) -> Dataset:
    reg = _registry()
    return reg.get(dataset_id or DEFAULT_ID) or reg[DEFAULT_ID]


def _read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def gold_entries(ds: Dataset) -> List[dict]:
    g = _read(ds.gold) if ds.gold and ds.gold.exists() else None
    return g if isinstance(g, list) else []


def describe(ds: Dataset) -> dict:
    status = _read(ds.storage / "kv_store_doc_status.json") or {}
    processed = sum(1 for v in status.values() if isinstance(v, dict) and v.get("status") == "processed")
    failed = sum(1 for v in status.values() if isinstance(v, dict) and v.get("status") == "failed")
    expected = len(list(ds.docs_dir.glob("*.md"))) if ds.docs_dir and ds.docs_dir.is_dir() else len(status)
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
        "updated": datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d") if mtime else None,
        "labeled": bool(gold), "gold_queries": len(gold),
        "has_reference_answers": any(g.get("reference") for g in gold),
        "status": state, "samples": samples,
    }


def list_all() -> List[dict]:
    return [describe(d) for d in _registry().values()]


def documents(ds: Dataset) -> List[dict]:
    """Indexed documents with chunk + entity counts (entity names = graph node ids)."""
    status = _read(ds.storage / "kv_store_doc_status.json") or {}
    ents = _read(ds.storage / "kv_store_full_entities.json") or {}
    out = []
    for doc_id, s in status.items():
        if not isinstance(s, dict) or s.get("status") != "processed":
            continue
        names = (ents.get(doc_id) or {}).get("entity_names") or []
        title = (s.get("content_summary") or "").split("\n", 1)[0].lstrip("# ").strip()
        out.append({"id": s.get("file_path") or doc_id, "title": title or s.get("file_path") or doc_id,
                    "chunks": int(s.get("chunks_count") or 0), "entities": list(names)})
    return sorted(out, key=lambda d: d["id"])
