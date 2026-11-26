"""
Build the full-text layer of an export: passages only. No LLM, no entity graph.

Every doc is cut into the same 500/50-token passages LightRAG made (its own chunker and
tokenizer, so passage ids and text are identical to the old build_index.py output), each
passage is embedded with nomic-embed-text ("search_document: " prefix, normalised), and
the result is written in the LightRAG-shaped files the backend already reads:

  vdb_chunks.json          passage vectors (vector / BM25 / hybrid search, the agent's reads)
  kv_store_text_chunks.json passages by id (Explorer's chunk view)
  kv_store_full_docs.json  whole docs
  kv_store_doc_status.json one row per doc; content duplicates are marked the way LightRAG
                           marks them ("Identical content already exists. Original doc_id: X")
                           so gold labels and the metadata build map them to the original

The meaning (who, what, which records relate) lives in the metadata graph
(build_metadata_graph.py); every record there carries its doc id (file_path), which is
how the agent follows a record into these passages. That replaces LightRAG's LLM
entity extraction, which nothing used for retrieval or answers and which cost ~26 min of
LLM calls per export.

Incremental: a passage whose id and text are unchanged keeps its stored vector, so a
re-run after new data only embeds what is new.

Usage:
  python build_passages.py <docs_dir> <storage_dir>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_metadata_graph import EMBED_MODEL, embed  # noqa: E402  (same embed call + normalisation)

CHUNK_TOKENS, OVERLAP_TOKENS = 500, 50   # = build_index.py / LightRAG settings
DIM = 768
# LightRAG-only files a passages build does not produce; removed so nothing reads stale data
LIGHTRAG_ONLY = ("graph_chunk_entity_relation.graphml", "vdb_entities.json", "vdb_relationships.json",
                 "kv_store_full_relations.json", "kv_store_relation_chunks.json",
                 "kv_store_llm_response_cache.json")


def chunker():
    from lightrag.chunker.token_size import chunking_by_token_size
    from lightrag.utils import TiktokenTokenizer
    tok = TiktokenTokenizer("gpt-4o-mini")                 # LightRAG's default tokenizer
    return lambda text: chunking_by_token_size(tok, text, None, False, OVERLAP_TOKENS, CHUNK_TOKENS)


def _old_vectors(storage: Path) -> dict:
    """{(passage id, text hash): vector} from an existing vdb_chunks.json, for reuse."""
    f = storage / "vdb_chunks.json"
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        n, dim = len(d["data"]), int(d["embedding_dim"])
        if dim != DIM or not n:
            return {}
        flat = struct.unpack(f"<{n * dim}f", base64.b64decode(d["matrix"]))
        return {(r["__id__"], hashlib.md5((r.get("content") or "").encode()).hexdigest()):
                list(flat[i * dim:(i + 1) * dim]) for i, r in enumerate(d["data"])}
    except Exception as e:
        print(f"[passages] existing vectors not reusable ({e!r:.80})")
        return {}


def build(docs_dir: Path, storage: Path) -> dict:
    t0 = time.time()
    storage.mkdir(parents=True, exist_ok=True)
    cut = chunker()
    now, iso = int(time.time()), datetime.now(timezone.utc).isoformat()
    status, full_docs, chunks, order, seen = {}, {}, {}, [], {}
    for p in sorted(docs_dir.glob("*.md")):
        did, text = p.stem, p.read_text(encoding="utf-8")
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:   # same text as an earlier doc: indexed once, under the original (as LightRAG does)
            status[did] = {"status": "failed", "file_path": p.name, "content_hash": h,
                           "content_summary": text[:100], "content_length": len(text),
                           "created_at": iso, "updated_at": iso,
                           "error_msg": f"Identical content already exists. Original doc_id: {seen[h]}, "
                                        f"file: {seen[h]}.md"}
            continue
        seen[h] = did
        parts = cut(text)
        ids = [f"{did}-chunk-{i:03d}" for i in range(len(parts))]
        for cid, c in zip(ids, parts):
            chunks[cid] = {"tokens": c["tokens"], "content": c["content"], "chunk_order_index": c["chunk_order_index"],
                           "full_doc_id": did, "file_path": p.name, "create_time": now, "update_time": now, "_id": cid}
            order.append(cid)
        full_docs[did] = {"content": text, "file_path": p.name, "content_hash": h,
                          "create_time": now, "update_time": now, "_id": did}
        status[did] = {"status": "processed", "file_path": p.name, "content_hash": h,
                       "content_summary": text[:100], "content_length": len(text),
                       "created_at": iso, "updated_at": iso, "chunks_count": len(ids), "chunks_list": ids,
                       "metadata": {"chunk_opts": f"size={CHUNK_TOKENS}, overlap={OVERLAP_TOKENS}",
                                    "builder": "build_passages.py"}}

    old = _old_vectors(storage)
    key = lambda cid: (cid, hashlib.md5(chunks[cid]["content"].encode()).hexdigest())
    todo = [cid for cid in order if key(cid) not in old]
    te = time.time()
    fresh = dict(zip(todo, embed([chunks[c]["content"] for c in todo]))) if todo else {}
    print(f"[passages] {len(full_docs)} docs, {len(order)} passages; embedded {len(todo)} new "
          f"({len(order) - len(todo)} reused) in {time.time() - te:.1f}s with {EMBED_MODEL}", flush=True)
    vecs = [fresh[c] if c in fresh else old[key(c)] for c in order]

    def dump(name, obj):
        tmp = storage / (name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, storage / name)

    flat = struct.pack(f"<{len(vecs) * DIM}f", *[x for v in vecs for x in v])
    dump("vdb_chunks.json", {"embedding_dim": DIM, "matrix": base64.b64encode(flat).decode(),
                             "data": [{"__id__": c, "__created_at__": now, "content": chunks[c]["content"],
                                       "full_doc_id": chunks[c]["full_doc_id"], "file_path": chunks[c]["file_path"]}
                                      for c in order]})
    dump("kv_store_text_chunks.json", chunks)
    dump("kv_store_full_docs.json", full_docs)
    dump("kv_store_full_entities.json", {})
    dump("kv_store_entity_chunks.json", {})
    dump("kv_store_doc_status.json", status)   # last: describe() treats this as the build being complete
    removed = [n for n in LIGHTRAG_ONLY if (storage / n).exists()]
    for n in removed:
        (storage / n).unlink()
    dups = sum(1 for v in status.values() if v["status"] == "failed")
    print(f"[passages] DONE in {time.time() - t0:.1f}s -> {storage}"
          + (f"; {dups} duplicate doc(s) mapped to their original" if dups else "")
          + (f"; removed LightRAG-only files: {', '.join(removed)}" if removed else ""), flush=True)
    return {"docs": len(full_docs), "passages": len(order), "embedded": len(todo), "duplicates": dups}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir")
    ap.add_argument("storage_dir")
    a = ap.parse_args()
    build(Path(a.docs_dir), Path(a.storage_dir))
