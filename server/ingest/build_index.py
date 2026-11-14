"""
Build a LightRAG index (graph + chunk vdb) from a folder of markdown docs,
with the SAME settings the backend queries with (retrieval/ollama_client.py):
nomic-embed-text (768d, search_document:/search_query: prefixes), gemma3:12b,
500/50-token chunks. Point the backend at the result with
PROTOGRAPH_CORPUS_STORAGE=<storage_dir>.

Resumable by default: docs already processed are skipped; failed records (and
the filename-duplicate records a plain retry leaves behind) are deleted and those
docs re-inserted, because LightRAG will not re-process a failed doc on its own.
--retry-failed limits the run to the failed docs.

Windows hardening (both seen in practice):
  - LightRAG saves each store by writing a temp file and os.replace()-ing it over
    the old one; antivirus / the search indexer briefly holding the file makes
    that raise "Access is denied". os.replace is retried with backoff.
  - LLM / embedding calls that hit a dropped connection are retried with a fresh
    HTTP session instead of failing the doc.
Content duplicates (the same text under another id) are left alone: LightRAG
indexes that text once under the original id.

--sync keeps a live index in step with a docs folder that changes (the bench's
sessions grow while you work): a content hash per doc is kept in
<storage>/doc_hashes.json; a processed doc whose text changed is deleted and
re-inserted, and a doc gone from the folder is deleted from the index. Docs
processed before hashes were tracked are assumed unchanged (their hash is recorded).

--entity-guidance FILE replaces LightRAG's generic entity types (Person, Artifact,
Method, ...) with the file's text, e.g. ingest/bench_entities.txt for electronics
work. --drop-entities FILE deletes the listed entity names (one per line, a
trailing * matches a prefix) after every run: speaker and boilerplate entities a
small extraction model creates anyway, which otherwise become hubs linking
everything to everything.

Usage:
  python build_index.py <docs_dir> <storage_dir> [--max-async N] [--retry-failed] [--sync]
                        [--entity-guidance FILE] [--drop-entities FILE]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import aiohttp  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from retrieval import ollama_client as ollama  # noqa: E402  (reuse its Ollama funcs + settings)

_os_replace = os.replace


def _replace_with_retry(src, dst, *a, **k):
    """os.replace, retried while Windows reports the target as locked (WinError 5/32)."""
    for attempt in range(40):
        try:
            return _os_replace(src, dst, *a, **k)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(0.05 * (attempt + 1), 1.0))


os.replace = _replace_with_retry   # build process only; LightRAG's atomic writes go through it


_RETIRED: list = []   # sessions replaced after a dropped connection, closed at the end


def _with_retry(fn, what: str, attempts: int = 4):
    async def wrapped(*a, **k):
        for i in range(attempts):
            try:
                return await fn(*a, **k)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, AttributeError, RuntimeError) as e:
                # RuntimeError covers "Session is closed" from a session swapped out mid-call
                if i == attempts - 1:
                    raise
                print(f"[build] {what} failed ({e!r:.80}); retry {i + 1}/{attempts - 1}", flush=True)
                # Swap in a fresh session WITHOUT closing the old one: other calls may still
                # be using it (llm_model_max_async > 1); closing it killed them all at once.
                # Replaced sessions are closed at the end of the build.
                old = ollama._SESSION
                if old is not None and old not in _RETIRED:
                    _RETIRED.append(old)
                    ollama._SESSION = aiohttp.ClientSession()
                await asyncio.sleep(5 * (i + 1))
    return wrapped


_llm = _with_retry(ollama._ollama_llm, "llm call")
_embed = _with_retry(ollama._ollama_embed, "embed call")


def _status(storage_dir: str) -> dict:
    import json
    f = Path(storage_dir) / "kv_store_doc_status.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _failed_doc_ids(storage_dir: str) -> tuple:
    """(failed real docs, leftover filename-duplicate records) from doc status."""
    import json
    f = Path(storage_dir) / "kv_store_doc_status.json"
    st = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    # "processing"/"pending" left behind by a crashed run count as failed: LightRAG
    # would otherwise treat a re-insert as a filename duplicate
    failed = [k for k, v in st.items()
              if v.get("status") in ("failed", "processing", "pending") and not k.startswith("dup-")]
    stale_dups = [k for k, v in st.items() if k.startswith("dup-") and v.get("status") == "failed"
                  and "File name already exists" in (v.get("error_msg") or "")]
    return failed, stale_dups


async def _drop_entities(rag, storage_dir: str, list_file: str) -> None:
    import networkx as nx
    pats = [l.strip().lower() for l in Path(list_file).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    gml = Path(storage_dir) / "graph_chunk_entity_relation.graphml"
    if not gml.exists():
        return
    names = list(nx.read_graphml(str(gml)).nodes)
    hit = [n for n in names if any(n.lower() == p or (p.endswith("*") and n.lower().startswith(p[:-1]))
                                   for p in pats)]
    for n in hit:
        await rag.adelete_by_entity(n)
    print(f"[build] dropped {len(hit)} boilerplate entities: {', '.join(hit[:12])}"
          f"{' ...' if len(hit) > 12 else ''}", flush=True)


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


async def main(docs_dir: str, storage_dir: str, max_async: int, retry_failed: bool = False,
               sync: bool = False, entity_guidance: str = "", drop_entities: str = "") -> None:
    docs = sorted(Path(docs_dir).glob("*.md"))
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    print(f"[build] {len(docs)} docs -> {storage_dir}  llm={ollama.LLM} embed={ollama.EMBED} host={ollama.HOST}", flush=True)

    ollama._SESSION = aiohttp.ClientSession()
    rag = LightRAG(
        working_dir=storage_dir,
        llm_model_func=_llm, llm_model_name=ollama.LLM,
        embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=2048,
                                     func=_embed, model_name=ollama.EMBED),
        chunk_token_size=500, chunk_overlap_token_size=50,
        llm_model_max_async=max_async, enable_llm_cache=True,
        **({"addon_params": {"entity_types_guidance": Path(entity_guidance).read_text(encoding="utf-8").strip()}}
           if entity_guidance else {}),
    )
    await rag.initialize_storages()
    failed, stale = _failed_doc_ids(storage_dir)
    for doc_id in failed + stale:
        r = await rag.adelete_by_doc_id(doc_id)
        print(f"[build] deleted failed record {doc_id}: {getattr(r, 'status', r)}", flush=True)
    done = {k for k, v in _status(storage_dir).items() if v.get("status") == "processed"}
    hash_file = Path(storage_dir) / "doc_hashes.json"
    hashes = json.loads(hash_file.read_text()) if hash_file.exists() else {}
    if sync:
        present = {p.stem: p for p in docs}
        for doc_id in sorted(k for k in done if not k.startswith("dup-")):
            p = present.get(doc_id)
            if p is None:
                r = await rag.adelete_by_doc_id(doc_id)
                hashes.pop(doc_id, None)
                done.discard(doc_id)
                print(f"[build] removed {doc_id} (gone from docs): {getattr(r, 'status', r)}", flush=True)
            elif doc_id not in hashes:
                hashes[doc_id] = _hash(p)            # indexed before hashes were kept: assume unchanged
            elif hashes[doc_id] != _hash(p):
                r = await rag.adelete_by_doc_id(doc_id)
                done.discard(doc_id)
                print(f"[build] changed {doc_id}: re-inserting ({getattr(r, 'status', r)})", flush=True)
        hash_file.write_text(json.dumps(hashes, indent=1))
    if retry_failed:
        docs = [p for p in docs if p.stem in set(failed)]
    else:
        docs = [p for p in docs if p.stem not in done]
    print(f"[build] {len(done)} already processed; {len(docs)} to insert "
          f"({len(failed)} of them retries)", flush=True)
    t0 = time.time()
    try:
        for i, p in enumerate(docs, 1):
            t = time.time()
            await rag.ainsert(p.read_text(encoding="utf-8"), ids=[p.stem], file_paths=[p.name])
            if sync:
                hashes[p.stem] = _hash(p)
                hash_file.write_text(json.dumps(hashes, indent=1))
            print(f"[build] {i}/{len(docs)} {p.name}  {time.time()-t:.0f}s  (total {(time.time()-t0)/60:.1f} min)", flush=True)
        if drop_entities:
            await _drop_entities(rag, storage_dir, drop_entities)
    finally:
        await rag.finalize_storages()
        await ollama._SESSION.close()
        for old in _RETIRED:
            try:
                await old.close()
            except Exception:
                pass
    print(f"[build] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir")
    ap.add_argument("storage_dir")
    ap.add_argument("--max-async", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true",
                    help="delete failed doc records and re-insert just those docs")
    ap.add_argument("--sync", action="store_true",
                    help="re-insert changed docs and delete removed ones (content hashes in doc_hashes.json)")
    ap.add_argument("--entity-guidance", default="", help="file replacing LightRAG's entity type guidance")
    ap.add_argument("--drop-entities", default="", help="file of entity names to delete after the run")
    a = ap.parse_args()
    asyncio.run(main(a.docs_dir, os.path.abspath(a.storage_dir), a.max_async, a.retry_failed, a.sync,
                     a.entity_guidance, a.drop_entities))
