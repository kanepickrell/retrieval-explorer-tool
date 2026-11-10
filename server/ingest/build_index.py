"""
Build a LightRAG index (graph + chunk vdb) from a folder of markdown docs,
with the SAME settings the backend queries with (retrieval/lightrag_provider.py):
nomic-embed-text (768d, search_document:/search_query: prefixes), gemma3:12b,
500/50-token chunks. Point the backend at the result with
PROTOGRAPH_CORPUS_STORAGE=<storage_dir>.

Resumable: LightRAG tracks doc status, so re-running skips finished docs.
LightRAG will NOT re-process a doc whose earlier attempt failed (it reports a
filename duplicate), so --retry-failed deletes those failed records (and the
duplicate records a plain retry leaves behind) before re-inserting the docs.
Content duplicates (the same text under another id) are left alone: LightRAG
indexes that text once under the original id.

Usage:
  python build_index.py <docs_dir> <storage_dir> [--max-async N] [--retry-failed]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import aiohttp  # noqa: E402
from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from retrieval import lightrag_provider as lp  # noqa: E402  (reuse its Ollama funcs + settings)


def _failed_doc_ids(storage_dir: str) -> tuple:
    """(failed real docs, leftover filename-duplicate records) from doc status."""
    import json
    f = Path(storage_dir) / "kv_store_doc_status.json"
    st = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    failed = [k for k, v in st.items() if v.get("status") == "failed" and not k.startswith("dup-")]
    stale_dups = [k for k, v in st.items() if k.startswith("dup-") and v.get("status") == "failed"
                  and "File name already exists" in (v.get("error_msg") or "")]
    return failed, stale_dups


async def main(docs_dir: str, storage_dir: str, max_async: int, retry_failed: bool = False) -> None:
    docs = sorted(Path(docs_dir).glob("*.md"))
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    print(f"[build] {len(docs)} docs -> {storage_dir}  llm={lp.LLM} embed={lp.EMBED} host={lp.HOST}", flush=True)

    lp._SESSION = aiohttp.ClientSession()
    rag = LightRAG(
        working_dir=storage_dir,
        llm_model_func=lp._ollama_llm, llm_model_name=lp.LLM,
        embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=2048,
                                     func=lp._ollama_embed, model_name=lp.EMBED),
        chunk_token_size=500, chunk_overlap_token_size=50,
        llm_model_max_async=max_async, enable_llm_cache=True,
    )
    await rag.initialize_storages()
    if retry_failed:
        failed, stale = _failed_doc_ids(storage_dir)
        for doc_id in failed + stale:
            r = await rag.adelete_by_doc_id(doc_id)
            print(f"[build] deleted failed record {doc_id}: {getattr(r, 'status', r)}", flush=True)
        wanted = set(failed)
        docs = [p for p in docs if p.stem in wanted]
        print(f"[build] retrying {len(docs)} failed docs", flush=True)
    t0 = time.time()
    try:
        for i, p in enumerate(docs, 1):
            t = time.time()
            await rag.ainsert(p.read_text(encoding="utf-8"), ids=[p.stem], file_paths=[p.name])
            print(f"[build] {i}/{len(docs)} {p.name}  {time.time()-t:.0f}s  (total {(time.time()-t0)/60:.1f} min)", flush=True)
    finally:
        await rag.finalize_storages()
        await lp._SESSION.close()
    print(f"[build] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docs_dir")
    ap.add_argument("storage_dir")
    ap.add_argument("--max-async", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true",
                    help="delete failed doc records and re-insert just those docs")
    a = ap.parse_args()
    asyncio.run(main(a.docs_dir, os.path.abspath(a.storage_dir), a.max_async, a.retry_failed))
