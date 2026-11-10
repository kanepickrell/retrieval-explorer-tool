"""
Build a LightRAG index (graph + chunk vdb) from a folder of markdown docs,
with the SAME settings the backend queries with (retrieval/lightrag_provider.py):
nomic-embed-text (768d, search_document:/search_query: prefixes), gemma3:12b,
500/50-token chunks. Point the backend at the result with
PROTOGRAPH_CORPUS_STORAGE=<storage_dir>.

Resumable: LightRAG tracks doc status, so re-running skips finished docs.

Usage:
  python build_index.py <docs_dir> <storage_dir> [--max-async N]
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


async def main(docs_dir: str, storage_dir: str, max_async: int) -> None:
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
    a = ap.parse_args()
    asyncio.run(main(a.docs_dir, os.path.abspath(a.storage_dir), a.max_async))
