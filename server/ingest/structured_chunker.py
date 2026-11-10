"""
Structure-aware chunking with a per-chunk context header, as a LightRAG
`chunking_func` (legacy 6-arg contract). Used to build a comparison index whose
ONLY difference from the default is how docs are cut into chunks.

Default LightRAG chunking cuts fixed 500-token windows from the start of a doc,
ignoring its structure: a chat answer or a list can be split mid-way, and a
chunk deep inside a long chat carries no hint of what the chat is about.

This chunker:
  1. Splits on structure first: "## " section starts (chat turns, headings), then
     blank-line paragraphs, then -- only for a single paragraph that is itself too
     long -- token windows with overlap.
  2. Packs whole units greedily up to the token budget, so a turn or a list stays
     in one chunk whenever it fits.
  3. Prefixes every chunk with a short header built from the doc itself (title,
     the user's opening question for chats, author line, or the one-line
     description for agents/tools/steps). The header is part of the chunk text,
     so it is embedded, BM25-indexed and seen by entity extraction: every chunk
     "knows" which doc it belongs to.

The header is derived from the content because chunking_func receives only the
text (no doc id); docs from export_to_docs.py start with "# <title>".
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

HEADER_MAX_CHARS = 260


def _first_line_after(text: str, marker: str) -> Optional[str]:
    i = text.find(marker)
    if i < 0:
        return None
    rest = text[i + len(marker):].strip()
    return rest.split("\n\n", 1)[0].strip() if rest else None


def build_header(content: str) -> str:
    lines = content.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else ""
    parts = []
    if "\n## User" in content or content.startswith("## User"):
        q = _first_line_after(content, "## User")
        author = re.search(r"^Author: (.+)$", content, re.M)
        parts.append(f'Chat: "{title}"' + (f" · {author.group(1).strip()}" if author else ""))
        if q and q.lower() != title.lower():
            parts.append(f"Opening question: {q}")
    else:
        parts.append(title or "Document")
        # agents/tools/steps/catalogs: the line after the title is a one-line description
        desc = next((l.strip() for l in lines[1:6] if l.strip() and not l.startswith("#")), "")
        if desc and len(desc) < 300:
            parts.append(desc)
    header = "[" + " | ".join(parts) + "]"
    return header if len(header) <= HEADER_MAX_CHARS else header[: HEADER_MAX_CHARS - 2] + "…]"


def _units(body: str) -> List[str]:
    """Structural units: '## ' sections; the caller splits oversized ones further."""
    secs = re.split(r"\n(?=## )", body)
    return [s.strip() for s in secs if s.strip()]


def structured_chunking(tokenizer, content: str, split_by_character: Optional[str] = None,
                        split_by_character_only: bool = False, chunk_overlap_token_size: int = 50,
                        chunk_token_size: int = 500) -> List[Dict[str, Any]]:
    enc = tokenizer.encode
    header = build_header(content)
    h_tok = len(enc(header)) + 2
    budget = max(120, chunk_token_size - h_tok)

    pieces: List[str] = []
    for unit in _units(content):
        if len(enc(unit)) <= budget:
            pieces.append(unit)
            continue
        for para in [p for p in re.split(r"\n\s*\n", unit) if p.strip()]:
            toks = enc(para)
            if len(toks) <= budget:
                pieces.append(para.strip())
                continue
            step = max(1, budget - chunk_overlap_token_size)   # token windows, last resort
            for start in range(0, len(toks), step):
                pieces.append(tokenizer.decode(toks[start:start + budget]).strip())
                if start + budget >= len(toks):
                    break

    chunks, cur, cur_tok = [], [], 0
    for p in pieces:
        t = len(enc(p)) + 1
        if cur and cur_tok + t > budget:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t
    if cur:
        chunks.append("\n\n".join(cur))

    out = []
    for i, c in enumerate(chunks):
        text = f"{header}\n\n{c}"
        out.append({"tokens": len(enc(text)), "content": text, "chunk_order_index": i})
    return out
