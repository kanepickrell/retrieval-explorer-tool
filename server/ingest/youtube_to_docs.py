"""
Turn the owner's liked-video transcripts (youtube-mcp, ~/repos/resnovae_chat/youtube/data/
transcripts/*.md) into one doc per video for build_index.py: the "youtube" dataset that Ping's
video_search reads.

  video-<videoid>.md
    # <title>
    YouTube video · <channel> · liked <day> · <url>
    ## About the video     the description, trimmed (links, hashtags and sponsor lines dropped)
    ## Transcript          the one-minute "[mm:ss] ..." paragraphs, unchanged

The structured chunker keeps whole paragraphs together and puts the title line on every chunk,
so a hit says which video it is from, and the paragraph's [mm:ss] says where in it.

Doc ids are the video ids, so a re-fetched transcript replaces its doc and an unliked video
whose file youtube-mcp removed drops out of the index (build_index.py --sync).

Usage: python youtube_to_docs.py <transcripts_dir> <out_docs_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_ABOUT = 1200
JUNK = re.compile(r"https?://|^#\w|\b(sponsor|subscribe|patreon|affiliate|discount code|merch)\b", re.I)


def _front(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip('"')
    return meta, text[m.end():]


def _section(body: str, name: str) -> str:
    m = re.search(rf"^## {name}\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    return m.group(1).strip() if m else ""


def video_doc(p: Path) -> tuple[str, str] | None:
    meta, body = _front(p.read_text(encoding="utf-8", errors="replace"))
    vid = meta.get("video_id") or (re.search(r"\[([\w-]{11})\]\.md$", p.name) or [None, None])[1]
    transcript = _section(body, "Transcript")
    if not vid or len(transcript.split()) < 30:
        return None
    title = meta.get("title") or p.stem
    about = "\n".join(l for l in _section(body, "Description").splitlines() if l.strip() and not JUNK.search(l))
    if len(about) > MAX_ABOUT:
        about = about[:MAX_ABOUT].rsplit(" ", 1)[0] + " …"
    line = " · ".join(x for x in ("YouTube video", meta.get("channel"), meta.get("duration"),
                                  f"liked {meta['liked_at'][:10]}" if meta.get("liked_at") else "",
                                  meta.get("url")) if x)
    out = [f"# {title}", "", line, ""]
    if about:
        out += ["## About the video", "", about, ""]
    out += ["## Transcript", "", transcript, ""]
    return f"video-{vid}.md", "\n".join(out)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    docs = {}
    for p in sorted(src.glob("*.md")):
        r = video_doc(p)
        if r:
            docs[r[0]] = r[1]
    changed = 0
    for name, text in docs.items():
        f = dst / name
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            changed += 1
    stale = [f for f in dst.glob("*.md") if f.name not in docs]
    for f in stale:
        f.unlink()
    print(f"{len(docs)} videos, {changed} written, {len(stale)} removed -> {dst}")


if __name__ == "__main__":
    main()
