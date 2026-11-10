"""
Generate harder eval queries from the gold chats, so retrieval techniques can be
told apart. The original eval queries are each chat's opening message, which
appears verbatim in its gold doc, so every technique finds it (a smoke test,
not a benchmark).

For each gold chat in eval_set.json the LLM writes three questions that are
answerable from that chat but do NOT reuse its wording:
  paraphrase  -- the original request, reworded
  fact        -- a specific detail stated in the conversation
  followup    -- a natural next question the conversation answers
plus a short reference answer for each, grounded in the chat text.

Standalone guard: questions referring to unstated context ("this scenario",
"the problem") are dropped.

Leakage guard: a question is dropped when most of its content words appear in
the original opening message (overlap_with_original > MAX_OVERLAP). Every kept
entry records that overlap and how much of the question's vocabulary occurs in
the gold doc at all (lexical_coverage), so results can be sliced by how
keyword-friendly a query is.

Caveat: gold = the source chat only. Another doc may also answer a generated
question; that shows up as a "miss" and is worth spot-checking.

Output: <export>/eval_set_generated.json (same gold format; query_type
generated_*), picked up by datasets.py alongside eval_set.json.

Usage:
  python generate_eval_queries.py <export_dir>
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import lightrag_provider as lp  # noqa: E402

DOC_CHARS = 9000
MAX_OVERLAP = 0.6
# questions that lean on context the searcher doesn't have ("this scenario",
# "the problem") aren't standalone; the LLM sometimes ignores the rule, so filter
DEICTIC = re.compile(
    r"\b(this|these|that|the above|the provided|the given|the same)\s+"
    r"(scenario|system|conversation|chat|code|file|files|problem|issue|document|documents|setup|"
    r"project|implementation|script|class|approach|error|tool|framework|environment)\b"
    r"|^if the (problem|issue|error)\b", re.I)
_STOP = set("""a an the and or of on in to for is are was were be been it its this that these those
with as at by from what which who how why when where can could would should do does did i you we
my your our me us please tell about there their them they any all some more most into over than
then so if not no yes using use used get make made""".split())

PROMPT = """Below is a conversation between a user and an AI assistant.

Write exactly 3 questions that someone could ask a search system, where THIS conversation contains the answer:
1. "paraphrase": the user's original request, reworded with different vocabulary (do not copy phrases from it).
2. "fact": a question about one specific detail that is stated in the conversation (a name, setting, step, command, ID or conclusion).
3. "followup": a natural follow-up question that the conversation also answers.

Rules:
- Each question must stand alone (no "this conversation", "the above", "you said").
- Do not copy more than 3 consecutive words from the user's first message.
- For each question give a short reference answer (1-3 sentences) using ONLY facts from the conversation.

Return ONLY a JSON array of 3 objects with keys "type", "question", "answer".

ORIGINAL FIRST MESSAGE:
{first}

CONVERSATION:
{doc}"""


def _content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _STOP and len(w) > 2}


def _overlap(q: str, ref: str) -> float:
    qw = _content_words(q)
    return round(len(qw & _content_words(ref)) / len(qw), 3) if qw else 1.0


def _json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, list) else []
    except Exception:
        return []


async def main(export_dir: str) -> None:
    ex = Path(export_dir)
    gold = json.loads((ex / "eval_set.json").read_text(encoding="utf-8"))
    await lp._ensure_session()
    out, dropped, t0 = [], [], time.time()
    try:
        for i, e in enumerate(gold, 1):
            chat_doc = next((g for g in e["gold_ids"] if g.startswith("chat-")), None)
            if not chat_doc or not (ex / "docs" / chat_doc).exists():
                continue
            doc = (ex / "docs" / chat_doc).read_text(encoding="utf-8")
            reply = await lp.llm(PROMPT.format(first=e["query"], doc=doc[:DOC_CHARS]),
                                 system_prompt="You write evaluation questions for a retrieval system. Output JSON only.")
            items = _json_array(reply)
            kept = 0
            for it in items:
                q = str(it.get("question") or "").strip()
                a = str(it.get("answer") or "").strip()
                kind = str(it.get("type") or "").strip().lower() or "unknown"
                if len(q) < 12 or not a:
                    dropped.append((e["id"], kind, "empty/short", q))
                    continue
                if DEICTIC.search(q):
                    dropped.append((e["id"], kind, "not standalone", q))
                    continue
                ov = _overlap(q, e["query"])
                if ov > MAX_OVERLAP:
                    dropped.append((e["id"], kind, f"overlap {ov}", q))
                    continue
                kept += 1
                out.append({
                    "id": f"{e['id']}-g{kept}",
                    "query": q,
                    "reference": a,
                    "gold_ids": e["gold_ids"],
                    "query_type": f"generated_{kind}",
                    "source_query_id": e["id"],
                    "generated_by": lp.LLM,
                    "overlap_with_original": ov,
                    "lexical_coverage": _overlap(q, doc),
                })
            print(f"[gen] {i}/{len(gold)} {e['id']}: kept {kept}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        await lp.aclose()
    (ex / "eval_set_generated.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen] wrote {len(out)} queries -> {ex / 'eval_set_generated.json'}", flush=True)
    for d in dropped:
        print("[gen] dropped", *d[:3], "|", d[3][:90], flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
