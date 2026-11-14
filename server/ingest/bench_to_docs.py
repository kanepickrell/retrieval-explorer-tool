"""
Convert the AI bench's own data into one markdown doc per item, ready for
build_index.py. Everything is read from local copies; nothing leaves this machine
except the text sent to the inference host (embeddings / extraction) at index time.

  pi sessions  ~/.pi/agent/sessions/*/*.jsonl  -> session-<id>.md
               user / assistant turns as "## User" / "## Assistant" sections; tool calls become one line each with the
               short result the bench spoke; thinking and system prompts are dropped.
  vault notes  ~/BenchVault/**/*.md            -> note-<path>.md (Home, Sessions/<day>,
               project notes, hand-written notes). log.md is skipped: its notes,
               measurements and issues are already in the day's Sessions note, and its
               routine tool lines duplicate the pi sessions.
  issues       <project>/issues.json           -> issue-<project>-<n>.md (hypotheses,
               ruled out, notes, resolution: the debugging history of one problem)
  measurements <project>/measurements.csv      -> measure-<project>-<day>.md
  project      <project>/project.yaml          -> folded into project-<project>.md
  extras       --extra file.md ...             -> note-<name>.md (order list, open
               feedback issues, anything else worth searching)

Test sessions and projects (e2e runs, "ZZ Test Bench", ...) are skipped by
--skip, a regex matched against session dirs and project names.

Doc ids are stable (session uuid, note path, issue number), so re-running only
changes docs whose content changed and eval gold labels keep pointing at them.

Usage:
  python bench_to_docs.py <out_docs_dir> [--vault ~/BenchVault]
         [--sessions <dir of pi session dirs>] [--extra f.md ...] [--skip REGEX]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MAX_TEXT = 6000           # cap one message (pasted logs, long answers)
MAX_TOOL_RESULT = 400     # tool results: the spoken line is what matters
MIN_USER_CHARS = 20       # sessions whose user turns total less are mic checks
DEFAULT_SKIP = r"pi-test|e2e|^zz|unused-test"
# Voice sessions open with mic checks; the title (which labels every chunk) should be the first real request.
SMALL_TALK = re.compile(r"^\W*(hey|hi|hello|ok(ay)?|yes|no|thanks?|thank you|can you hear me|are you there|"
                        r"testing|test|what do you know)\b", re.I)


def pick_title(user_msgs: list[str]) -> str:
    for t in user_msgs:
        line = t.strip().splitlines()[0]
        if len(line.split()) >= 4 and not SMALL_TALK.match(line):
            return clip(line, 90)
    return clip(user_msgs[0].strip().splitlines()[0], 90)
# Tool lines in a session doc: screen/window control says nothing worth remembering (the
# assistant's own reply says what happened) and only feeds UI noise into the graph; pi's
# coding tools keep the call but not the output (file listings, file contents).
UI_TOOLS = {"show", "window", "keys", "layout_set", "dashboard_window", "screen_clear", "view_scroll",
            "view_page", "camera_show", "app_screen", "app_command", "note_nav", "places", "timer_set"}
NO_RESULT_TOOLS = {"ls", "read", "bash", "find", "grep", "edit", "write", "note_read", "doc_read",
                   "memory_search"}   # memory_search: don't re-index past passages as new content
SECRETS = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"      # JWT
                     r"|\b(sk|pk|ghp|gho|xox[bap])[-_][A-Za-z0-9_-]{16,}"                # API keys
                     r"|(?i:(token|secret|password|passwd|api[_-]?key)\s*[=:]\s*)\S{8,}")
SKIP_DIRS = {".git", ".host", ".stfolder", ".stversions", "media", "kicad", "firmware", "docs"}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80] or "x"


def scrub(s: str) -> str:
    """Text leaves this machine at index time (to the inference host): drop anything key-shaped."""
    return SECRETS.sub("[redacted]", s)


def clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


# ───────────────────────────── pi sessions ─────────────────────────────

def _text_parts(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text").strip()


def _tool_result_line(text: str) -> str:
    """The bench's tool results lead with what to say; keep that, not the detail dump."""
    m = re.search(r"Say to the user[^:]*:\s*(.+?)(?:\n\n|$)", text, re.S)
    return clip(re.sub(r"\s+", " ", m.group(1) if m else text), MAX_TOOL_RESULT)


def _args(a) -> str:
    if not isinstance(a, dict):
        return ""
    return ", ".join(f"{k}={clip(str(v), 120)}" for k, v in a.items())


def session_doc(path: Path) -> tuple[str, str] | None:
    head, turns = None, []          # turns: [role, [lines]]
    results: dict[str, str] = {}
    calls: list[tuple[list, int, str, str]] = []   # (turn lines, index, call id, rendered call) to fill in results
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "session":
            head = o
        if o.get("type") != "message":
            continue
        m = o["message"]
        role = m.get("role")
        if role == "user":
            # talk.py prefixes bench state ("[radio: off ...]", "[boards on USB: ...]"): context, not the user's words
            t = re.sub(r"^(\[[^\]\n]*\]\s*\n)+", "", _text_parts(m.get("content")))
            if t:
                turns.append(["User", [clip(t, MAX_TEXT)], o.get("timestamp", "")])
        elif role == "assistant":
            if not turns or turns[-1][0] != "Assistant":
                turns.append(["Assistant", [], o.get("timestamp", "")])
            lines = turns[-1][1]
            for p in m.get("content") or []:
                if p.get("type") == "text" and p.get("text", "").strip():
                    lines.append(clip(p["text"], MAX_TEXT))
                elif p.get("type") == "toolCall" and p.get("name") not in UI_TOOLS:
                    lines.append(None)    # placeholder, filled once the result is seen
                    calls.append((lines, len(lines) - 1, p.get("id", ""), f"{p.get('name')}({_args(p.get('arguments'))})"))
        elif role == "toolResult":
            results[m.get("toolCallId", "")] = ("error: " if m.get("isError") else "") + \
                _tool_result_line(_text_parts(m.get("content")))
    for lines, i, cid, call in calls:
        r = None if call.split("(", 1)[0] in NO_RESULT_TOOLS else results.get(cid)
        lines[i] = f"Tool: {call}" + (f" → {r}" if r else "")
    turns = [t for t in turns if t[1]]
    users = [t for t in turns if t[0] == "User"]
    if not head or sum(len(" ".join(t[1])) for t in users) < MIN_USER_CHARS:
        return None
    title = pick_title([t[1][0] for t in users])
    day = head.get("timestamp", "")[:10]
    where = re.sub(r"^/home/[^/]+", "~", head.get("cwd", ""))   # the Pi's home, whoever runs this
    start = head.get("timestamp", "")[11:16]
    out = [f"# {title}", "", f"Bench conversation · {day} {start} · {where} · {len(users)} requests", ""]
    for role, lines, _ts in turns:          # no per-turn time: the chunk header reads the line after "## User"
        out += [f"## {role}", "\n".join(lines), ""]
    return f"session-{head.get('id', path.stem)}.md", scrub("\n".join(out))


# ───────────────────────────── vault ─────────────────────────────

def vault_docs(vault: Path, skip: re.Pattern) -> dict[str, str]:
    docs: dict[str, str] = {}
    for p in sorted(vault.rglob("*.md")):
        rel = p.relative_to(vault)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.name in ("log.md", "issues.md"):        # see module docstring; issues come from issues.json
            continue
        if len(rel.parts) > 1 and skip.search(rel.parts[0].lower()):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<!-- bench:(start|end) -->\n?", "", text)
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)     # frontmatter
        if not text.strip():
            continue
        if not text.lstrip().startswith("# "):
            text = f"# {p.stem}\n\n{text}"
        kind = "Bench session day" if rel.parts[0] == "Sessions" else "Bench note"
        title, _, body = text.lstrip().partition("\n")
        docs[f"note-{slugify(str(rel.with_suffix('')))}.md"] = scrub(f"{title}\n\n{kind} · {rel}\n{body}")

    for proj in sorted(d for d in vault.iterdir() if d.is_dir() and not d.name.startswith(".")):
        if proj.name == "Sessions" or skip.search(proj.name.lower()):
            continue
        name = proj.name
        y = proj / "project.yaml"
        if y.exists():
            try:
                import yaml
                cfg = yaml.safe_load(y.read_text()) or {}
                name = str(cfg.get("name") or name)
                lines = [f"# {name} (project)", "", f"Bench project settings · {proj.name}/project.yaml", ""]
                lines += [f"- {k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}" for k, v in cfg.items()]
                docs[f"project-{slugify(proj.name)}.md"] = "\n".join(lines)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {y}: {e}", file=sys.stderr)
        ij = proj / "issues.json"
        if ij.exists():
            for i in json.loads(ij.read_text() or "[]"):
                lines = [f"# Issue #{i['id']}: {i['title']}", "",
                         f"Bench issue in {name} · {i.get('status', '')} · opened {i.get('opened', '')}", ""]
                for label, key in (("Hypotheses", "hypotheses"), ("Ruled out", "ruled_out"), ("Notes", "notes")):
                    if i.get(key):
                        lines += [f"## {label}", *[f"- {x}" for x in i[key]], ""]
                if i.get("resolution"):
                    lines += ["## Resolution", i["resolution"], ""]
                docs[f"issue-{slugify(proj.name)}-{i['id']}.md"] = "\n".join(lines)
        mc = proj / "measurements.csv"
        if mc.exists():
            by_day = defaultdict(list)
            for r in csv.DictReader(mc.open()):
                by_day[(r.get("time") or "")[:10]].append(r)
            for day, rows in by_day.items():
                lines = [f"# {name} measurements {day}", "", f"Bench measurements in {name} · {day}", ""]
                lines += [f"- {r['time'][11:16]} {r['net']} = {r['value']} {r['unit']}"
                          + (f" ({r['instrument']})" if r.get("instrument") else "")
                          + (f" -- {r['note']}" if r.get("note") else "") for r in rows]
                docs[f"measure-{slugify(proj.name)}-{day}.md"] = "\n".join(lines)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vault", default=str(Path.home() / "BenchVault"))
    ap.add_argument("--sessions", help="dir holding pi's per-cwd session dirs")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--skip", default=DEFAULT_SKIP)
    a = ap.parse_args()
    skip = re.compile(a.skip, re.I)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    docs: dict[str, str] = {}
    n_skipped = 0
    if a.sessions:
        for d in sorted(Path(a.sessions).iterdir()):
            if skip.search(d.name):
                n_skipped += len(list(d.glob("*.jsonl")))
                continue
            for f in sorted(d.glob("*.jsonl")):
                r = session_doc(f)
                if r:
                    docs[r[0]] = r[1]
                else:
                    n_skipped += 1
    if Path(a.vault).is_dir():
        docs.update(vault_docs(Path(a.vault), skip))
    for e in a.extra:
        p = Path(e)
        if p.exists() and p.read_text().strip():
            t = p.read_text()
            docs[f"note-{slugify(p.stem)}.md"] = t if t.startswith("# ") else f"# {p.stem}\n\n{t}"

    changed = 0
    for name, text in docs.items():
        f = out / name
        if not f.exists() or f.read_text(encoding="utf-8") != text:
            f.write_text(text, encoding="utf-8")
            changed += 1
    stale = [f for f in out.glob("*.md") if f.name not in docs]
    for f in stale:
        f.unlink()
    kinds = defaultdict(int)
    for n in docs:
        kinds[n.split("-", 1)[0]] += 1
    print(f"{len(docs)} docs ({dict(kinds)}), {changed} written, {len(stale)} removed, "
          f"{n_skipped} sessions skipped (test or too short) -> {out}")


if __name__ == "__main__":
    main()
