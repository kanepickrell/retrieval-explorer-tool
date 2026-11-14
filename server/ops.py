"""
Ops: every chime-in call as a persistent "job", for the Ops dashboard.

A job is one POST /api/assist from a caller (Open WebUI's Chime In bridge, or this app's
own chat replay). With engine="ping" one job is one run of the Ping cycle over the user
turns that call brought; with engine="assist" it is one rewrite -> retrieve -> judge pass.

Jobs are written to SQLite (PROTOGRAPH_OPS_DB, default server/ops_data/ops.db) so the
dashboard keeps history across restarts. A job is inserted as "running" when the call
arrives and updated when it finishes, so in-flight work is visible. Rows left "running"
by a crash are marked "lost" on startup.

  GET /api/ops/summary?hours=24          KPI strip: users, sessions, jobs, rates, latency
  GET /api/ops/users?hours=24            one row per user
  GET /api/ops/jobs?after=&user=&limit=  newest first; `after` = only rows changed since that rev
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

DB_PATH = Path(os.getenv("PROTOGRAPH_OPS_DB") or Path(__file__).resolve().parent / "ops_data" / "ops.db")
ACTIVE_S = 30 * 60          # a session counts as active if it had a job in the last 30 min

router = APIRouter(prefix="/api/ops", tags=["ops"])
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_rev = 0                    # bumps on every write; the page polls with ?after=<rev>

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  rev         INTEGER NOT NULL,
  at          REAL NOT NULL,
  done_at     REAL,
  status      TEXT NOT NULL,          -- running | suggested | quiet | error | lost
  user_id     TEXT,
  user_name   TEXT,
  chat_id     TEXT,
  session_id  TEXT,
  source      TEXT,
  engine      TEXT,
  dataset     TEXT,
  chat_model  TEXT,                   -- the model the user chats with (Open WebUI)
  message     TEXT,                   -- the newest user message the call carried
  reason      TEXT,                   -- why it stayed quiet / the error
  confidence  REAL,                   -- top live hypothesis after the job (ping)
  delta       REAL,                   -- change in that confidence this job
  gate        REAL,
  total_ms    REAL,
  detail      TEXT                    -- JSON: stages, gate criteria, suggestion, mood
);
CREATE INDEX IF NOT EXISTS jobs_at ON jobs(at);
CREATE INDEX IF NOT EXISTS jobs_rev ON jobs(rev);
CREATE INDEX IF NOT EXISTS jobs_user ON jobs(user_id);
"""


def _db() -> sqlite3.Connection:
    global _conn, _rev
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(jobs)")}
        if "chat_model" not in cols:      # db created before the column existed
            _conn.execute("ALTER TABLE jobs ADD COLUMN chat_model TEXT")
        _rev = _conn.execute("SELECT COALESCE(MAX(rev), 0) FROM jobs").fetchone()[0]
        _rev += 1
        _conn.execute("UPDATE jobs SET status='lost', reason='engine restarted mid-job', rev=? WHERE status='running'", (_rev,))
        _conn.commit()
    return _conn


def _next_rev() -> int:
    global _rev
    _rev += 1
    return _rev


def _source(user_agent: Optional[str]) -> str:
    ua = (user_agent or "").lower()
    if "httpx" in ua or "python" in ua:
        return "open-webui"
    if "mozilla" in ua:
        return "explorer"
    return "other"


def start(req, user_agent: Optional[str]) -> Optional[int]:
    """Insert a running job for an /api/assist request. Never raises: Ops must not break chime-in."""
    try:
        users = [t for t in req.turns if t.role == "user" and (t.content or "").strip()]
        msg = users[-1].content.strip()[-600:] if users else ""
        src = _source(user_agent)
        with _lock:
            db = _db()
            cur = db.execute(
                "INSERT INTO jobs (rev, at, status, user_id, user_name, chat_id, source, engine, dataset, chat_model, message) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (_next_rev(), time.time(), "running",
                 getattr(req, "user_id", None) or ("local" if src == "explorer" else None),
                 getattr(req, "user_name", None) or ("Explorer (you)" if src == "explorer" else None),
                 req.chat_id, src, req.engine, req.dataset, getattr(req, "chat_model", None), msg))
            db.commit()
            return cur.lastrowid
    except Exception:
        return None


def _stages(resp: dict) -> list:
    """Compact stage list for either engine: [{name, status, ms, title}]."""
    cyc = resp.get("cycle")
    if cyc:   # ping: the cycle's eight stages, in order
        out = []
        for name in cyc.get("order", []):
            st = (cyc.get("stages") or {}).get(name) or {}
            out.append({"name": name, "status": st.get("status", "pending"), "ms": st.get("ms", 0),
                        "title": st.get("title", "")})
        return out
    return [{"name": s.get("label", ""), "status": "done", "ms": s.get("ms", 0), "title": str(s.get("stat", ""))}
            for s in resp.get("stages") or []]


def _engine_models() -> dict:
    try:
        from retrieval import ollama_client as o
        return {"llm": o.LLM, "judge": o.JUDGE_LLM, "embed": os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")}
    except Exception:
        return {}


def finish(job_id: Optional[int], resp: Optional[dict], ms: float, error: Optional[str] = None) -> None:
    if job_id is None:
        return
    try:
        resp = resp or {}
        mood = resp.get("mood") or {}
        cyc = resp.get("cycle") or {}
        sugg = resp.get("suggestions") or []
        status = "error" if error else ("suggested" if sugg and not resp.get("stay_quiet") else "quiet")
        reason = error or (None if status == "suggested" else (cyc.get("gate_failed") or resp.get("reason")))
        detail = {"stages": _stages(resp), "criteria": cyc.get("criteria") or [], "gate_title": cyc.get("gate_title"),
                  "turns_run": cyc.get("turns_run"), "hypotheses": cyc.get("hypotheses") or [],
                  "suggestions": [{"doc": s.get("doc"), "title": s.get("title"), "why": s.get("why"), "kind": s.get("kind")}
                                  for s in sugg],
                  "query": resp.get("query"), "mood": mood, "follow_ups": resp.get("follow_ups") or [],
                  "engine_models": _engine_models()}
        with _lock:
            db = _db()
            db.execute(
                "UPDATE jobs SET rev=?, done_at=?, status=?, reason=?, session_id=?, confidence=?, delta=?, gate=?, "
                "total_ms=?, detail=? WHERE id=?",
                (_next_rev(), time.time(), status, (reason or "")[:400] or None, resp.get("session"),
                 mood.get("confidence"), mood.get("delta"), mood.get("gate"), round(ms, 1),
                 json.dumps(detail, default=str), job_id))
            db.commit()
    except Exception:
        pass


# ───────────────────────── read API ─────────────────────────

def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["detail"] = json.loads(d["detail"]) if d.get("detail") else None
    d["user"] = d.pop("user_id") or "anonymous"
    d["user_name"] = d["user_name"] or d["user"]
    return d


def _pct(xs: list, p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 1)


@router.get("/summary")
def summary(hours: float = Query(24, gt=0, le=24 * 90)):
    since = time.time() - hours * 3600
    with _lock:
        rows = _db().execute("SELECT status, user_id, chat_id, session_id, at, total_ms FROM jobs WHERE at >= ?",
                             (since,)).fetchall()
    now = time.time()
    done = [r for r in rows if r["status"] in ("suggested", "quiet")]
    lat = [r["total_ms"] for r in done if r["total_ms"] is not None]
    count = lambda st: sum(1 for r in rows if r["status"] == st)   # noqa: E731
    return {
        "hours": hours, "rev": _rev,
        "users": len({r["user_id"] or "anonymous" for r in rows}),
        "active_users": len({r["user_id"] or "anonymous" for r in rows if now - r["at"] < ACTIVE_S}),
        "sessions": len({r["chat_id"] or r["session_id"] for r in rows}),
        "active_sessions": len({r["chat_id"] or r["session_id"] for r in rows if now - r["at"] < ACTIVE_S}),
        "jobs": len(rows), "running": count("running"),
        "suggested": count("suggested"), "quiet": count("quiet"), "errors": count("error") + count("lost"),
        "chime_rate": round(count("suggested") / len(done), 3) if done else None,
        "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95),
    }


@router.get("/users")
def users(hours: float = Query(24, gt=0, le=24 * 90)):
    since = time.time() - hours * 3600
    with _lock:
        rows = _db().execute(
            "SELECT COALESCE(user_id,'anonymous') AS user, MAX(user_name) AS user_name, "
            "COUNT(*) AS jobs, COUNT(DISTINCT COALESCE(chat_id, session_id)) AS sessions, "
            "SUM(status='suggested') AS suggested, SUM(status='quiet') AS quiet, "
            "SUM(status IN ('error','lost')) AS errors, SUM(status='running') AS running, "
            "MAX(at) AS last_at, AVG(total_ms) AS avg_ms "
            "FROM jobs WHERE at >= ? GROUP BY COALESCE(user_id,'anonymous') ORDER BY last_at DESC", (since,)).fetchall()
        # each user's latest confidence: how warm Ping is on their most recent chat
        conf = {r["u"]: (r["confidence"], r["chat_id"]) for r in _db().execute(
            "SELECT COALESCE(user_id,'anonymous') AS u, confidence, chat_id FROM jobs j WHERE at >= ? AND id = "
            "(SELECT MAX(id) FROM jobs k WHERE COALESCE(k.user_id,'anonymous') = COALESCE(j.user_id,'anonymous') "
            "AND k.confidence IS NOT NULL)", (since,)).fetchall()}
    out = []
    for r in rows:
        d = dict(r)
        d["user_name"] = d["user_name"] or d["user"]
        d["avg_ms"] = round(d["avg_ms"], 1) if d["avg_ms"] is not None else None
        d["confidence"] = conf.get(d["user"], (None, None))[0]
        out.append(d)
    return out


@router.get("/jobs")
def jobs(after: int = Query(0, ge=0), user: Optional[str] = None, limit: int = Query(60, ge=1, le=500),
         hours: float = Query(24, gt=0, le=24 * 90)):
    since = time.time() - hours * 3600
    q, args = "SELECT * FROM jobs WHERE at >= ? AND rev > ?", [since, after]
    if user:
        q += " AND COALESCE(user_id,'anonymous') = ?"
        args.append(user)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _lock:
        rows = _db().execute(q, args).fetchall()
    return {"rev": _rev, "jobs": [_row(r) for r in rows]}
