"""
The Ping cycle: a profile of what the user is after, kept across the whole
conversation, tested through the follow-up questions offered after each reply,
and only acted on once it is confident.

Per user turn, eight stages (each recorded, so the Agent page can show them):

  1 observe   the new user message (+ the assistant reply that follows it); does it
              echo one of the follow-ups offered last turn (embedding similarity)?
  2 update    LLM updates the profile from the new message only: goal, stage,
              who/what/why, which follow-up (if any) the user took (1.a), and whether
              the message supports or contradicts each hypothesis (1.b); new
              hypotheses for needs not covered, retire dead ones
  3 check L1  every live hypothesis against the metadata graph, deterministically
              (agent_loop._validate: names the user said, records that exist,
              non-empty scope, rewrite agrees with the user's words)
  4 probe     LLM writes the 3 follow-ups shown after the reply: each useful on its
              own, and where possible one that tells open hypotheses apart
  5 gate      suggest only if the top hypothesis is confident, held up on L1 this
              turn, was confirmed by the user (a follow-up taken, or a supporting
              message), the conversation is about work, and it isn't a repeat
  6 dive L2   scoped retrieval for the top hypothesis + the evidence check
              (agent_loop._execute / _dive_and_check)
  7 suggest   one record, framed by kind (artifact / insight / idea), with the reason
  8 learn     confidence snapshot and what changed this turn

Confidence: support +0.15 (+0.30 when it came through a follow-up the user took),
contradict -0.25, no signal -0.05; L1 validated +0.10, weak -0.05, rejected -0.30.
A hypothesis below 0.10 is retired.

Replays use real chats from the export: the real user message, the real reply,
and the real next message as ground truth for the probes and hypotheses.

  GET    /api/ping/chats                 replayable real chats (from /api/assist/chats)
  POST   /api/ping/sessions              {dataset, chat_id?} -> new session
  POST   /api/ping/sessions/{id}/next    replay: run the cycle on the chat's next user turn
  POST   /api/ping/sessions/{id}/say     {message}: the chat model replies, then the cycle runs on both
  GET    /api/ping/sessions              recent sessions
  GET    /api/ping/sessions/{id}         full state: profile, turns, every cycle's stages
  DELETE /api/ping/sessions              reset (also clears the agent's run list)
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import agent_loop as A
import datasets as DS

router = APIRouter(prefix="/api/ping", tags=["Ping cycle"])

STAGES = ["observe", "update", "check", "probe", "gate", "dive", "suggest", "learn"]
MAX_SESSIONS = 30
MAX_ACTIVE = 4                 # live hypotheses kept
NEW_CONF = 0.35
GATE_CONF = 0.65
ECHO_SIM = 0.85                # a message this close to a follow-up counts as taking it
SCOPE_NODES = 40
DELTA = {"support": 0.15, "support_probe": 0.30, "contradict": -0.25, "neutral": -0.05,
         "validated": 0.10, "weak": -0.05, "rejected": -0.30}
KINDS = ("artifact", "insight", "idea")

_SESSIONS: "OrderedDict[str, Session]" = OrderedDict()
_IDS = itertools.count(1)


# ───────────────────────── session state ─────────────────────────

class Session:
    def __init__(self, dataset: str, chat: Optional[dict], source: str):
        self.id = f"s{next(_IDS)}"
        self.at = time.time()
        self.dataset, self.chat, self.source = dataset, chat, source
        self.pos = 0                       # replay: user turns consumed
        self.turns: List[dict] = []        # conversation so far
        self.profile = {"goal": "", "stage": "", "facts": {"who": "", "what": "", "why": ""},
                        "hypotheses": [], "probes": [], "suggested": []}
        self.cycles: List[dict] = []
        self.status = "idle"               # idle | running | error | done
        self.pending: Optional[dict] = None  # typed turn in flight: {"user", "reply"} while the chat model answers
        self.error: Optional[str] = None
        self._hid, self._pid = itertools.count(1), itertools.count(1)
        self.lock = asyncio.Lock()
        _SESSIONS[self.id] = self
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)

    @property
    def user_turns_total(self) -> int:
        return sum(1 for t in (self.chat or {}).get("turns", []) if t["role"] == "user")

    def summary(self) -> dict:
        probes = [p for p in self.profile["probes"] if p["used"] is not None]
        tops = [c["eval"]["top_supported"] for c in self.cycles if c.get("eval", {}).get("top_supported") is not None]
        return {"id": self.id, "at": self.at, "source": self.source, "dataset": self.dataset,
                "status": self.status, "title": (self.chat or {}).get("title") or (self.turns[0]["content"][:80] if self.turns else ""),
                "author": (self.chat or {}).get("author"), "turn": len(self.cycles),
                "turns_total": self.user_turns_total if self.chat else None,
                "probes_offered": len(probes), "probes_used": sum(1 for p in probes if p["used"]),
                "top_supported": sum(tops), "top_scored": len(tops),
                "suggestions": len(self.profile["suggested"])}


def _live(s: Session) -> List[dict]:
    return [h for h in s.profile["hypotheses"] if h["status"] == "active"]


def _top(s: Session) -> Optional[dict]:
    live = _live(s)
    return max(live, key=lambda h: h["confidence"]) if live else None


def _bump(h: dict, delta: float, turn: int, source: str, why: str) -> None:
    h["confidence"] = round(min(1.0, max(0.0, h["confidence"] + delta)), 3)
    h["history"].append({"turn": turn, "delta": round(delta, 2), "source": source, "why": why[:200],
                         "conf": h["confidence"]})


# ───────────────────────── stage plumbing ─────────────────────────

class _Log:
    """Collects agent_loop emits during the dive, as the stage's sub-steps."""
    def __init__(self):
        self.events: List[dict] = []

    def emit(self, kind: str, title: str, detail: str = "", **data) -> dict:
        ev = {"kind": kind, "title": title, "detail": detail, "data": data}
        self.events.append(ev)
        return ev


def _stage(cycle: dict, name: str, status: str, title: str, detail: str = "", t0: Optional[float] = None, **data):
    cycle["stages"][name] = {"status": status, "title": title, "detail": detail, "data": data,
                             "ms": round((time.perf_counter() - t0) * 1000) if t0 else 0}


# ───────────────────────── prompts ─────────────────────────

_HYP = ('{"kind": "artifact" | "insight" | "idea", "claim": "<what would help and where it lives, at most 15 words>", '
        '"answer_type": "records" | "document" | "detail", '
        '"target": "chat" | "file" | "agent" | "step" | "tool" | "user" | "any", '
        '"scope": {"users": [], "teams": [], "agents": [], "files": [], "chats": [], "terms": []}, '
        '"query": "<search query in the user\'s own words, at most 15 words>"}')

_UPDATE_PROMPT = """You keep a running profile of what a user is working on in a chat, so a teammate assistant ("Ping") can point them to what would help, once it is sure:
- artifact: a document, file, repo or chat that has what they need
- insight: how someone else (maybe on another team) handled the same thing
- idea: an agent, tool or automation that could do it for them

KNOWN RECORDS in the team's knowledge base:
{vocab}

PROFILE SO FAR:
goal: {goal}
stage: {stage}
who: {who} | what: {what} | why: {why}
hypotheses about what would help (id, kind, confidence, claim):
{hyps}
follow-up questions offered after the last reply (id [hypothesis it tests]: text):
{probes}

LATEST EXCHANGE:
{exchange}

Update the profile from the NEW USER MESSAGE. Reply with JSON only:
{{"goal": "<one sentence>", "stage": "chitchat" | "exploring" | "working" | "wrapping_up",
 "facts": {{"who": "", "what": "", "why": ""}},
 "probe_used": "<id of the follow-up the user asked or clearly built on, else null>",
 "updates": [{{"id": "h1", "effect": "support" | "contradict" | "neutral", "why": "<short>"}}],
 "new": [{hyp}],
 "retire": ["<id>"]}}

Rules:
- updates: one per listed hypothesis. support = the message points the same way (mentions its records or terms, asks for that kind of help); contradict = it moves away (another system, tool or topic, or rules it out); neutral = no signal.
- new: only for needs the listed hypotheses don't cover, at most 2. Chats cover any subject, not only Agent Hub; don't assume agents or files unless the user says so.
- scope: ONLY names the user actually wrote. facts: only what the user stated; "" otherwise."""

_PROBE_PROMPT = """A user is chatting with an AI assistant. After each reply, 3 follow-up questions are offered that the user can click to send.

GOAL so far: {goal}
LATEST REPLY (what the user just read):
{reply}

Open hypotheses about what the user ultimately needs (id, confidence, claim):
{hyps}

Write the 3 follow-ups. Each must be a natural next question for the user given the reply, written as the user would ask it, and useful on its own. Where you can, make one help tell the open hypotheses apart, and say which hypothesis it tests.
Reply with JSON only: {{"probes": [{{"text": "...", "tests": "<hypothesis id or null>"}}, ...]}}"""


def _hyp_lines(s: Session) -> str:
    live = _live(s)
    return "\n".join(f"  {h['id']} ({h['kind']}, {h['confidence']:.2f}): {h['claim']}" for h in live) or "  (none yet)"


def _probe_lines(probes: List[dict]) -> str:
    return "\n".join(f"  {p['id']} [{p['tests'] or '-'}]: {p['text']}" for p in probes) or "  (none)"


def _clip(t: str, n: int) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n] + "…"


# ───────────────────────── the cycle ─────────────────────────

async def run_turn(s: Session, user_msg: str, reply: Optional[str], next_msg: Optional[str] = None) -> dict:
    """One full cycle for a user message (and, in a replay, the reply that followed it)."""
    from retrieval import vector_provider as vp
    L = A._layers(s.dataset)
    turn = len(s.cycles) + 1
    prev_reply = next((t["content"] for t in reversed(s.turns) if t["role"] == "assistant"), None)
    s.turns.append({"role": "user", "content": user_msg})
    cycle = {"turn": turn, "user": user_msg, "reply": reply, "stages": {k: {"status": "pending"} for k in STAGES},
             "focus": {}, "eval": {}, "at": time.time()}
    s.cycles.append(cycle)
    s.pending = None   # the cycle now carries this turn (message + reply)
    exclude = set(s.profile["suggested"])
    if s.chat:
        exclude.add(s.chat["doc"])     # a replayed chat must not find itself
        if s.chat.get("author"):       # ...nor point the user at their own activity page
            exclude.add(f"person-{s.chat['author']}.md")
    P = s.profile

    # 1 observe ─────────────────────────────────────────────
    t0 = time.perf_counter()
    offered = [p for p in P["probes"] if p["turn"] == turn - 1]
    sims = {}
    if offered:
        qv = await asyncio.to_thread(vp._embed_query, user_msg)
        for p in offered:
            pv = await asyncio.to_thread(vp._embed_query, p["text"])
            if qv and pv:
                sims[p["id"]] = round(sum(a * b for a, b in zip(qv, pv)), 3)
    echo = max(sims, key=sims.get) if sims else None
    last_sugg = next((c.get("suggestion") for c in reversed(s.cycles[:-1]) if c.get("suggestion")), None)
    _stage(cycle, "observe", "done", f"User turn {turn}" + (f" · closest follow-up {echo} ({sims[echo]:.2f})" if echo else ""),
           _clip(user_msg, 240), t0, sims=sims, offered=[p["id"] for p in offered],
           last_suggestion=last_sugg and last_sugg["doc"])

    # 2 update profile ──────────────────────────────────────
    t0 = time.perf_counter()
    exchange = (f"ASSISTANT (previous reply): {_clip(prev_reply, 1500)}\n" if prev_reply else "") + \
               f"USER (new message): {_clip(user_msg, 2500)}"
    u = await A._llm_json(_UPDATE_PROMPT.format(
        vocab=A._vocabulary(L), goal=P["goal"] or "(unknown)", stage=P["stage"] or "(unknown)",
        who=P["facts"]["who"] or "?", what=P["facts"]["what"] or "?", why=P["facts"]["why"] or "?",
        hyps=_hyp_lines(s), probes=_probe_lines(offered), exchange=exchange, hyp=_HYP),
        "You maintain a user profile from a conversation. Output JSON only.")
    P["goal"] = str(u.get("goal") or P["goal"]).strip()[:300]
    P["stage"] = u.get("stage") if u.get("stage") in ("chitchat", "exploring", "working", "wrapping_up") else (P["stage"] or "exploring")
    for k in ("who", "what", "why"):
        v = str((u.get("facts") or {}).get(k) or "").strip()
        if v and v.lower() not in ("unknown", "none", "n/a", "?"):
            P["facts"][k] = v[:200]
    used = u.get("probe_used") if isinstance(u.get("probe_used"), str) else None
    if echo and sims[echo] >= ECHO_SIM:
        used = used or echo
    used = used if used in {p["id"] for p in offered} else None
    for p in offered:
        p["used"] = p["id"] == used
        p["sim"] = sims.get(p["id"])
    changes, by_id = [], {h["id"]: h for h in _live(s)}
    for up in u.get("updates") or []:
        h = by_id.get(str(up.get("id")))
        if not h or not isinstance(up, dict):
            continue
        eff = up.get("effect") if up.get("effect") in ("support", "contradict", "neutral") else "neutral"
        via_probe = eff == "support" and used and next((p["tests"] for p in offered if p["id"] == used), None) == h["id"]
        key = "support_probe" if via_probe else eff
        _bump(h, DELTA[key], turn, "follow-up taken" if via_probe else "message", str(up.get("why") or eff))
        if eff == "support":
            h["confirmed"] = True
        changes.append({"id": h["id"], "effect": key, "why": str(up.get("why") or "")[:200], "conf": h["confidence"]})
    for hid in u.get("retire") or []:
        h = by_id.get(str(hid))
        if h:
            h["status"] = "retired"
            h["history"].append({"turn": turn, "delta": 0, "source": "retired", "why": "the model retired it", "conf": h["confidence"]})
            changes.append({"id": h["id"], "effect": "retired", "why": "", "conf": h["confidence"]})
    born = []
    for raw in (A._clean_hyps(u.get("new")) if u.get("new") else []):
        kind = next((x.get("kind") for x in (u.get("new") or []) if isinstance(x, dict) and x.get("claim", "").strip()[:300] == raw["claim"]), None)
        h = {**raw, "id": f"h{next(s._hid)}", "kind": kind if kind in KINDS else "artifact", "confidence": NEW_CONF,
             "status": "active", "born": turn, "confirmed": False, "history": [], "l1": None, "top_streak": 0}
        h["history"].append({"turn": turn, "delta": NEW_CONF, "source": "new", "why": "proposed", "conf": NEW_CONF})
        P["hypotheses"].append(h)
        born.append(h["id"])
    live = sorted(_live(s), key=lambda h: -h["confidence"])
    for h in live[MAX_ACTIVE:]:
        h["status"] = "retired"
        h["history"].append({"turn": turn, "delta": 0, "source": "retired", "why": f"more than {MAX_ACTIVE} live", "conf": h["confidence"]})
    # ground truth for last turn's prediction: did this real message support last turn's top hypothesis?
    prev_top = (s.cycles[-2]["focus"].get("top") if len(s.cycles) > 1 else None)
    if prev_top:
        eff = next((c["effect"] for c in changes if c["id"] == prev_top), "neutral")
        s.cycles[-2]["eval"]["top_supported"] = eff.startswith("support")
        s.cycles[-2]["eval"]["top_effect"] = eff
    if offered:
        s.cycles[-2]["eval"]["probe_used"] = used
    _stage(cycle, "update", "done", f"Profile updated · {len(changes)} change(s), {len(born)} new",
           P["goal"], t0, goal=P["goal"], stage=P["stage"], facts=dict(P["facts"]), probe_used=used,
           changes=changes, born=born)

    # 3 check on L1 ─────────────────────────────────────────
    t0 = time.perf_counter()
    convo = A._convo(s.turns)
    checks, focus = [], {"hyps": {}}
    for h in _live(s):
        v = await asyncio.to_thread(A._validate, L, {k: h[k] for k in ("claim", "answer_type", "target", "scope", "query")},
                                    convo, user_msg, exclude)
        h["target"] = v["_h"]["target"]
        h["scope"] = v["_h"]["scope"]
        _bump(h, DELTA[v["verdict"]], turn, "L1 check", v["why"])
        h["l1"] = {"verdict": v["verdict"], "why": v["why"], "scope": len(v["scope"] or []) if v["scope"] is not None else None}
        h["_v"] = v
        if h["confidence"] < 0.10:
            h["status"] = "retired"
            h["history"].append({"turn": turn, "delta": 0, "source": "retired", "why": "confidence below 0.10", "conf": h["confidence"]})
        seeds = [n for k in ("users", "teams", "agents", "files", "chats", "topics") for n in (v.get("resolved") or {}).get(k, [])]
        scope_nodes = [n for d in (v["scope"] or [])[:SCOPE_NODES] for n in L["doc_nodes"].get(L["alias"].get(d, d), [])[:1]]
        hits = [n for c in (v.get("passages") or v.get("cards") or []) for n in L["doc_nodes"].get(L["alias"].get(c["doc"], c["doc"]), [])[:1]]
        focus["hyps"][h["id"]] = {"conf": h["confidence"], "seeds": seeds, "scope": scope_nodes, "hits": hits,
                                  "verdict": v["verdict"], "status": h["status"]}
        checks.append({"id": h["id"], "verdict": v["verdict"], "why": v["why"], "conf": h["confidence"],
                       "scope": h["l1"]["scope"], "dropped": v.get("dropped"), "missing": v.get("missing"),
                       "retarget": v.get("retarget"), "status": h["status"]})
    top = _top(s)
    for h in _live(s):
        h["top_streak"] = (h["top_streak"] + 1) if top and h["id"] == top["id"] else 0
    focus["top"] = top["id"] if top else None
    cycle["focus"] = focus
    _stage(cycle, "check", "done", f"{sum(c['verdict'] == 'validated' for c in checks)} of {len(checks)} held up on the graph",
           f"top: {top['id']} ({top['confidence']:.2f}) {top['claim']}" if top else "no live hypothesis", t0, checks=checks)

    # 4 probe ───────────────────────────────────────────────
    t0 = time.perf_counter()
    open_h = [h for h in _live(s)]
    if reply is None and not open_h:
        _stage(cycle, "probe", "skip", "No reply and nothing to test", "", t0)
    else:
        pr = await A._llm_json(_PROBE_PROMPT.format(goal=P["goal"] or "(unknown)",
                                                    reply=_clip(reply or f"(no reply yet; the user said) {user_msg}", 2500),
                                                    hyps=_hyp_lines(s)),
                               "You write follow-up questions for a chat. Output JSON only.")
        new_probes = []
        ids = {h["id"] for h in open_h}
        for x in (pr.get("probes") or [])[:3]:
            if isinstance(x, dict) and str(x.get("text") or "").strip():
                p = {"id": f"p{next(s._pid)}", "text": str(x["text"]).strip()[:240],
                     "tests": x.get("tests") if x.get("tests") in ids else None, "turn": turn, "used": None, "sim": None}
                P["probes"].append(p)
                new_probes.append(p)
        _stage(cycle, "probe", "done", f"{len(new_probes)} follow-ups offered, {sum(1 for p in new_probes if p['tests'])} test a hypothesis",
               "shown after the reply; next turn checks whether the user took one", t0, probes=new_probes)
        if next_msg is not None:
            cycle["eval"]["next_user"] = _clip(next_msg, 300)

    # 5 gate ────────────────────────────────────────────────
    t0 = time.perf_counter()
    crit = []
    if top:
        crit = [
            {"name": "about work", "ok": P["stage"] in ("exploring", "working") and bool(P["facts"]["what"]),
             "value": f"stage {P['stage']}; what: {P['facts']['what'] or '?'}"},
            {"name": f"confidence ≥ {GATE_CONF}", "ok": top["confidence"] >= GATE_CONF, "value": f"{top['confidence']:.2f}"},
            {"name": "held up on L1 this turn", "ok": (top.get("l1") or {}).get("verdict") == "validated",
             "value": (top.get("l1") or {}).get("verdict")},
            {"name": "confirmed by the user", "ok": bool(top.get("confirmed")),
             "value": "a supporting message or follow-up" if top.get("confirmed") else "not yet"},
            {"name": "not a repeat", "ok": not any(c.get("suggestion", {}) and c["suggestion"].get("hypothesis") == top["id"]
                                                   for c in s.cycles[-3:-1]),
             "value": "no suggestion from it in the last 2 turns"},
        ]
    open_ = bool(crit) and all(c["ok"] for c in crit)
    _stage(cycle, "gate", "done", "Open: enough to suggest" if open_ else "Closed: keep listening",
           f"{sum(c['ok'] for c in crit)} of {len(crit)} criteria" if crit else "no hypothesis yet", t0,
           open=open_, criteria=crit, hypothesis=top and top["id"])

    # 6 dive L2 / 7 suggest ─────────────────────────────────
    cycle["suggestion"] = None
    if not open_:
        _stage(cycle, "dive", "skip", "Not diving: the gate is closed", "")
        _stage(cycle, "suggest", "skip", "Stayed quiet", "the gate is closed")
    else:
        t0 = time.perf_counter()
        log = _Log()
        h = top
        hv = {k: h[k] for k in ("claim", "answer_type", "target", "scope", "query")}
        docs = await asyncio.to_thread(A._execute, log, L, hv, h["_v"], user_msg, exclude)
        goal = {"goal": P["goal"], "need": h["claim"]}
        chk = await A._dive_and_check(log, L, goal, hv, docs, user_msg, exclude)
        if chk.get("best"):
            pick = docs[chk["best"] - 1]
            docs = [pick] + [d for d in docs if d is not pick]
        ev_nodes = [n for e in log.events if e["kind"] == "evidence" for x in e["data"].get("evidence", [])
                    for n in L["doc_nodes"].get(L["alias"].get(x["doc"], x["doc"]), [])[:1]]
        cycle["focus"]["evidence"] = ev_nodes
        _stage(cycle, "dive", "done", f"{len(docs)} doc(s) via the {h['answer_type']} route · check: {chk['verdict']}",
               chk.get("why", ""), t0, steps=log.events, check=chk,
               docs=[{"doc": d["doc"], "title": A._title(L, d["doc"]), "via": d["via"]} for d in docs[:6]])
        t0 = time.perf_counter()
        pick = next((d for d in docs if d["doc"] not in P["suggested"]), None)
        if pick and chk["verdict"] in ("answers", "partial"):
            sug = {"kind": h["kind"], "doc": pick["doc"], "title": A._title(L, pick["doc"]), "why": chk.get("why", ""),
                   "hypothesis": h["id"], "verdict": chk["verdict"]}
            cycle["suggestion"] = sug
            P["suggested"].append(pick["doc"])
            cycle["focus"]["suggestion"] = L["doc_nodes"].get(pick["doc"], [])[:1]
            _stage(cycle, "suggest", "done", f"Suggested ({h['kind']}): {sug['title']}", sug["why"], t0, suggestion=sug)
        else:
            _stage(cycle, "suggest", "skip", "Stayed quiet: the evidence didn't meet the need",
                   chk.get("why", ""), t0, check=chk)

    # 8 learn ───────────────────────────────────────────────
    t0 = time.perf_counter()
    snap = [{"id": h["id"], "kind": h["kind"], "claim": h["claim"], "conf": h["confidence"], "status": h["status"],
             "l1": (h.get("l1") or {}).get("verdict")} for h in P["hypotheses"]]
    prev = {x["id"]: x["conf"] for x in (s.cycles[-2]["snapshot"] if len(s.cycles) > 1 else [])}
    moves = [f"{x['id']} {prev.get(x['id'], 0):.2f}→{x['conf']:.2f}" for x in snap
             if abs(x["conf"] - prev.get(x["id"], 0)) >= 0.01 and x["status"] == "active"]
    cycle["snapshot"] = snap
    _stage(cycle, "learn", "done", "Profile after this turn", "; ".join(moves) or "no change", t0, moves=moves)
    for h in P["hypotheses"]:
        h.pop("_v", None)
    if reply:
        s.turns.append({"role": "assistant", "content": reply})
    return cycle


# ───────────────────────── replay data (real chats) ─────────────────────────

def _real_chats(dataset: str) -> List[dict]:
    from assist import replayable_chats
    return replayable_chats(dataset=DS.get(dataset).id.split("~")[0], min_user_turns=2)


def _turn_pair(chat: dict, k: int) -> tuple:
    """The k-th user message of a chat, the reply after it, and the next user message."""
    users = [i for i, t in enumerate(chat["turns"]) if t["role"] == "user"]
    if k >= len(users):
        return None, None, None
    i = users[k]
    end = users[k + 1] if k + 1 < len(users) else len(chat["turns"])
    reply = next((t["content"] for t in chat["turns"][i + 1:end] if t["role"] == "assistant"), None)
    nxt = chat["turns"][users[k + 1]]["content"] if k + 1 < len(users) else None
    return chat["turns"][i]["content"], reply, nxt


# ───────────────────────── API ─────────────────────────

class NewSession(BaseModel):
    dataset: Optional[str] = None
    chat_id: Optional[str] = None
    source: str = "agent-page"


class Say(BaseModel):
    message: str


def _state(s: Session) -> dict:
    from retrieval import lightrag_provider as lp
    return {**s.summary(), "error": s.error, "chat": s.chat and {k: s.chat[k] for k in ("chat_id", "doc", "title", "author")},
            "turns": s.turns, "profile": s.profile, "cycles": s.cycles, "stages": STAGES,
            "pending": s.pending, "chat_model": lp.LLM}


# The chat model the user talks to: a plain assistant, as in Open WebUI. It has no access
# to the knowledge base -- surfacing what the team already has is Ping's job, alongside it.
CHAT_SYSTEM = ("You are a helpful AI assistant in a team's chat app. Answer the user directly and concisely, "
               "using markdown where it helps. You do not have access to the team's internal documents, files or past chats.")


async def _chat_reply(s: Session, msg: str) -> str:
    from retrieval import lightrag_provider as lp
    await lp._ensure_session()
    hist = [{"role": t["role"], "content": t["content"][:4000]} for t in s.turns[-8:]]
    out = await lp._ollama_llm(msg, system_prompt=CHAT_SYSTEM, history_messages=hist)
    return (out or "").strip() or "(the chat model returned nothing)"


async def _run(s: Session, msg: str, reply: Optional[str], nxt: Optional[str], chat: bool = False):
    """One turn. chat=True (a typed message): the chat model answers first, then the cycle runs on both."""
    async with s.lock:
        s.status, s.error = "running", None
        try:
            if chat:
                s.pending = {"user": msg, "reply": None}
                reply = await _chat_reply(s, msg)
                s.pending = {"user": msg, "reply": reply}
            await run_turn(s, msg, reply, nxt)
            s.status = "done" if s.chat and s.pos >= s.user_turns_total else "idle"
        except Exception as e:
            s.status, s.error = "error", f"{type(e).__name__}: {e}"[:300]
            if s.cycles:
                s.cycles[-1]["error"] = s.error
        finally:
            s.pending = None


@router.get("/chats")
def chats(dataset: Optional[str] = None):
    return [{k: c[k] for k in ("chat_id", "doc", "title", "author", "user_turns")} for c in _real_chats(dataset)]


@router.post("/sessions")
def new_session(req: NewSession):
    ds = DS.get(req.dataset)
    base = ds.id.split("~")[0]
    chat = None
    if req.chat_id:
        chat = next((c for c in _real_chats(base) if c["chat_id"] == req.chat_id), None)
        if not chat:
            raise HTTPException(404, "no such replayable chat in this export")
    s = Session(base, chat, req.source)
    return _state(s)


@router.post("/sessions/{sid}/next")
async def replay_next(sid: str, wait: bool = False):
    s = _SESSIONS.get(sid)
    if not s or not s.chat:
        raise HTTPException(404, "no such replay session")
    if s.lock.locked():
        raise HTTPException(409, "a turn is already running")
    msg, reply, nxt = _turn_pair(s.chat, s.pos)
    if msg is None:
        raise HTTPException(400, "end of chat")
    s.pos += 1
    task = _run(s, msg, reply, nxt)
    if wait:
        await task
    else:
        asyncio.create_task(task)
    return {"id": s.id, "turn": s.pos}


@router.post("/sessions/{sid}/say")
async def say(sid: str, req: Say, wait: bool = False):
    s = _SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session")
    if s.lock.locked():
        raise HTTPException(409, "a turn is already running")
    task = _run(s, req.message.strip(), None, None, chat=True)
    if wait:
        await task
    else:
        asyncio.create_task(task)
    return {"id": s.id}


@router.get("/sessions")
def sessions():
    return [s.summary() for s in reversed(list(_SESSIONS.values()))]


@router.get("/sessions/{sid}")
def session(sid: str):
    s = _SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "no such session (sessions are kept in memory and lost on restart)")
    return _state(s)


@router.delete("/sessions")
def reset():
    """Clear every Ping session and the agent's run list: a clean dashboard."""
    _SESSIONS.clear()
    A._RUNS.clear()
    return {"ok": True}
