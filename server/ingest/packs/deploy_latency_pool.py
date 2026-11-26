"""
Planted-artifact pack: "service latency spikes about ten minutes after each deploy".

A cross-team operations problem whose answer is spread over several artifacts,
with a distractor. Everything here is fictional test content (made-up service
names, hosts and private-range IPs), planted by ingest/plant.py into a COPY of an
export so the real data stays clean. Planted users are new pseudonyms with teams
(Platform / Backend / SRE), so team-aware suggestions can be tested too.

The coherent root cause, carried in pieces across the artifacts:
  The shared base deploy chart never sets the connection pool's max-lifetime, so
  it stays at the 30-minute library default. The database proxy (PgBouncer) closes
  idle server connections after server_idle_timeout = 600 s (ten minutes). Right
  after a deploy, traffic keeps the pooled connections busy and healthy; as load
  settles, about ten minutes later PgBouncer closes the now-idle backends. The app
  pool still believes they are valid (a recent base-chart change dropped the
  connection test query to shave latency), so it hands out dead connections. Each
  request then pays a socket timeout plus a retry, and p99 latency climbs while the
  pool slowly rebuilds. Fix: set max-lifetime below the proxy idle timeout and/or
  re-enable keepalive. Permanent fix is PLAT-412 (a safe default in the base chart).

SPOILER: ANSWER_KEY at the bottom says which artifacts should be found for the
test problems. Don't read it before testing.
"""

NAME = "deploy-latency-pool"

USERS = {"user-19": "Platform", "user-20": "Backend", "user-21": "SRE"}

# ───────────────────────── files (PDFs and code) ─────────────────────────

RUNBOOK = """SERVICE DEPLOY HEALTH RUNBOOK (v2)
Owner: Platform team. Last revised 2026-11-05.

1. Purpose
This runbook covers a healthy rollout of a backend service through the shared
deploy pipeline: how the base chart and the per-service overlay combine, the
post-deploy checks, and the regressions we have seen often enough to list here.

2. Before you deploy
- Announce the rollout in the deploys channel.
- Confirm the per-service overlay is pinned to a reviewed values file.
- Check the service's connection-pool and timeout overrides are present; the base
  chart does not set them all (see Known issues KI-7).

3. Rollout sequence
3.1 From the deploy host run: rollout.sh --service <name> --env prod --confirm
3.2 The script renders platform-base/values.yaml plus the service overlay, applies
    the manifest, and waits for the new pods to pass readiness.
3.3 Expected duration: 4 to 8 minutes for a rolling update.

4. Post-deploy checks
4.1 All pods Ready and the old replica set scaled to zero.
4.2 p99 latency on the service dashboard is within its normal band.
4.3 Connection-pool gauge (active/idle) is stable, not climbing.
4.4 Re-check p99 at the 15-minute mark, not only right after the rollout (see KI-7).

5. Known issues
KI-5 Readiness flaps during the first minute.
  Symptom: pods briefly report not-ready just after start. Cause: warm-up before the
  JIT settles. Fix: none needed; it clears within a minute. Do not raise the probe
  failure threshold to hide it.

KI-7 p99 latency spikes about ten minutes after a deploy, then stays high.
  Symptom: right after the rollout latency is normal; roughly ten minutes later p99
  climbs sharply and does not recover on its own. Restarting the pods resets the
  clock: it is fine again for about ten minutes, then spikes.
  Cause: the base chart (platform-base/values.yaml) does not set the pool
  max-lifetime, so it stays at the 30-minute default. The database proxy closes
  idle server connections at server_idle_timeout = 600 s. Fresh connections from the
  deploy stay busy and healthy for a while; once load settles, at about the
  ten-minute mark the proxy closes the idle backends. The pool keeps handing out
  those connections (a recent base change removed the connection test query), so
  each request pays a socket timeout and a retry before a good connection is made.
  Fix: in the service overlay set the pool max-lifetime below the proxy idle
  timeout (maxLifetime: 570000, i.e. 9m30s) and re-enable keepaliveTime: 120000.
  Verify: watch the pool gauge after the fix; idle connections should be recycled
  before ten minutes and p99 should stay flat past the 15-minute mark.
  Permanent fix is tracked in PLAT-412 (set a safe max-lifetime default in the base
  chart so every service inherits it).

6. Contacts
Platform on-call, rotation pinned in the team channel.
"""

BASE_VALUES = """PLATFORM BASE DEPLOY VALUES (platform-base)
Platform team. Version 2026-10.

Scope
The base Helm values every backend service inherits, overlaid by each service's own
values file. Intended to carry safe, boring defaults so services only override what
is special about them.

Connection pool (datasource.hikari)
  minimumIdle: 2
  maximumPoolSize: 20
  connectionTimeout: 3000
  idleTimeout: 600000
  # maxLifetime: intentionally unset here -> library default (1800000, 30 min)
  # connectionTestQuery: removed 2026-10-12 to shave ~1ms/borrow; validation now
  #   relies on the driver's isValid(), which does not catch a proxy-closed socket

Database proxy (PgBouncer)
  pool_mode: transaction
  server_idle_timeout: 600        # seconds; idle server connections are closed at 10 min
  server_lifetime: 3600

Ownership
Each service owns its overlay. Anything not safe as a shared default (pool
max-lifetime, keepalive, per-service timeouts) is set in the service overlay, not
here. Changes to the base values require a Platform ticket and a new chart release.

Note
Because max-lifetime is unset, a service that also does not set it in its overlay
inherits the 30-minute default, which is longer than the proxy's 600 s idle timeout.
See the deploy runbook, KI-7.
"""

ROLLOUT_SCRIPT = """#!/usr/bin/env bash
# rollout.sh: deploy a backend service through the shared pipeline.
# SRE team. Usage: rollout.sh --service <name> --env prod --confirm
set -euo pipefail

SERVICE=""
ENV="prod"
BASE_VALUES="platform-base/values.yaml"
OVERLAY_DIR="services"

log() { echo "[$(date +%H:%M:%S)] $*"; }

render_values() {
  log "rendering base + overlay for $SERVICE"
  # NOTE: pool overrides a service adds to its own overlay are merged on top of the
  # base; anything the base leaves unset (e.g. hikari.maxLifetime) falls through to
  # the library default unless the overlay sets it.
  helm template "$SERVICE" platform-base \\
    -f "$BASE_VALUES" \\
    -f "$OVERLAY_DIR/$SERVICE/values-$ENV.yaml" > "/tmp/$SERVICE.manifest.yaml"
}

apply_and_wait() {
  log "applying manifest for $SERVICE ($ENV)"
  kubectl apply -f "/tmp/$SERVICE.manifest.yaml"
  kubectl rollout status "deploy/$SERVICE" --timeout=480s
}

[[ "${1:-}" == "--service" && -n "${2:-}" ]] || { echo "usage: $0 --service <name> --env prod --confirm"; exit 2; }
SERVICE="$2"
[[ "${3:-}" == "--env" && -n "${4:-}" && "${5:-}" == "--confirm" ]] || { echo "usage: $0 --service <name> --env prod --confirm"; exit 2; }
ENV="$4"
render_values
apply_and_wait
log "rollout complete; run the post-deploy checks in the runbook (re-check p99 at 15 min)"
"""

VALIDATOR = '''"""deploy_latency_validator.py: post-deploy checks for a backend service.

SRE / Automation team. Run after rollout.sh. Flags the regressions the deploy
runbook lists as known issues, so nobody finds them ten minutes into prod traffic:
a pool max-lifetime that outlives the DB proxy idle timeout, missing keepalive, and
p99 already drifting above its band.
"""
from __future__ import annotations

import sys

import requests

# where the service exposes its config + pool metrics (Prometheus-style)
METRICS = "http://127.0.0.1:9090"
PROXY_IDLE_TIMEOUT_S = 600          # PgBouncer server_idle_timeout from the base values
P99_BUDGET_MS = 400


def gauge(name: str) -> float:
    """Latest value of a single numeric gauge from the service's /metrics-json."""
    r = requests.get(f"{METRICS}/metrics-json", timeout=5)
    r.raise_for_status()
    return float(r.json().get(name))


def pool_config() -> dict:
    r = requests.get(f"{METRICS}/config/datasource", timeout=5)
    r.raise_for_status()
    return r.json()


def problems() -> list[str]:
    out = []
    cfg = pool_config()
    max_lifetime_ms = int(cfg.get("maxLifetime", 1800000))   # 30-min library default if unset
    if max_lifetime_ms >= PROXY_IDLE_TIMEOUT_S * 1000:
        out.append(f"pool maxLifetime {max_lifetime_ms}ms >= proxy idle timeout "
                   f"{PROXY_IDLE_TIMEOUT_S}s: idle backends get closed before the pool recycles "
                   f"them (deploy runbook KI-7). Set maxLifetime below {PROXY_IDLE_TIMEOUT_S * 1000}.")
    if not cfg.get("keepaliveTime"):
        out.append("pool keepaliveTime unset: nothing probes idle connections between uses.")
    p99 = gauge("http_request_duration_ms_p99")
    if p99 > P99_BUDGET_MS:
        out.append(f"p99 {p99:.0f}ms over budget {P99_BUDGET_MS}ms.")
    return out


def main() -> int:
    probs = problems()
    for p in probs:
        print("FAIL", p)
    print("OK" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
'''

FILES = [
    {"key": "runbook", "user": "user-19", "filename": "Service-Deploy-Health-Runbook-v2.pdf",
     "content_type": "application/pdf", "when": "2026-11-05 14:20", "text": RUNBOOK, "pdf": True},
    {"key": "base_values", "user": "user-19", "filename": "Platform-Base-Deploy-Values.pdf",
     "content_type": "application/pdf", "when": "2026-10-13 09:30", "text": BASE_VALUES, "pdf": True},
    {"key": "rollout_script", "user": "user-21", "filename": "rollout.sh", "content_type": "text/x-shellscript",
     "when": "2026-10-20 11:15", "text": ROLLOUT_SCRIPT},
    {"key": "validator", "user": "user-21", "filename": "deploy_latency_validator.py", "content_type": "text/x-python",
     "when": "2026-11-07 16:40", "text": VALIDATOR},
]

# ───────────────────────── chats ─────────────────────────

CHATS = [
    {"key": "backend_p99", "user": "user-20", "title": "p99 climbs a few minutes after every release",
     "when": "2026-11-06 10:05", "model": "gemma-4-31b-it",
     "tags": ["backend", "latency", "database", "deploys"],
     "summary": "A Backend user saw p99 latency on their service climb about ten minutes after every release, with the pool "
                "showing connection errors. They traced it to the DB proxy closing idle connections the pool still thought "
                "were live, and fixed it by setting maxLifetime in their service overlay. They note the base chart does not "
                "set it.",
     "pain_points": ["recurs on every deploy", "fix lives in each service, not the base chart"],
     "turns": [
         ("Every time we release, p99 on our orders service is fine for about ten minutes and then jumps from ~120ms to "
          "well over a second and stays there. Nothing in the release itself touches the hot path. Rolling the pods restarts "
          "the clock: good for ten minutes, then the spike again. What would do that?",
          "A spike that starts at a fixed delay after every deploy and resets when you restart the pods points at something "
          "with a timer that starts when the process does, not at your code path. The usual culprits:\n\n"
          "1. **Connection pool vs. a downstream idle timeout**: fresh connections from startup stay busy and healthy, then an "
          "idle timeout somewhere on the path closes them and the pool keeps using the dead ones.\n"
          "2. A cache that expires on a fixed TTL and stampedes.\n\n"
          "Since restarting resets it and it's a fixed ~10 min, check your pool metrics: do active/idle counts and connection "
          "errors start climbing right when p99 does?"),
         ("Yes - at the ten minute mark the pool starts logging connection resets and 'connection is not available' waits, "
          "then p99 follows. Our DB goes through PgBouncer.",
          "That's the signature. PgBouncer closes idle server connections at `server_idle_timeout`, which defaults to 600 "
          "seconds here. Your pool's `maxLifetime` is longer than that (the 30-minute library default if you never set it), "
          "so the pool holds connections PgBouncer has already closed and hands them out; each one fails, waits, and retries. "
          "Set the pool `maxLifetime` below the proxy idle timeout - e.g. 570000ms - so the pool recycles connections before "
          "PgBouncer drops them, and turn on `keepaliveTime` so idle ones get probed."),
         ("Set maxLifetime: 570000 and keepaliveTime: 120000 in our overlay and the spike is gone - flat past 30 minutes now. "
          "Surprised this isn't just in the base chart; every service is going to hit it.",
          "Agreed - it belongs in the base values so nobody has to rediscover it. Worth filing a Platform ticket to set a safe "
          "max-lifetime default there, and adding a post-deploy check that re-reads p99 at the 15-minute mark instead of right "
          "after the rollout."),
     ]},
    {"key": "platform_basechart", "user": "user-19", "title": "Add a pool max-lifetime default to the base chart",
     "when": "2026-11-09 15:40", "model": "gemma-4-31b-it", "attach": {0: ["rollout_script"]},
     "tags": ["platform", "helm", "database", "deploys"],
     "summary": "A Platform user wrote the base-chart change for PLAT-412: set a safe hikari maxLifetime and keepaliveTime in "
                "platform-base so every service inherits a value below the PgBouncer idle timeout, and re-enable a lightweight "
                "validation. They attached the rollout script for context.",
     "pain_points": [],
     "turns": [
         ("Backend hit the ten-minute post-deploy latency spike and traced it to pool maxLifetime being longer than our "
          "PgBouncer server_idle_timeout (600s). I want to fix it once in the base chart (PLAT-412) instead of per service. "
          "The rollout script that renders base+overlay is attached. What should the base hikari block be?",
          "Set a safe default below the proxy idle timeout so every service inherits it, and let overlays still override:\n\n"
          "```yaml\ndatasource:\n  hikari:\n    minimumIdle: 2\n    maximumPoolSize: 20\n    connectionTimeout: 3000\n"
          "    idleTimeout: 540000        # 9 min, < server_idle_timeout\n"
          "    maxLifetime: 570000        # 9m30s, < PgBouncer server_idle_timeout 600s\n"
          "    keepaliveTime: 120000      # probe idle connections every 2 min\n"
          "    validationTimeout: 2000\n```\n\n"
          "Since the test query was removed, `keepaliveTime` is what now catches a proxy-closed socket before a borrow. Ship it "
          "as a minor chart bump and call it out in the changelog so teams know they can drop their per-service override."),
         ("Good. I'll release it as a chart minor and update the deploy runbook KI-7 to say the base chart now sets this.",
          "Right. Keep the KI-7 verification next to it (watch the pool gauge recycle connections before ten minutes, p99 flat "
          "past fifteen), and once everyone is on the new chart the per-service maxLifetime overrides can be removed."),
     ]},
    {"key": "backend_nightly", "user": "user-20", "title": "Service slow overnight, no deploy",
     "when": "2026-10-29 08:50", "model": "gemma-4-31b-it",
     "tags": ["backend", "latency", "jvm", "memory"],
     "summary": "A Backend user's service got gradually slower overnight with no deploy; it was a slow heap leak driving long "
                "GC pauses, not a connection-pool or deploy issue. A heap bump and a leak fix resolved it.",
     "pain_points": ["gradual, not sudden", "unrelated to deploys"],
     "turns": [
         ("Our reporting service got slower and slower overnight - p99 drifted up over about eight hours. No deploy, no traffic "
          "spike. By morning it was crawling. A restart fixed it immediately.",
          "A slow drift over hours that a restart clears, with no deploy, usually isn't the connection pool - that bites at a "
          "fixed delay after a deploy. A gradual climb points at something that accumulates while the process runs: a memory "
          "leak driving longer and longer GC pauses, an unbounded cache, or a file/thread handle leak. Check the heap-used and "
          "GC-pause graphs overnight: if used heap trends up and pauses lengthen as p99 rises, it's a leak."),
         ("Heap used climbed all night and old-gen GC pauses went from 40ms to over a second by morning. Found an unbounded "
          "cache keyed by request id. Bounded it and bumped the heap; steady now.",
          "That's the one - an unbounded cache is a classic slow leak. The bound is the real fix; the heap bump just buys "
          "headroom. Add an alert on used-heap-after-GC trending up so you catch it before p99 does."),
     ]},
]

# ───────────────────────── Agent Hub ─────────────────────────

AGENTS = [
    {"key": "validator_agent", "owner": "user-21", "name": "Deploy Latency Validator", "slug": "deploy-latency-validator",
     "when": "2026-11-09 10:00", "manual_time_minutes": 30,
     "description": "Runs the post-deploy checks on a backend service and flags connection-pool, keepalive and p99 "
                    "regressions before prod traffic finds them.",
     "tags": ["deploys", "latency", "database", "validation"],
     "system": "You validate a backend service after a deploy. Ask for (or run) deploy_latency_validator.py and read its "
               "output. For every FAIL line, explain the likely cause using the Service Deploy Health runbook's known issues "
               "(KI-7: pool max-lifetime outliving the DB proxy idle timeout; missing keepalive) and give the exact overlay "
               "fix and how to verify it. Never declare a deploy healthy while p99 is still climbing at the ten-minute mark.",
     "skills": [{"name": "pool-metrics-audit",
                 "description": "Read connection-pool and proxy metrics to explain why latency climbs minutes after a deploy",
                 "content": "---\nname: pool-metrics-audit\n---\nWhen latency is fine right after a deploy and spikes at a fixed "
                            "delay (often ~10 min), compare the pool's maxLifetime to the database proxy's idle timeout. If "
                            "maxLifetime is longer, the proxy closes idle backends the pool still hands out; each borrow of a "
                            "dead connection costs a timeout plus a retry. Pool 'connection is not available' waits and "
                            "connection-reset counts rising together with p99 confirm it. The fix is to recycle connections "
                            "before the proxy does (maxLifetime below the proxy idle timeout) and probe idle ones (keepaliveTime)."}]},
]

INSTALLS = [("user-19", "validator_agent"), ("user-20", "validator_agent")]

# ───────────────────────── test problems (safe to read) ─────────────────────────

PROBLEMS = """\
You just deployed a backend service. It looked fine, then about ten minutes later p99
latency spiked and stayed high. Restarting the pods made it fine again for about ten
minutes, then it spiked once more.

Ways to test (type them on the Agent page; each starts or continues a session):
  1. Start vague, then add detail over a few messages, the way someone would in a real chat:
       "our service gets slow after we deploy"
       -> "it's fine right after, then latency spikes a bit later"
       -> "about ten minutes after the deploy, and restarting the pods resets it"
     Watch when the gate opens and what kind of thing Ping suggests (artifact / insight / idea).
  2. Ask it straight, in one message.
  3. Precision: "our service got slower and slower overnight, no deploy". A different problem
     with a similar symptom (a slow heap leak / GC). Ping should NOT offer the deploy / pool
     material for it.
  4. Ask who to talk to: "who else has hit this, and is anyone automating the post-deploy checks?"
"""

# ───────────────────────── SPOILER: answer key ─────────────────────────

ANSWER_KEY = {
    "latency spikes ~10 min after a deploy": {
        "artifact": ["Service-Deploy-Health-Runbook-v2.pdf (KI-7: base chart leaves pool maxLifetime at the 30-min default, "
                     "longer than the PgBouncer 600 s idle timeout)",
                     "Add a pool max-lifetime default to the base chart (Platform chat with the hikari values block, PLAT-412)",
                     "rollout.sh (renders base + overlay; base leaves maxLifetime unset)"],
        "insight": ["p99 climbs a few minutes after every release (Backend, user-20: same symptom; fixed with "
                    "maxLifetime: 570000 + keepaliveTime in the service overlay)"],
        "idea": ["Deploy Latency Validator agent (SRE, user-21) and deploy_latency_validator.py"],
        "related, weaker": ["Platform-Base-Deploy-Values.pdf (pool block with maxLifetime unset; PgBouncer server_idle_timeout 600 s)"],
    },
    "service slow overnight, no deploy": {
        "artifact": ["Service slow overnight, no deploy (Backend chat: unbounded cache -> heap leak -> long GC pauses)"],
        "should NOT suggest": ["the deploy runbook, base values, rollout script, validator, or the p99-after-release chat"],
    },
    "who else / automation": {
        "insight": ["user-20 (Backend) hit it; user-19 (Platform) owns the runbook and the base-chart fix"],
        "idea": ["Deploy Latency Validator (user-21, SRE)"],
    },
}
