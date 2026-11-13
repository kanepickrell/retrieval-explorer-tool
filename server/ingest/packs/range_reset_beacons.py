"""
Planted-artifact pack: "agents go quiet after a range reset".

A cross-team problem whose answer is spread over several artifacts, with
distractors. Everything here is fictional test content (made-up hosts, IPs in
private ranges, a fictional opfor.lab domain), planted by ingest/plant.py into a
COPY of an export so the real data stays clean. Planted users are new
pseudonyms with teams, so team-aware suggestions can be tested too.

SPOILER: ANSWER_KEY at the bottom says which artifacts should be found for the
test problems. Don't read it before testing.
"""

NAME = "range-reset-beacons"

USERS = {"user-16": "Range", "user-17": "OPFOR", "user-18": "Automation"}

# ───────────────────────── files (PDFs and code) ─────────────────────────

RUNBOOK = """RANGE RESET RUNBOOK: Exercise Network (v3)
Owner: Range team. Last revised 2026-11-04.

1. Purpose
This runbook resets the exercise network to its clean baseline between exercise iterations: victim hosts, the perimeter firewall (FW-EX01, pfSense), exercise DNS and the scoring engine. Red-team infrastructure (C2 servers, redirectors) is NOT reset and keeps running.

2. Before you reset
- Announce the reset in the range-ops channel at least 30 minutes ahead.
- Confirm with the exercise lead that no operation is in flight.
- Snapshot anything you changed by hand on victim hosts; it will be lost.

3. Reset sequence
3.1 From the range jump box run: reset_range.sh --env exercise --confirm
3.2 The script reverts victim VMs to the clean-baseline snapshot, restores FW-EX01 from the golden configuration, reseeds unbound with the exercise zone and restarts the scoring engine.
3.3 Expected duration: 25 to 35 minutes. Do not interrupt the firewall restore step.

4. Post-reset checks
4.1 Scoring engine shows every service green.
4.2 Victim hosts are reachable from the jump box (ping sweep of 10.40.20.0/24).
4.3 Red-team C2 reachability from a victim host: curl http://10.40.0.10:8888/ must return HTTP 200 within 2 seconds.
4.4 Exercise DNS resolves the in-game domains (dig @10.40.0.53 portal.exercise.lab).

5. Known issues
KI-12 Exercise DNS overrides missing after reset.
  Symptom: in-game names do not resolve. Cause: unbound starts before the zone file is copied. Fix: unbound-control reload.

KI-14 Implants check in after a reset, then go quiet about ten minutes later.
  Symptom: agents and implants on victim hosts beacon normally right after the reset, then stop at roughly the 10-minute mark. Redeploying the agent does not help; new agents stop as well.
  Cause: the golden firewall configuration (range-fw-golden-2026-09.xml) predates the EXERCISE_C2 alias. After the restore, outbound TCP 8888 from the victim VLAN (10.40.20.0/24) to the red-team C2 server (10.40.0.10) is no longer permitted. Connections that were already open survive in the pfSense state table until the TCP established-state timeout, which we tune to 600 seconds, so everything looks healthy for about ten minutes.
  Fix: Firewall > Aliases > Import exercise_c2_alias.xml, then apply the rule "Allow EXERCISE_C2 egress" above the default deny. Or from the pfSense shell: pfSsh.php playback restore_c2_egress
  Verify: pfctl -sr | grep 8888 shows the pass rule; curl from a victim host to 10.40.0.10:8888 returns 200.
  Permanent fix is tracked in RANGE-231 (regenerate the golden configuration with the alias included).

6. Contacts
Range on-call, rotation pinned in the team channel.
"""

FW_BASELINE = """EXERCISE NETWORK FIREWALL BASELINE (FW-EX01)
Range team. Version 2026-09.

Scope
The baseline policy applied to the exercise perimeter firewall at build time and whenever the range is restored to its golden state.

Policy
- Default deny, inbound and outbound, on every exercise VLAN.
- Victim VLAN 10.40.20.0/24 may reach: exercise DNS 10.40.0.53 on UDP/TCP 53; the package mirror 10.40.0.80 on TCP 80 and 443; the scoring engine 10.40.0.20 on TCP 443.
- ICMP echo is allowed inside the exercise network for troubleshooting.
- No direct internet egress from any exercise VLAN.

Exceptions
Red-team and white-cell infrastructure is NOT part of this baseline. Exceptions are expressed as firewall aliases owned by the team that needs them (for example the C2 servers) and are added on top of the baseline by that team.

State handling
TCP established-state timeout is set to 600 seconds (default is far longer) so that stale exercise sessions are cleared quickly between iterations.

Change control
Changes to the baseline require a Range team ticket and a new golden configuration export.
"""

RESET_SCRIPT = """#!/usr/bin/env bash
# reset_range.sh: return the exercise network to its clean baseline.
# Range team. Usage: reset_range.sh --env exercise --confirm
set -euo pipefail

ENV="exercise"
GOLDEN_FW="range-fw-golden-2026-09.xml"
FW_HOST="fw-ex01.range.lab"
VICTIM_FOLDER="/Range/exercise/victims"

log() { echo "[$(date +%H:%M:%S)] $*"; }

revert_victims() {
  log "reverting victim VMs to clean-baseline"
  for vm in $(govc ls "$VICTIM_FOLDER"); do
    govc snapshot.revert -vm "$vm" clean-baseline
    govc vm.power -on "$vm"
  done
}

restore_firewall() {
  log "restoring $FW_HOST from $GOLDEN_FW"
  scp "configs/$GOLDEN_FW" "admin@$FW_HOST:/tmp/golden.xml"
  ssh "admin@$FW_HOST" "cp /tmp/golden.xml /conf/config.xml && rm /tmp/config.cache && /etc/rc.reload_all"
  # NOTE: aliases added after the golden export are not in $GOLDEN_FW
}

reseed_dns() {
  log "reseeding unbound with the exercise zone"
  scp zones/exercise.lab.conf "admin@dns-ex01.range.lab:/var/unbound/exercise.conf"
  ssh "admin@dns-ex01.range.lab" "unbound-control reload"
}

restart_scoring() {
  log "restarting scoring engine"
  ssh "svc@score01.range.lab" "systemctl restart scoring-engine"
}

[[ "${1:-}" == "--env" && "${2:-}" == "$ENV" && "${3:-}" == "--confirm" ]] || { echo "usage: $0 --env exercise --confirm"; exit 2; }
revert_victims
restore_firewall
reseed_dns
restart_scoring
log "reset complete; run the post-reset checks in the runbook"
"""

VALIDATOR = '''"""range_reset_validator.py: post-reset checks for the exercise range.

Automation team. Run after reset_range.sh and before the exercise resumes.
Flags the regressions the Range runbook lists as known issues, so nobody finds
them mid-exercise: missing firewall aliases / egress rules for red-team C2,
exercise DNS overrides, and agents that stop checking in.
"""
from __future__ import annotations

import subprocess
import sys
import time

import requests

FW_HOST = "fw-ex01.range.lab"
REQUIRED_ALIASES = {"EXERCISE_C2": ["10.40.0.10"]}
REQUIRED_EGRESS = [  # (source net, destination, port, proto)
    ("10.40.20.0/24", "10.40.0.10", 8888, "tcp"),   # Caldera contact (HTTP)
    ("10.40.20.0/24", "10.40.0.53", 53, "udp"),     # exercise DNS
]
CALDERA = "http://10.40.0.10:8888"


def pf(cmd: str) -> str:
    return subprocess.run(["ssh", f"admin@{FW_HOST}", cmd], capture_output=True, text=True, check=True).stdout


def missing_aliases() -> list[str]:
    tables = pf("pfctl -sT")
    return [name for name in REQUIRED_ALIASES if name not in tables.split()]


def missing_egress() -> list[tuple]:
    rules = pf("pfctl -sr")
    gaps = []
    for src, dst, port, proto in REQUIRED_EGRESS:
        if not any(f"port = {port}" in r and proto in r and "pass" in r for r in rules.splitlines()):
            gaps.append((src, dst, port, proto))
    return gaps


def stale_agents(api_key: str, window_s: int = 300) -> list[str]:
    """Caldera agents whose last beacon is older than window_s seconds."""
    agents = requests.get(f"{CALDERA}/api/v2/agents", headers={"KEY": api_key}, timeout=5).json()
    now = time.time()
    return [a["paw"] for a in agents if now - a.get("last_seen_epoch", 0) > window_s]


def main(api_key: str) -> int:
    problems = []
    if (gone := missing_aliases()):
        problems.append(f"missing firewall aliases: {gone} (golden config predates them?)")
    if (gaps := missing_egress()):
        problems.append(f"missing egress rules: {gaps}")
    if (stale := stale_agents(api_key)):
        problems.append(f"{len(stale)} agents stale > 5 min: {stale[:10]}")
    for p in problems:
        print("FAIL", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
'''

FILES = [
    {"key": "runbook", "user": "user-16", "filename": "Range-Reset-Runbook-v3.pdf", "content_type": "application/pdf",
     "when": "2026-11-04 15:10", "text": RUNBOOK, "pdf": True},
    {"key": "fw_baseline", "user": "user-16", "filename": "Exercise-Network-Firewall-Baseline.pdf",
     "content_type": "application/pdf", "when": "2026-09-22 10:05", "text": FW_BASELINE, "pdf": True},
    {"key": "reset_script", "user": "user-16", "filename": "reset_range.sh", "content_type": "text/x-shellscript",
     "when": "2026-09-30 09:40", "text": RESET_SCRIPT},
    {"key": "validator", "user": "user-18", "filename": "range_reset_validator.py", "content_type": "text/x-python",
     "when": "2026-11-07 13:25", "text": VALIDATOR},
]

# ───────────────────────── chats ─────────────────────────

CHATS = [
    {"key": "opfor_sandcat", "user": "user-17", "title": "Sandcat agents dropping after range rebuild",
     "when": "2026-11-06 09:12", "model": "gemma-4-31b-it",
     "tags": ["cybersecurity", "caldera", "red_teaming", "networking"],
     "summary": "An OPFOR user's Caldera sandcat agents stopped checking in about ten minutes after the range was rebuilt. "
                "They found new connections to the C2 port were being dropped while old ones lingered, worked around it with an "
                "HTTPS redirector on 443, and learned from the Range team that the golden pfSense config is missing their C2 alias.",
     "pain_points": ["intermittent connectivity", "workaround, not a fix"],
     "turns": [
         ("We're running the purple-team iteration on the range. Range rebuilt the environment this morning and our sandcat "
          "agents check in fine, then about ten minutes later every one of them goes stale in the Caldera UI. I redeployed "
          "on two hosts and the new agents do the same thing. The Caldera server itself is up and I can reach the UI. Ideas?",
          "A few things produce that pattern:\n\n1. **Agent watchdog / sleep settings**: if the group's watchdog is shorter "
          "than the beacon interval, agents kill themselves. Check the sandcat `-w` value.\n2. **Untrusted timer**: Caldera marks "
          "agents untrusted if they miss beacons for longer than the untrusted timer.\n3. **Something on the path**: a proxy or "
          "firewall with an idle timeout that drops the contact.\n\nSince it is every agent at the same time and new ones fail "
          "too, I'd look at the network path first. From a victim host, can you open a *new* connection to the contact port?"),
         ("watchdog is default and so is the untrusted timer. From a victim box, curl http://10.40.0.10:8888 just hangs now, "
          "but sessions that were open from before the 10 min mark were working. So something is dropping new connections to 8888.",
          "That is the classic signature of a **stateful firewall rule change**: flows that were established before the change "
          "stay in the state table until their timeout, while new flows hit the new policy and are dropped. On pfSense:\n\n"
          "- `pfctl -sr | grep 8888` to see whether a pass rule for the contact port still exists\n"
          "- `pfctl -ss | grep 8888` to see the old states draining\n- Firewall > Aliases, check the alias your rule refers to\n\n"
          "If the range restore replaced the firewall config, your egress rule may simply not be in it any more."),
         ("Got around it: put sandcat behind our HTTPS redirector on 443 (`-server https://redir01.opfor.lab -group red`) and "
          "everything checks in again. Range says their golden pfSense config is old and is missing our C2 alias; they're "
          "supposed to fix it. Annoying that this will bite the next team too.",
          "Good workaround: 443 is allowed by the baseline, so the redirector path survives the restore. To stop this biting "
          "the next team:\n\n- ask Range to add the C2 alias and egress rule to the golden config (or to a post-restore step)\n"
          "- add a post-reset check that opens a new connection to the C2 contact port from a victim host\n\n"
          "Until then, anyone using the plain HTTP contact on 8888 will see agents go stale about ten minutes after each reset."),
     ]},
    {"key": "range_golden", "user": "user-16", "title": "Golden firewall config refresh",
     "when": "2026-11-08 16:30", "model": "gemma-4-31b-it", "attach": {0: ["reset_script"]},
     "tags": ["networking", "pfsense", "automation"],
     "summary": "A Range user asked for a pfSense playback script that re-adds the EXERCISE_C2 alias and its egress rule after "
                "the golden-config restore, as a stopgap until the golden configuration is regenerated (RANGE-231).",
     "pain_points": [],
     "turns": [
         ("Our reset script (attached) restores FW-EX01 from an old golden config that doesn't have the EXERCISE_C2 alias. "
          "Until RANGE-231 lands, I want a pfSsh.php playback script called restore_c2_egress that re-adds the alias "
          "(10.40.0.10) and a pass rule for tcp 8888 from 10.40.20.0/24 above the default deny. Can you write it?",
          "Here is a playback script; save it as `/etc/phpshellsessions/restore_c2_egress` on the firewall:\n\n```php\n"
          "require_once('config.inc');\nrequire_once('filter.inc');\n\n$alias = ['name' => 'EXERCISE_C2', 'type' => 'host',\n"
          "          'address' => '10.40.0.10', 'descr' => 'Red-team C2 (re-added after golden restore)'];\n"
          "config_set_path('aliases/alias/', array_merge(config_get_path('aliases/alias', []), [$alias]));\n\n"
          "$rule = ['type' => 'pass', 'interface' => 'opt2', 'ipprotocol' => 'inet', 'protocol' => 'tcp',\n"
          "         'source' => ['network' => '10.40.20.0/24'], 'destination' => ['address' => 'EXERCISE_C2', 'port' => '8888'],\n"
          "         'descr' => 'Allow EXERCISE_C2 egress'];\n$rules = config_get_path('filter/rule', []);\narray_unshift($rules, $rule);\n"
          "config_set_path('filter/rule', $rules);\nwrite_config('restore_c2_egress');\nfilter_configure();\n```\n\n"
          "Run it with `pfSsh.php playback restore_c2_egress` right after `restore_firewall` in reset_range.sh."),
         ("Thanks. I'll call it from reset_range.sh and add KI-14 to the runbook so people know why their implants die at the 10 minute mark.",
          "Sounds right. Put the verification next to it in the runbook (`pfctl -sr | grep 8888` and a curl from a victim host), "
          "and once RANGE-231 regenerates the golden config the playback step can be removed."),
     ]},
    {"key": "opfor_sliver", "user": "user-17", "title": "Sliver implants not calling back",
     "when": "2026-10-28 14:02", "model": "gemma-4-31b-it",
     "tags": ["cybersecurity", "sliver", "red_teaming"],
     "summary": "An OPFOR user's Sliver implants stopped calling back over HTTPS; the listener's certificate had expired. "
                "Regenerating the certificate and restarting the listener fixed it.",
     "pain_points": ["expired certificate"],
     "turns": [
         ("Our Sliver implants stopped calling back sometime overnight. Nothing changed on our side as far as I know. The "
          "HTTPS listener on the team server is running.",
          "When HTTPS implants stop all at once with no change on your side, check the listener's TLS certificate first: "
          "an expired or replaced certificate makes the implant's TLS handshake fail. `openssl s_client -connect <server>:443` "
          "shows the expiry. Also check the team server's clock."),
         ("Yep, the cert expired yesterday. Regenerated it with the https listener's --cert/--key and restarted the job, implants are back.",
          "Good. Consider letsencrypt or a longer-lived internal CA certificate for exercise listeners, and put the expiry in "
          "the exercise checklist."),
     ]},
]

# ───────────────────────── Agent Hub ─────────────────────────

AGENTS = [
    {"key": "validator_agent", "owner": "user-18", "name": "Range Reset Validator", "slug": "range-reset-validator",
     "when": "2026-11-08 11:00", "manual_time_minutes": 45,
     "description": "Runs the post-reset checks on the exercise range and flags firewall, DNS and C2 reachability "
                    "regressions before the exercise resumes.",
     "tags": ["cyber-range", "validation", "pfsense", "caldera"],
     "system": "You validate the exercise range after a reset. Ask for (or run) range_reset_validator.py and read its output. "
               "For every FAIL line, explain the likely cause using the Range runbook's known issues (KI-12 DNS overrides, "
               "KI-14 C2 egress missing after the golden firewall restore) and give the exact fix and how to verify it. "
               "Never declare the range ready while any check fails.",
     "skills": [{"name": "pfsense-rule-audit",
                 "description": "Read pfctl rule and state output to explain why new connections fail while old ones work",
                 "content": "---\nname: pfsense-rule-audit\n---\nWhen new connections to a port fail but established ones keep "
                            "working, the policy changed under live states. `pfctl -sr` lists rules (look for a pass rule for "
                            "the port above the default block), `pfctl -sT` lists alias tables, `pfctl -ss` shows states "
                            "draining. States expire at the TCP established timeout, which explains a delay before failures."}]},
]

INSTALLS = [("user-16", "validator_agent"), ("user-17", "validator_agent")]

# ───────────────────────── test problems (safe to read) ─────────────────────────

PROBLEMS = """\
You're running tomorrow's exercise rehearsal. The range was reset this morning. The Caldera
agents on the victim hosts show up fine, but roughly ten minutes later they all go stale.
Redeploying them didn't help.

Ways to test (type them on the Agent page; each starts or continues a session):
  1. Start vague, then add detail over a few messages, the way someone would in a real chat:
       "our caldera agents keep disappearing"
       -> "it happens after the range gets reset"
       -> "they check in, then after about ten minutes they're gone, redeploying doesn't help"
     Watch when the gate opens and what kind of thing Ping suggests (artifact / insight / idea).
  2. Ask it straight, in one message.
  3. Precision: "our sliver implants stopped calling back overnight". A different problem with a
     similar symptom. Ping should not offer the range-reset material for it.
  4. Ask who to talk to: "who else has run into this, and is anyone automating the checks?"
"""

# ───────────────────────── SPOILER: answer key ─────────────────────────

ANSWER_KEY = {
    "range reset, agents stale after ~10 min": {
        "artifact": ["Range-Reset-Runbook-v3.pdf (KI-14: golden firewall config lacks EXERCISE_C2; state timeout 600 s)",
                     "Golden firewall config refresh (chat with the restore_c2_egress playback script)",
                     "reset_range.sh (restores range-fw-golden-2026-09.xml)"],
        "insight": ["Sandcat agents dropping after range rebuild (OPFOR, user-17: same symptom; HTTPS redirector on 443 workaround)"],
        "idea": ["Range Reset Validator agent (Automation, user-18) and range_reset_validator.py"],
        "related, weaker": ["Exercise-Network-Firewall-Baseline.pdf (default deny; C2 exceptions are aliases; 600 s timeout)"],
    },
    "sliver implants not calling back": {
        "artifact": ["Sliver implants not calling back (OPFOR chat: expired listener certificate)"],
        "should NOT suggest": ["the range-reset runbook, script, validator or sandcat chat"],
    },
    "who else / automation": {
        "insight": ["user-17 (OPFOR) hit it; user-16 (Range) owns the runbook and the golden config"],
        "idea": ["Range Reset Validator (user-18, Automation)"],
    },
}
