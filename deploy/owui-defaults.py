"""
Apply the Open WebUI settings the ProtoGraph chime-in is tuned for, to a running stack.

Run once after the stack is up (and again any time; it's idempotent):

    docker cp ../retrieval-explorer/deploy/owui-defaults.py open-webui:/tmp/owui-defaults.py
    docker exec open-webui python /tmp/owui-defaults.py

(protograph-up.sh / protograph-up.ps1 in the stack repo do this for you.)

What it sets, and why:
  * Follow-up generation off: Ping writes the follow-ups under each reply and learns from
    the one the user clicks, so Open WebUI's own would compete with it (and with the GPU).
  * For each tool-capable chat model below:
      - "Ask User" built-in tool off: the model was interrupting with clarification forms
        instead of answering; Ping's follow-ups do the narrowing.
      - think = "low" for gpt-oss (it can't turn thinking off; "low" is ~24 chars of reasoning).
Settings live in Open WebUI's database (the open-webui-data volume), not in env, so a fresh
volume needs this once. Changing them later in Admin > Settings is fine.
"""
import json
import sqlite3
import time

DB = "/app/backend/data/webui.db"
MODELS = {
    "gpt-oss:20b": {"think": "low"},
    "nemotron-3.5-lightning:latest": {"think": False},
    "lfm2:24b": {},
}
CAPABILITIES = {"file_context": True, "vision": True, "file_upload": True, "web_search": True,
                "image_generation": True, "code_interpreter": True, "terminal": True, "citations": True,
                "status_updates": True, "memory": True, "builtin_tools": True}

c = sqlite3.connect(DB)
changed = False

# follow-up generation (a persistent config key; insert it if this volume never stored one)
cur = c.execute("select value from config where key='task.follow_up.enable'").fetchone()
if cur is None:
    c.execute("insert into config (key, value, updated_at) values ('task.follow_up.enable', 'false', ?)", (int(time.time()),))
    changed = True
elif str(cur[0]).strip('"').lower() != "false":
    c.execute("update config set value='false' where key='task.follow_up.enable'")
    changed = True
print("follow-up generation: off")

admin = c.execute("select id from user where role='admin' order by created_at limit 1").fetchone()
if not admin:
    raise SystemExit("no admin user yet: sign in to Open WebUI once (creates the admin), then rerun")
now = int(time.time())
for mid, params_set in MODELS.items():
    row = c.execute("select params, meta from model where id=?", (mid,)).fetchone()
    if not row and mid != "gpt-oss:20b":
        continue          # only touch other models if this install already has them
    if row:
        params = json.loads(row[0] or "{}")
        meta = json.loads(row[1] or "{}")
    else:
        params, meta = {}, {"profile_image_url": "/static/favicon.png", "description": None,
                            "capabilities": dict(CAPABILITIES), "tags": []}
    before = (json.dumps(params, sort_keys=True), json.dumps(meta, sort_keys=True))
    params.update(params_set)
    meta.setdefault("builtinTools", {})["user_input"] = False
    if row and before == (json.dumps(params, sort_keys=True), json.dumps(meta, sort_keys=True)):
        print(f"{mid}: already set")
        continue
    changed = True
    if row:
        c.execute("update model set params=?, meta=?, updated_at=? where id=?", (json.dumps(params), json.dumps(meta), now, mid))
    else:
        c.execute("insert into model (id, user_id, base_model_id, name, params, meta, updated_at, created_at, is_active) "
                  "values (?,?,?,?,?,?,?,?,1)", (mid, admin[0], None, mid, json.dumps(params), json.dumps(meta), now, now))
    print(f"{mid}: {'updated' if row else 'created'} (params {params}, Ask User off)")
c.commit()
if changed:
    print("changed: restart open-webui to load them (docker restart open-webui)")
    raise SystemExit(10)       # deploy scripts restart Open WebUI only on this code
print("nothing to change")
