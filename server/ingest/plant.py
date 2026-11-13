"""
Plant a pack of test artifacts (PDFs, code, chats, Agent Hub agents) into a COPY
of an export, to test whether Ping / the agent can find them.

  1. copy <export> to <export>-planted (refuses to overwrite anything that isn't a
     planted copy: it must hold PLANTED.json)
  2. append the pack's records to the copy's tables in Open WebUI's own shape:
     openwebui/file.json (uploads with their extracted text), chat.json +
     chat_message.json (parent-linked turns) + chat_review.json + chat_file.json,
     hub/agent.json + agent_version.json (manifest with skills) + hub_install.json
  3. render the pack's PDFs (headless Chrome) into <copy>/uploads/, for realism;
     the index uses the extracted text, as Open WebUI would
  4. run export_to_docs on the copy and keep only the NEW docs (+ changed
     catalogs), so the existing docs stay byte-identical to what was indexed
  5. fast-index those docs into the copy's storage/: chunked like LightRAG
     (gpt-4o-mini tokens, 500 with 50 overlap), embedded with nomic, appended to
     vdb_chunks / text_chunks / full_docs / doc_status. Vector, BM25, the agent
     and Ping see them; LightRAG's own graph modes don't (no entity extraction)
  6. teams.json for the pack's users, then build_metadata_graph + build_structural_gold

Records are fictional and marked: ids are uuid5("plant:<pack>:<key>"),
PLANTED.json lists them, and every planted doc's status says fast_index.

Usage:
  python plant.py <export_dir> <pack>        e.g. plant.py datasets/incoming/agenthub-export-2026-11-10 range_reset_beacons
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_to_docs as E  # noqa: E402

EMBED_HOST = os.getenv("PROTOGRAPH_EMBED_HOST", "http://10.128.241.200:11434")
EMBED_MODEL = os.getenv("PROTOGRAPH_EMBED_MODEL", "nomic-embed-text")
CHROME = os.getenv("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
CHUNK_TOKENS, OVERLAP = 500, 50          # LightRAG defaults, as the export was built
COPY_DIRS = ("hub", "openwebui", "requests", "docs", "storage")
COPY_FILES = ("manifest.json", "eval_set.json", "eval_set_generated.json", "ping.json", "teams.json")


def uid(pack: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"plant:{pack}:{key}"))


def epoch(when: str) -> int:
    return int(datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ───────────────────────── 1. copy ─────────────────────────

def copy_export(src: Path, dst: Path) -> None:
    if dst.exists():
        if not (dst / "PLANTED.json").exists():
            raise SystemExit(f"{dst} exists and is not a planted copy; refusing to overwrite it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for d in COPY_DIRS:
        if (src / d).is_dir():
            shutil.copytree(src / d, dst / d)
    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


# ───────────────────────── 2. records ─────────────────────────

def plant_records(dst: Path, P) -> dict:
    L = lambda rel: json.loads((dst / rel).read_text(encoding="utf-8"))
    files, chats, msgs = L("openwebui/file.json"), L("openwebui/chat.json"), L("openwebui/chat_message.json")
    reviews, chat_files = L("openwebui/chat_review.json"), L("openwebui/chat_file.json")
    agents, versions, installs = L("hub/agent.json"), L("hub/agent_version.json"), L("hub/hub_install.json")
    ids = {"files": {}, "chats": {}, "agents": {}}

    for f in P.FILES:
        fid, t = uid(P.NAME, f["key"]), epoch(f["when"])
        ids["files"][f["key"]] = fid
        files.append({"id": fid, "user_id": f["user"], "filename": f["filename"],
                      "meta": {"name": f["filename"], "content_type": f["content_type"],
                               "size": len(f["text"].encode()), "file_hash": sha(f["text"])},
                      "created_at": str(t), "updated_at": str(t + 15), "hash": sha(f["text"]),
                      "data": {"status": "completed", "content": f["text"]},
                      "path": f"/app/backend/data/uploads/{fid}_{f['filename']}"})

    for c in P.CHATS:
        cid, t0 = uid(P.NAME, c["key"]), epoch(c["when"])
        ids["chats"][c["key"]] = cid
        prev, t = None, t0
        for i, (u, a) in enumerate(c["turns"]):
            for role, text in (("user", u), ("assistant", a)):
                mid = uid(P.NAME, f"{c['key']}:{role}{i}")
                attached = [{"type": "file", "file": {"id": ids["files"][k], "filename": next(x["filename"] for x in P.FILES if x["key"] == k)}}
                            for k in (c.get("attach") or {}).get(i, [])] if role == "user" else None
                msgs.append({"id": f"{cid}-{mid}", "chat_id": cid, "user_id": c["user"], "role": role,
                             "parent_id": prev, "content": json.dumps(text), "output": None,
                             "model_id": c["model"] if role == "assistant" else None, "files": attached,
                             "sources": None, "embeds": None, "done": "1", "status_history": None, "error": None,
                             "usage": None, "created_at": str(t), "updated_at": str(t), "context_summary": None, "meta": None})
                if attached:
                    for k in (c.get("attach") or {}).get(i, []):
                        chat_files.append({"id": uid(P.NAME, f"{c['key']}:attach:{k}"), "user_id": c["user"], "chat_id": cid,
                                           "file_id": ids["files"][k], "message_id": mid,
                                           "created_at": str(t), "updated_at": str(t)})
                prev = mid
                t += 45 if role == "user" else 240
        chats.append({"id": cid, "user_id": c["user"], "title": c["title"], "created_at": str(t0), "updated_at": str(t),
                      "share_id": None, "archived": "0", "chat": {"id": cid, "title": c["title"], "models": [c["model"]]},
                      "pinned": "0", "meta": {"tags": c["tags"]}, "folder_id": None, "tasks": None, "summary": c["summary"],
                      "last_read_at": str(t), "current_message_id": prev, "variables": {}, "timer_at": None})
        reviews.append({"chat_id": cid, "user_id": c["user"], "model_id": c["model"], "signals": [], "score": "1",
                        "user_turns": str(len(c["turns"])), "heuristics_at": str(t), "summary": c["summary"],
                        "summary_at": str(t), "pain_points": c.get("pain_points", []), "chat_updated_at": str(t)})

    for a in P.AGENTS:
        aid, vid, t = uid(P.NAME, a["key"]), uid(P.NAME, a["key"] + ":v1"), epoch(a["when"])
        ids["agents"][a["key"]] = aid
        n_inst = sum(1 for _, k in P.INSTALLS if k == a["key"])
        agents.append({"id": aid, "slug": a["slug"], "name": a["name"], "description": a["description"], "tags": a["tags"],
                       "manual_time_minutes": str(a["manual_time_minutes"]), "owner_id": a["owner"], "forked_from_id": None,
                       "forked_from_slug": None, "latest_version": "1", "install_count": str(n_inst), "is_public": "True",
                       "created_at": str(t), "updated_at": str(t), "rating_count": "0", "rating_sum": "0", "deleted_at": None})
        manifest = {"manifest_version": 1, "name": a["name"], "slug": a["slug"], "description": a["description"],
                    "tags": a["tags"], "base_model_id": "gemma-4-31b-it", "manual_time_minutes": a["manual_time_minutes"],
                    "system": a["system"], "params": {}, "capabilities": {}, "profile_image_url": "",
                    "skills": a.get("skills", []), "tools": []}
        versions.append({"id": vid, "agent_id": aid, "version": "1", "manifest": manifest, "changelog": "Initial version",
                         "created_by": a["owner"], "created_at": str(t), "parent_version_id": None,
                         "content_hash": sha(json.dumps(manifest, sort_keys=True))})
    for user, key in P.INSTALLS:
        a = next(x for x in P.AGENTS if x["key"] == key)
        t = epoch(a["when"]) + 3600
        installs.append({"id": uid(P.NAME, f"install:{user}:{key}"), "kind": "agent", "item_id": ids["agents"][key],
                         "item_slug": a["slug"], "version_id": uid(P.NAME, key + ":v1"), "version": 1, "user_id": user,
                         "base_model_id": "gemma-4-31b-it", "model_config": {}, "created_at": str(t)})

    for rel, rows in (("openwebui/file.json", files), ("openwebui/chat.json", chats), ("openwebui/chat_message.json", msgs),
                      ("openwebui/chat_review.json", reviews), ("openwebui/chat_file.json", chat_files),
                      ("hub/agent.json", agents), ("hub/agent_version.json", versions), ("hub/hub_install.json", installs)):
        dump(dst / rel, rows)
    man = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    man["people"] = sorted(set(man.get("people", [])) | set(P.USERS))
    man["planted"] = {"pack": P.NAME, "note": "fictional test artifacts added by ingest/plant.py"}
    dump(dst / "manifest.json", man)
    teams = json.loads((dst / "teams.json").read_text(encoding="utf-8")) if (dst / "teams.json").exists() else {}
    dump(dst / "teams.json", {**teams, **P.USERS})
    return ids


# ───────────────────────── 3. PDFs ─────────────────────────

def render_pdfs(dst: Path, P, ids: dict) -> list:
    if not Path(CHROME).exists():
        print(f"[plant] no Chrome at {CHROME}; skipping PDF rendering (the index uses the text either way)")
        return []
    out_dir = dst / "uploads"
    out_dir.mkdir(exist_ok=True)
    made = []
    for f in P.FILES:
        if not f.get("pdf"):
            continue
        lines = f["text"].strip().split("\n")
        body = "".join(f"<h1>{html.escape(l)}</h1>" if i == 0 else f"<p>{html.escape(l) or '&nbsp;'}</p>"
                       for i, l in enumerate(lines))
        page = (f"<!doctype html><meta charset='utf-8'><style>body{{font:11pt Georgia,serif;margin:2cm;color:#111}}"
                f"h1{{font:bold 15pt Arial,sans-serif}}p{{margin:.25em 0;white-space:pre-wrap}}</style>{body}")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "page.html"
            src.write_text(page, encoding="utf-8")
            pdf = out_dir / f"{ids['files'][f['key']]}_{f['filename']}"
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={td}/profile",
                            "--no-pdf-header-footer", f"--print-to-pdf={pdf}", src.as_uri()],
                           capture_output=True, timeout=120)
            if pdf.exists():
                made.append(pdf.name)
    return made


# ───────────────────────── 4. docs ─────────────────────────

def new_docs(dst: Path) -> list:
    docs = dst / "docs"
    existing = {p.name: p.read_text(encoding="utf-8") for p in docs.glob("*.md")}
    with tempfile.TemporaryDirectory() as td:
        E.main(str(dst), td)
        changed = []
        for p in sorted(Path(td).glob("*.md")):
            text = p.read_text(encoding="utf-8")
            if p.name not in existing or (p.name.startswith("catalog-") and existing[p.name] != text):
                (docs / p.name).write_text(text, encoding="utf-8")
                changed.append(p.name)
    return changed


# ───────────────────────── 5. fast index ─────────────────────────

def _embed(texts: list) -> np.ndarray:
    out = []
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(texts), 16):
            r = c.post(f"{EMBED_HOST}/api/embed",
                       json={"model": EMBED_MODEL, "input": ["search_document: " + t for t in texts[i:i + 16]]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    m = np.array(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def fast_index(storage: Path, docs_dir: Path, names: list, track: str) -> int:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    R = lambda f: json.loads((storage / f).read_text(encoding="utf-8"))
    vdb, chunks, full, status = R("vdb_chunks.json"), R("kv_store_text_chunks.json"), R("kv_store_full_docs.json"), R("kv_store_doc_status.json")
    dim = int(vdb["embedding_dim"])
    mat = np.frombuffer(base64.b64decode(vdb["matrix"]), dtype=np.float32).reshape(len(vdb["data"]), dim)
    doc_ids = {n[:-3] for n in names}
    keep = [i for i, r in enumerate(vdb["data"]) if r.get("full_doc_id") not in doc_ids]   # re-indexed docs drop their old chunks
    rows, mat = [vdb["data"][i] for i in keep], mat[keep]
    chunks = {k: v for k, v in chunks.items() if v.get("full_doc_id") not in doc_ids}
    now, iso = int(time.time()), datetime.now(timezone.utc).isoformat()
    new_rows, texts = [], []
    for name in names:
        doc_id, text = name[:-3], (docs_dir / name).read_text(encoding="utf-8")
        toks = enc.encode(text)
        pieces = [toks[s:s + CHUNK_TOKENS] for s in range(0, max(1, len(toks)), CHUNK_TOKENS - OVERLAP)]
        pieces = [p for k, p in enumerate(pieces) if p and (k == 0 or len(p) > OVERLAP)]
        ids = []
        for k, p in enumerate(pieces):
            cid, content = f"{doc_id}-chunk-{k:03d}", enc.decode(p).strip()
            ids.append(cid)
            chunks[cid] = {"tokens": len(p), "content": content, "chunk_order_index": k, "full_doc_id": doc_id,
                           "file_path": name, "llm_cache_list": [], "create_time": now, "update_time": now, "_id": cid}
            new_rows.append({"__id__": cid, "__created_at__": now, "__write_seq__": time.time_ns(), "content": content,
                             "full_doc_id": doc_id, "file_path": name})
            texts.append(content)
        h = hashlib.md5(text.encode()).hexdigest()
        full[doc_id] = {"content": text, "file_path": name, "parse_format": "raw", "content_hash": h,
                        "create_time": now, "update_time": now, "_id": doc_id}
        status[doc_id] = {"status": "processed", "content_summary": text[:100], "content_length": len(text),
                          "created_at": iso, "updated_at": iso, "file_path": name, "track_id": track, "content_hash": h,
                          "metadata": {"fast_index": True, "note": "chunk index only (vector + BM25), no entity extraction"},
                          "chunks_count": len(ids), "chunks_list": ids}
    vecs = _embed(texts) if texts else np.zeros((0, dim), np.float32)
    for r, v in zip(new_rows, vecs):   # LightRAG's per-row form: float16 -> zlib -> base64
        r["vector"] = base64.b64encode(zlib.compress(v.astype(np.float16).tobytes())).decode()
    rows += new_rows
    mat = np.vstack([mat, vecs]) if len(vecs) else mat
    dump(storage / "vdb_chunks.json", {"embedding_dim": dim, "data": rows, "matrix": base64.b64encode(mat.astype(np.float32).tobytes()).decode()})
    dump(storage / "kv_store_text_chunks.json", chunks)
    dump(storage / "kv_store_full_docs.json", full)
    dump(storage / "kv_store_doc_status.json", status)
    return len(new_rows)


# ───────────────────────── main ─────────────────────────

def main(src: Path, pack: str) -> None:
    P = importlib.import_module(f"packs.{pack}")
    dst = src.parent / f"{src.name}-planted"
    t0 = time.time()
    copy_export(src, dst)
    ids = plant_records(dst, P)
    pdfs = render_pdfs(dst, P, ids)
    names = new_docs(dst)
    n = fast_index(dst / "storage", dst / "docs", names, f"plant_{P.NAME}")
    dump(dst / "PLANTED.json", {"pack": P.NAME, "source": src.name, "at": datetime.now(timezone.utc).isoformat(),
                                "users": P.USERS, "ids": ids, "docs": names, "pdfs": pdfs})
    (dst / "PROBLEMS.txt").write_text(P.PROBLEMS, encoding="utf-8")
    (dst / "ANSWER_KEY.json").write_text(json.dumps(P.ANSWER_KEY, indent=2), encoding="utf-8")
    import build_metadata_graph
    build_metadata_graph.build(dst, dst / "storage-metadata")
    import build_structural_gold
    build_structural_gold.main(str(dst))
    print(f"[plant] {P.NAME} -> {dst.name}: {len(names)} docs ({n} chunks) indexed, {len(pdfs)} PDFs, "
          f"{time.time() - t0:.0f}s")
    for d in names:
        print("   +", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("pack")
    a = ap.parse_args()
    main(Path(a.export_dir).resolve(), a.pack)
