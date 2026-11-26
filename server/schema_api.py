"""
/api/schema: the semantic model for Prospector's Semantic model panel (see schema_model.py).

  GET  /api/schema                     {schema, yaml, version, history, errors}
  PUT  /api/schema                     {schema | yaml, base_version}: validate and save as the next
                                       version (409 when someone saved since base_version)
  POST /api/schema/validate            {schema | yaml} -> {errors}
  GET  /api/schema/metrics?dataset=    one dataset measured against the schema, plus its history
                                       (a snapshot is recorded each time the dataset is re-ingested)
  GET  /api/schema/datasets            every export: size, conformance, whether its tags are stale
  GET  /api/schema/suggestions?dataset= bottom-up term proposals (grouped, checked) + term health
  POST /api/schema/suggestions/{approve,merge,reject}   apply a reviewed proposal (saves a version)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import schema_model as SM

router = APIRouter(prefix="/api/schema", tags=["schema"])
INCOMING = Path(__file__).resolve().parent / "datasets" / "incoming"


class SchemaBody(BaseModel):
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")   # "schema" shadows a BaseModel method
    yaml: Optional[str] = None
    base_version: Optional[int] = None

    model_config = {"populate_by_name": True}


def _body_schema(b: SchemaBody) -> dict:
    if b.yaml is not None:
        try:
            return SM.parse(b.yaml)
        except Exception as e:
            raise HTTPException(400, f"YAML: {e}")
    if b.schema_ is None:
        raise HTTPException(400, "send schema or yaml")
    return b.schema_


def _export_dir(dataset_id: str) -> Optional[Path]:
    base = (dataset_id or "").split("~")[0]
    p = INCOMING / base
    return p if base and p.is_dir() else None


@router.get("")
def get_schema():
    try:
        schema, text = SM.load()
    except Exception as e:
        raise HTTPException(500, f"schema.yaml: {e}")
    return {"schema": schema, "yaml": text, "version": schema.get("version"), "path": str(SM.SCHEMA_PATH),
            "history": SM.history(), "errors": SM.validate(schema)}


@router.post("/validate")
def validate(b: SchemaBody):
    return {"errors": SM.validate(_body_schema(b))}


@router.put("")
def put_schema(b: SchemaBody):
    schema = _body_schema(b)
    current, _ = SM.load()
    if b.base_version is not None and int(current.get("version") or 0) != b.base_version:
        raise HTTPException(409, f"the schema was saved by someone else (now v{current.get('version')}); reload first")
    try:
        v = SM.save(schema)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return get_schema() | {"saved": v}


@router.get("/metrics")
def metrics(dataset: str):
    ex = _export_dir(dataset)
    if ex is None:
        raise HTTPException(404, f"{dataset} is not an export under datasets/incoming (no metadata to measure)")
    schema, _ = SM.load()
    m = SM.measure(schema, ex)
    return {"dataset": ex.name, "schema_version": schema.get("version"), **m, "history": SM.track(ex, m)}


class ReviewBody(BaseModel):
    taxonomy: str
    term: Optional[str] = None            # approve: the new term's name
    names: List[str] = []                 # merge / reject: suggested names (approve: kept as aliases)
    description: str = ""
    parent: Optional[str] = None          # approve: group term to put it under (topics)
    into: Optional[str] = None            # merge: the existing term that absorbs the names
    base_version: Optional[int] = None


def _exports() -> List[Path]:
    return [ex for ex in sorted(INCOMING.iterdir()) if (ex / "docs").is_dir()] if INCOMING.is_dir() else []


@router.get("/suggestions")
def suggestions(dataset: Optional[str] = None):
    """Bottom-up proposals from every export's tagging runs, checked against the current schema,
    plus term-health advisories and Hub tags for the selected dataset."""
    schema, _ = SM.load()
    ex = _export_dir(dataset) if dataset else None
    return {"schema_version": schema.get("version"), **SM.suggestions(schema, _exports(), ex)}


@router.post("/suggestions/{action}")
def review(action: str, b: ReviewBody):
    """approve: add a term · merge: add the names as aliases of `into` · reject: never suggest
    them again. Each saves the schema as the next version (409 if it changed since base_version)."""
    current, _ = SM.load()
    if b.base_version is not None and int(current.get("version") or 0) != b.base_version:
        raise HTTPException(409, f"the schema was saved by someone else (now v{current.get('version')}); reload first")
    try:
        if action == "approve":
            new = SM.approve_term(current, b.taxonomy, b.term or "", b.description, b.parent, b.names)
        elif action == "merge":
            new = SM.merge_terms(current, b.taxonomy, b.names, b.into or "")
        elif action == "reject":
            new = SM.reject_terms(current, b.taxonomy, b.names)
        else:
            raise HTTPException(404, f"unknown action {action}")
        v = SM.save(new)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return get_schema() | {"saved": v}


@router.get("/datasets")
def datasets_overview():
    schema, _ = SM.load()
    out = []
    for ex in sorted(INCOMING.iterdir()) if INCOMING.is_dir() else []:
        if not (ex / "docs").is_dir():
            continue
        m = SM.measure(schema, ex)
        SM.track(ex, m)
        out.append({"dataset": ex.name, "docs": m["docs"], "nodes": m["nodes"], "edges": m["edges"],
                    "has_graph": m["has_graph"], "violations": m["violations"],
                    "unknown_classes": len(m["unknown_classes"]), "unknown_relations": len(m["unknown_relations"]),
                    "graph_built_at": m["graph_built_at"],
                    "stale": [t for t, v in m["taxonomies"].items() if v["stale"]],
                    "coverage": {t: [v["assigned"], v["records"]] for t, v in m["taxonomies"].items()}})
    return out
