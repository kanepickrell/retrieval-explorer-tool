import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { AlertTriangle, Check, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import {
  getSchema, getSchemaMetrics, getSchemaSuggestions, listSchemaDatasets, saveSchema, validateSchema,
  type ClassPropertyMetrics, type SchemaDatasetRow, type SchemaDoc, type SchemaMetrics, type SchemaProperty, type SchemaPropertyType,
  type SchemaTaxonomy, type SchemaTerm, type SemanticSchema,
} from "@/lib/api";
import SchemaMap, { type MapSelection } from "./SchemaMap";
import Suggestions from "./Suggestions";

/**
 * Prospector's Semantic model panel: the shared schema (server/datasets/schema/schema.yaml)
 * as a map and as editable tables, measured against the selected dataset. Metrics are
 * recomputed from the dataset's metadata graph and tags every time the panel loads, and the
 * backend records a snapshot whenever a dataset was rebuilt or re-tagged, so the History tab
 * shows what each ingest added.
 */

const TABS = ["map", "classes", "relations", "taxonomies", "suggestions", "glossary", "rules", "yaml", "history"] as const;
type Tab = (typeof TABS)[number];
const STATUSES = ["implemented", "proposed"] as const;
// taxonomies tag_docs.py reads by name: renaming would silently stop the classifier using them
const CORE_TAXONOMIES = ["topics", "doc_types"];

const fmtTime = (t: number | null | undefined) => (t ? new Date(t * 1000).toLocaleString() : "—");
const csv = (v: string) => v.split(",").map((s) => s.trim()).filter(Boolean);

/** Rebuild an object with one key renamed, keeping key order. */
function renameKey<T>(obj: Record<string, T>, from: string, to: string): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(obj)) out[k === from ? to : k] = v;
  return out;
}

/** Rename a class everywhere it is referenced. */
function renameClass(s: SemanticSchema, from: string, to: string) {
  s.classes = renameKey(s.classes, from, to);
  for (const c of Object.values(s.classes)) if (c.parent === from) c.parent = to;
  const sw = (xs?: string[]) => xs?.map((x) => (x === from ? to : x));
  for (const r of Object.values(s.relations)) { r.from = sw(r.from) ?? []; r.to = sw(r.to) ?? []; }
  for (const t of Object.values(s.taxonomies)) { t.applies_to = sw(t.applies_to); if (t.class === from) t.class = to; }
  for (const r of Object.values(s.rules)) { if (r.from === from) r.from = to; if (r.to === from) r.to = to; }
}

function renameRelation(s: SemanticSchema, from: string, to: string) {
  s.relations = renameKey(s.relations, from, to);
  for (const r of Object.values(s.rules)) r.via = r.via.map((p) => p.map((x) => (x === from ? to : x)));
}

/** A key (name) field: commits on blur or Enter, so renames don't fire per keystroke. */
function NameInput({ value, onCommit, className }: { value: string; onCommit: (v: string) => void; className?: string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  const commit = () => { const t = v.trim(); if (t && t !== value) onCommit(t); else setV(value); };
  return (
    <input value={v} onChange={(e) => setV(e.target.value)} onBlur={commit}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setV(value); }}
      className={clsx("w-full rounded border border-transparent bg-transparent px-1.5 py-0.5 font-mono text-xs text-ink outline-none hover:border-line focus:border-local/70", className)} />
  );
}

/** A plain text field bound straight to the draft. */
function Field({ value, onChange, placeholder, mono }: { value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean }) {
  return (
    <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      className={clsx("w-full rounded border border-transparent bg-transparent px-1.5 py-0.5 text-xs text-ink outline-none placeholder:text-muted/50 hover:border-line focus:border-local/70", mono && "font-mono")} />
  );
}

/** Comma-separated list field: edits as text, stored as a list on blur. */
function ListInput({ value, onCommit, placeholder }: { value: string[]; onCommit: (v: string[]) => void; placeholder?: string }) {
  const joined = (value ?? []).join(", ");
  const [v, setV] = useState(joined);
  useEffect(() => setV(joined), [joined]);
  return (
    <input value={v} onChange={(e) => setV(e.target.value)} onBlur={() => onCommit(csv(v))} placeholder={placeholder}
      onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
      className="w-full rounded border border-transparent bg-transparent px-1.5 py-0.5 font-mono text-xs text-ink outline-none placeholder:text-muted/50 hover:border-line focus:border-local/70" />
  );
}

function Bar({ value, max, tone = "local" }: { value: number; max: number; tone?: "local" | "warn" | "bad" }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="h-1.5 w-full rounded-full bg-surface-2">
      <div className={clsx("h-1.5 rounded-full", tone === "local" ? "bg-local/80" : tone === "warn" ? "bg-warn/80" : "bg-bad/80")}
        style={{ width: `${pct}%` }} />
    </div>
  );
}

function StatusSelect({ value, onChange }: { value?: string; onChange: (v: "implemented" | "proposed") => void }) {
  return (
    <select value={value ?? "implemented"} onChange={(e) => onChange(e.target.value as "implemented" | "proposed")}
      className={clsx("rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] outline-none", value === "proposed" ? "text-muted" : "text-ink")}>
      {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
    </select>
  );
}

const th = "px-2 py-1.5 font-normal";
const td = "px-1 py-1 align-top";

export default function SemanticModel({ datasetId }: { datasetId: string }) {
  const [doc, setDoc] = useState<SchemaDoc | null>(null);
  const [draft, setDraft] = useState<SemanticSchema | null>(null);
  const [metrics, setMetrics] = useState<SchemaMetrics | null>(null);
  const [metricsErr, setMetricsErr] = useState<string | null>(null);
  const [overview, setOverview] = useState<SchemaDatasetRow[]>([]);
  // ?model=<tab> opens a tab directly (shareable links to e.g. the taxonomies)
  const [tab, setTab] = useState<Tab>(() => {
    const t = new URLSearchParams(window.location.search).get("model");
    return (TABS as readonly string[]).includes(t ?? "") ? (t as Tab) : "map";
  });
  const [errors, setErrors] = useState<string[]>([]);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [sel, setSel] = useState<MapSelection>(null);
  const [yamlText, setYamlText] = useState("");
  const [ready, setReady] = useState(0);     // suggestions ready for review (tab badge)
  // classes tab: expanded property editors (?class=File opens one directly)
  const [openProps, setOpenProps] = useState<Set<string>>(() => {
    const c = new URLSearchParams(window.location.search).get("class");
    return new Set(c ? c.split(",") : []);
  });

  const loadSchema = useCallback(async () => {
    const d = await getSchema();
    setDoc(d); setDraft(d.schema); setYamlText(d.yaml); setErrors(d.errors);
  }, []);
  const loadMetrics = useCallback(async () => {
    if (!datasetId) return;
    try {
      const [m, o] = await Promise.all([getSchemaMetrics(datasetId), listSchemaDatasets().catch(() => [])]);
      setMetrics(m); setOverview(o); setMetricsErr(null);
      getSchemaSuggestions(datasetId).then((x) => setReady(x.groups.filter((g) => g.ready).length)).catch(() => {});
    } catch (e: any) {
      setMetrics(null); setMetricsErr(String(e?.message ?? e));
    }
  }, [datasetId]);

  useEffect(() => { loadSchema().catch((e) => setMsg({ ok: false, text: String(e?.message ?? e) })); }, [loadSchema]);
  useEffect(() => { loadMetrics(); }, [loadMetrics]);

  const dirty = useMemo(() => !!doc && !!draft && JSON.stringify(doc.schema) !== JSON.stringify(draft), [doc, draft]);
  const yamlDirty = !!doc && yamlText !== doc.yaml;

  // live validation of form edits
  useEffect(() => {
    if (!draft || !dirty) { if (doc) setErrors(doc.errors); return; }
    const t = window.setTimeout(() => validateSchema({ schema: draft }).then((r) => setErrors(r.errors)).catch(() => {}), 350);
    return () => window.clearTimeout(t);
  }, [draft, dirty, doc]);

  const edit = (fn: (s: SemanticSchema) => void) =>
    setDraft((d) => { if (!d) return d; const c = structuredClone(d); fn(c); return c; });

  async function save(body: { schema?: SemanticSchema; yaml?: string }) {
    if (!doc) return;
    setSaving(true); setMsg(null);
    try {
      const d = await saveSchema({ ...body, base_version: doc.version });
      setDoc(d); setDraft(d.schema); setYamlText(d.yaml); setErrors(d.errors);
      setMsg({ ok: true, text: `Saved as v${d.version}.` });
      loadMetrics();
    } catch (e: any) {
      setMsg({ ok: false, text: String(e?.message ?? e) });
    } finally {
      setSaving(false);
    }
  }

  if (!doc || !draft) {
    return <section className="card text-sm text-muted">{msg?.text ?? "Loading the semantic model…"}</section>;
  }

  const s = draft;
  const classNames = Object.keys(s.classes);
  const stale = metrics ? Object.entries(metrics.taxonomies).filter(([, v]) => v.stale).map(([k]) => k) : [];
  const exportPath = `server/datasets/incoming/${metrics?.dataset ?? datasetId.split("~")[0]}`;

  return (
    <section className="card">
      {/* header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="caption mr-1">Semantic model</h2>
        <span className="font-mono text-[11px] text-muted">v{doc.version} · {classNames.length} classes · {Object.keys(s.relations).length} relations</span>
        {metrics && (
          <>
            <span className="font-mono text-[11px] text-muted">· measured on {metrics.dataset}</span>
            <span className={clsx("rounded border px-1.5 py-px font-mono text-[10.5px]", metrics.violations ? "border-bad/60 text-bad" : "border-good/50 text-good")}>
              {metrics.violations ? `${metrics.violations} edges break the schema` : "graph conforms"}
            </span>
            {(metrics.property_violations ?? 0) > 0 && (
              <span className="rounded border border-bad/60 px-1.5 py-px font-mono text-[10.5px] text-bad">
                {metrics.property_violations} property values of the wrong type
              </span>
            )}
            {stale.length > 0 && (
              <span className="rounded border border-warn/60 px-1.5 py-px font-mono text-[10.5px] text-warn">
                {stale.join(", ")}: tagged with an older vocabulary
              </span>
            )}
          </>
        )}
        <div className="ml-auto flex items-center gap-2">
          {msg && <span className={clsx("text-xs", msg.ok ? "text-good" : "text-bad")}>{msg.text}</span>}
          {dirty && (
            <>
              <button className="btn" onClick={() => setDraft(doc.schema)} disabled={saving}><RotateCcw size={12} /> Discard</button>
              <button className="btn border-local/60" onClick={() => save({ schema: draft })} disabled={saving || errors.length > 0}
                title={errors.length ? "Fix the errors below first" : "Save as the next version"}>
                <Save size={12} /> {saving ? "Saving…" : "Save"}
              </button>
            </>
          )}
        </div>
      </div>
      {metricsErr && <p className="mb-2 text-xs text-muted">No metrics for this dataset: {metricsErr}</p>}
      {errors.length > 0 && (
        <ul className="mb-3 rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-ink">
          {errors.map((e) => <li key={e} className="flex gap-2"><AlertTriangle size={12} className="mt-0.5 shrink-0 text-bad" />{e}</li>)}
        </ul>
      )}

      {/* tabs */}
      <div role="tablist" className="mb-3 flex flex-wrap gap-0.5 rounded-md border border-line bg-bg p-0.5 font-mono text-[11px]">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            className={clsx("rounded px-2.5 py-1 transition-colors", tab === t ? "bg-surface-2 text-ink" : "text-muted hover:text-ink")}>
            {t}
            {t === "suggestions" && ready > 0 && (
              <span className="ml-1 rounded-full bg-local/20 px-1.5 text-[10px] text-local">{ready}</span>
            )}
          </button>
        ))}
      </div>

      {tab === "map" && (
        <div className="flex flex-col gap-3">
          <SchemaMap schema={s} metrics={metrics} selected={sel} onSelect={setSel} />
          <MapDetail schema={s} metrics={metrics} sel={sel} onEdit={(t) => setTab(t)} />
        </div>
      )}

      {tab === "classes" && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead><tr className="caption text-left">
              <th className={th}>class</th><th className={th}>parent</th><th className={th}>status</th>
              <th className={th}>description</th><th className={th}>key</th><th className={th}>source</th>
              <th className={clsx(th, "text-right")}>in graph</th><th className={th}>properties</th><th />
            </tr></thead>
            <tbody>
              {classNames.map((c) => {
                const d = s.classes[c] ?? {};
                const own = Object.keys(d.properties ?? {}).length;
                const pm = metrics?.properties?.[c];
                const bad = pm ? Object.values(pm.properties).reduce((n, p) => n + p.bad, 0) : 0;
                const undeclared = pm ? Object.keys(pm.undeclared).length : 0;
                const open = openProps.has(c);
                return (
                  <Fragment key={c}>
                  <tr className="border-t border-line">
                    <td className={clsx(td, "w-36")}><NameInput value={c} onCommit={(v) => edit((x) => renameClass(x, c, v))}
                      className={d.abstract ? "uppercase tracking-wider text-muted" : ""} /></td>
                    <td className={clsx(td, "w-32")}>
                      <select value={d.parent ?? ""} onChange={(e) => edit((x) => { x.classes[c] = { ...x.classes[c], parent: e.target.value || undefined }; })}
                        className="w-full rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink outline-none">
                        <option value="">—</option>
                        {classNames.filter((p) => p !== c).map((p) => <option key={p} value={p}>{p}</option>)}
                      </select>
                    </td>
                    <td className={clsx(td, "w-28")}>
                      {d.abstract
                        ? <span className="px-1.5 font-mono text-[11px] text-muted">group</span>
                        : <StatusSelect value={d.status} onChange={(v) => edit((x) => { x.classes[c] = { ...x.classes[c], status: v }; })} />}
                    </td>
                    <td className={td}><Field value={d.description ?? ""} onChange={(v) => edit((x) => { x.classes[c] = { ...x.classes[c], description: v }; })} /></td>
                    <td className={clsx(td, "w-40")}><Field mono value={d.key ?? ""} onChange={(v) => edit((x) => { x.classes[c] = { ...x.classes[c], key: v || undefined }; })} /></td>
                    <td className={clsx(td, "w-40")}><Field mono value={d.source ?? ""} onChange={(v) => edit((x) => { x.classes[c] = { ...x.classes[c], source: v || undefined }; })} /></td>
                    <td className={clsx(td, "w-20 pr-2 text-right font-mono text-xs tabular text-muted")}>
                      {metrics ? (d.abstract ? metrics.classes[c]?.total : metrics.classes[c]?.count) ?? 0 : "—"}
                    </td>
                    <td className={clsx(td, "w-28")}>
                      <button aria-expanded={open} title="Show and edit this class's properties"
                        onClick={() => setOpenProps((o) => { const n = new Set(o); if (n.has(c)) n.delete(c); else n.add(c); return n; })}
                        className={clsx("rounded px-1.5 py-0.5 font-mono text-[11px] hover:bg-surface-2",
                          bad ? "text-bad" : undeclared ? "text-warn" : open ? "text-ink" : "text-muted")}>
                        {open ? "▾" : "▸"} {own}{bad ? ` · ${bad} bad` : undeclared ? ` · +${undeclared}` : ""}
                      </button>
                    </td>
                    <td className={clsx(td, "w-8")}>
                      <button aria-label={`Delete class ${c}`} className="p-1 text-muted hover:text-bad"
                        onClick={() => edit((x) => { delete x.classes[c]; })}><Trash2 size={12} /></button>
                    </td>
                  </tr>
                  {open && (
                    <tr className="bg-surface-2/30">
                      <td colSpan={9} className="px-3 pb-3 pt-1">
                        <PropertyEditor schema={s} cls={c} metrics={pm}
                          onChange={(props) => edit((x) => { x.classes[c] = { ...x.classes[c], properties: Object.keys(props).length ? props : undefined }; })} />
                      </td>
                    </tr>
                  )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          {metrics && Object.keys(metrics.unknown_classes).length > 0 && (
            <p className="mt-2 text-xs text-warn">
              In the graph but not in the schema: {Object.entries(metrics.unknown_classes).map(([k, n]) => `${k} (${n})`).join(", ")}
            </p>
          )}
          <div className="mt-2 flex gap-2">
            <button className="btn" onClick={() => edit((x) => { x.classes[uniq(x.classes, "NewClass")] = { status: "proposed", description: "" }; })}>
              <Plus size={12} /> class
            </button>
            <button className="btn" onClick={() => edit((x) => { x.classes[uniq(x.classes, "NewGroup")] = { abstract: true, description: "" }; })}>
              <Plus size={12} /> group
            </button>
          </div>
        </div>
      )}

      {tab === "relations" && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead><tr className="caption text-left">
              <th className={th}>relation</th><th className={th}>from</th><th className={th}>to</th><th className={th}>status</th>
              <th className={th}>source</th><th className={clsx(th, "text-right")}>edges</th><th className={clsx(th, "text-right")}>break from/to</th><th />
            </tr></thead>
            <tbody>
              {Object.entries(s.relations).map(([r, d]) => {
                const m = metrics?.relations[r];
                return (
                  <tr key={r} className="border-t border-line">
                    <td className={clsx(td, "w-32")}><NameInput value={r} onCommit={(v) => edit((x) => renameRelation(x, r, v))} /></td>
                    <td className={clsx(td, "w-48")}><ListInput value={d.from} placeholder="User, Chat" onCommit={(v) => edit((x) => { x.relations[r] = { ...x.relations[r], from: v }; })} /></td>
                    <td className={clsx(td, "w-48")}><ListInput value={d.to} placeholder="Topic" onCommit={(v) => edit((x) => { x.relations[r] = { ...x.relations[r], to: v }; })} /></td>
                    <td className={clsx(td, "w-28")}><StatusSelect value={d.status} onChange={(v) => edit((x) => { x.relations[r] = { ...x.relations[r], status: v }; })} /></td>
                    <td className={td}><Field mono value={d.source ?? ""} onChange={(v) => edit((x) => { x.relations[r] = { ...x.relations[r], source: v || undefined }; })} /></td>
                    <td className={clsx(td, "w-16 pr-2 text-right font-mono text-xs tabular text-muted")}>{m ? m.count : "—"}</td>
                    <td className={clsx(td, "w-24 pr-2 text-right font-mono text-xs tabular", m?.violations ? "text-bad" : "text-muted")}
                      title={m?.examples.join("\n")}>{m ? m.violations : "—"}</td>
                    <td className={clsx(td, "w-8")}>
                      <button aria-label={`Delete relation ${r}`} className="p-1 text-muted hover:text-bad"
                        onClick={() => edit((x) => { delete x.relations[r]; })}><Trash2 size={12} /></button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {metrics && Object.keys(metrics.unknown_relations).length > 0 && (
            <p className="mt-2 text-xs text-warn">
              Edge types in the graph the schema doesn't define: {Object.entries(metrics.unknown_relations).map(([k, n]) => `${k} (${n})`).join(", ")}
            </p>
          )}
          <button className="btn mt-2" onClick={() => edit((x) => { x.relations[uniq(x.relations, "new_relation")] = { from: [], to: [], status: "proposed" }; })}>
            <Plus size={12} /> relation
          </button>
        </div>
      )}

      {tab === "taxonomies" && (
        <div className="flex flex-col gap-5">
          <p className="text-xs text-muted">
            A taxonomy is a controlled list of terms for labelling records. <b className="font-semibold text-ink">assigned by llm</b>: the
            classifier (tag_docs.py) picks terms for each record; today it runs <span className="font-mono">topics</span> and{" "}
            <span className="font-mono">doc_types</span>. <b className="font-semibold text-ink">assigned by author</b>: the people who publish
            the items set the tags themselves (Agent Hub manifest tags); this taxonomy maps their free-form tags onto your terms through each
            term's name and aliases. Nothing is relabelled and the graph doesn't change; the counts show how much of what authors wrote your
            terms cover. Edits here stay in the draft until you press Save, which writes schema.yaml as the next version.
          </p>
          {Object.entries(s.taxonomies).map(([name, t]) => {
            const m = metrics?.taxonomies[name];
            const maxCount = Math.max(1, ...Object.values(m?.terms ?? {}), ...Object.values(m?.groups ?? {}));
            const author = t.assigned_by === "author";
            const core = CORE_TAXONOMIES.includes(name);
            const setTax = (patch: Partial<SchemaTaxonomy>) => edit((x) => { x.taxonomies[name] = { ...x.taxonomies[name], ...patch }; });
            return (
              <div key={name} className="rounded-md border border-line p-3">
                <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                  {core
                    ? <h3 className="font-title text-sm font-semibold" title="tag_docs.py reads this taxonomy by name">{name}</h3>
                    : <div className="w-44"><NameInput value={name} className="font-title font-semibold"
                        onCommit={(v) => edit((x) => { x.taxonomies = renameKey(x.taxonomies, name, v); })} /></div>}
                  <div className="min-w-[16rem] flex-1"><Field value={t.description ?? ""} onChange={(v) => setTax({ description: v })} placeholder="what this taxonomy labels" /></div>
                  <button aria-label={`Delete taxonomy ${name}`} title="Delete this taxonomy" className="p-1 text-muted hover:text-bad"
                    onClick={() => {
                      const warn = core ? `\n\ntag_docs.py reads "${name}" by name: without it the classifier stops assigning it.` : "";
                      if (window.confirm(`Delete the taxonomy "${name}" and its ${Object.keys(t.terms ?? {}).length} top-level terms from the draft?${warn}`))
                        edit((x) => { delete x.taxonomies[name]; });
                    }}><Trash2 size={13} /></button>
                </div>
                <TaxonomySettings name={name} t={t} classNames={classNames} onChange={setTax} />
                {m && (
                  <div className="mb-3 grid max-w-xl grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 text-xs">
                    <Bar value={m.assigned} max={m.records} tone={m.assigned < m.records ? "warn" : "local"} />
                    <span className="font-mono text-[11px] tabular text-muted">
                      {author ? `${m.assigned}/${m.records} items have a tag these terms cover` : `${m.assigned}/${m.records} records classified`}
                    </span>
                  </div>
                )}
                {m?.stale && (
                  <div className="mb-3 rounded-md border border-warn/50 bg-warn/10 px-3 py-2 text-xs">
                    <div className="mb-1 text-warn">
                      {m.tagged_vocab ? "This dataset was classified with an older set of terms." : "This dataset hasn't been classified with these terms yet."}{" "}
                      Re-run the LLM classifier, then rebuild the metadata graph.
                      {m.tagged_vocab ? " --incremental only asks about the added or changed terms and keeps every other tag; drop it to re-classify everything." : ""}
                    </div>
                    <code className="block whitespace-pre-wrap font-mono text-[11px] text-ink">
                      python server/ingest/tag_docs.py {exportPath}{m.tagged_vocab ? " --incremental" : ""}{"\n"}python server/ingest/build_metadata_graph.py {exportPath}
                    </code>
                  </div>
                )}
                <TermTree terms={t.terms} counts={{ ...(m?.terms ?? {}), ...(m?.groups ?? {}) }} max={maxCount} hasMetrics={!!m}
                  showAliases={author}
                  onChange={(terms) => edit((x) => { x.taxonomies[name] = { ...x.taxonomies[name], terms }; })} />
                {m && author && Object.keys(m.not_in_schema).length > 0 && (
                  <div className="mt-3">
                    <div className="caption mb-1 text-[10px]">Author tags no term or alias covers yet · click one to add it as a term, or type it into a term's aliases</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(m.not_in_schema).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
                        <button key={k} title={`on ${n} item(s) · add as a term`} className="rounded border border-line px-1.5 py-px font-mono text-[10.5px] text-muted hover:border-local/60 hover:text-ink"
                          onClick={() => edit((x) => { x.taxonomies[name].terms[k] = { description: "" }; })}>
                          {k} <span className="text-muted/60">{n}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {m && !author && Object.keys(m.not_in_schema).length > 0 && (
                  <p className="mt-2 text-xs text-warn">Assigned but no longer in the schema: {Object.entries(m.not_in_schema).map(([k, n]) => `${k} (${n})`).join(", ")}</p>
                )}
                {m && Object.keys(m.graph_terms_not_in_schema).some((k) => !(t.rejected ?? []).includes(k)) && (
                  <div className="mt-3">
                    <div className="caption mb-1 text-[10px]">{t.class} nodes in the graph that aren't terms (e.g. Hub items' own tags) · click to add</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(m.graph_terms_not_in_schema).filter(([k]) => !(t.rejected ?? []).includes(k)).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
                        <button key={k} title={`${n} edge(s) · add as a term`} className="rounded border border-line px-1.5 py-px font-mono text-[10.5px] text-muted hover:border-local/60 hover:text-ink"
                          onClick={() => edit((x) => { x.taxonomies[name].terms[k] = { description: "" }; })}>
                          {k} <span className="text-muted/60">{n}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
          <button className="btn self-start" onClick={() => edit((x) => {
            x.taxonomies[uniq(x.taxonomies, "new-taxonomy")] = {
              description: "", class: "Topic", applies_to: [], assigned_by: "author", terms: { "new-term": { description: "" } },
            };
          })}>
            <Plus size={12} /> taxonomy
          </button>
        </div>
      )}

      {tab === "suggestions" && (
        <Suggestions datasetId={datasetId} doc={doc} dirty={dirty || yamlDirty} exportPath={exportPath}
          onSaved={(d) => { setDoc(d); setDraft(d.schema); setYamlText(d.yaml); setErrors(d.errors); loadMetrics(); }} />
      )}

      {tab === "glossary" && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead><tr className="caption text-left">
              <th className={th}>term</th><th className={th}>definition</th><th className={th}>not to be confused with</th>
              <th className={th}>synonyms</th><th className={clsx(th, "text-right")}>docs using it</th><th />
            </tr></thead>
            <tbody>
              {Object.entries(s.glossary).map(([g, d]) => (
                <tr key={g} className="border-t border-line">
                  <td className={clsx(td, "w-32")}><NameInput value={g} onCommit={(v) => edit((x) => { x.glossary = renameKey(x.glossary, g, v); })} /></td>
                  <td className={td}><Field value={d.definition ?? ""} onChange={(v) => edit((x) => { x.glossary[g] = { ...x.glossary[g], definition: v }; })} /></td>
                  <td className={td}><Field value={d.not ?? ""} onChange={(v) => edit((x) => { x.glossary[g] = { ...x.glossary[g], not: v || undefined }; })} /></td>
                  <td className={clsx(td, "w-48")}><ListInput value={d.synonyms ?? []} onCommit={(v) => edit((x) => { x.glossary[g] = { ...x.glossary[g], synonyms: v.length ? v : undefined }; })} /></td>
                  <td className={clsx(td, "w-24 pr-2 text-right font-mono text-xs tabular text-muted")}>{metrics ? metrics.glossary[g] ?? 0 : "—"}</td>
                  <td className={clsx(td, "w-8")}>
                    <button aria-label={`Delete term ${g}`} className="p-1 text-muted hover:text-bad"
                      onClick={() => edit((x) => { delete x.glossary[g]; })}><Trash2 size={12} /></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button className="btn mt-2" onClick={() => edit((x) => { x.glossary[uniq(x.glossary, "new term")] = { definition: "" }; })}>
            <Plus size={12} /> term
          </button>
        </div>
      )}

      {tab === "rules" && (
        <div className="flex flex-col gap-3">
          <p className="text-xs text-muted">Derived edges: paths through the graph that imply a new relation. Counts are how many edges the rule would add to this dataset.</p>
          {Object.entries(s.rules).map(([r, d]) => {
            const m = metrics?.rules[r];
            return (
              <div key={r} className="rounded-md border border-line p-3">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="w-40"><NameInput value={r} onCommit={(v) => edit((x) => { x.rules = renameKey(x.rules, r, v); })} /></span>
                  <span className="font-mono text-[11px] text-muted">{d.from} → {d.to}</span>
                  <StatusSelect value={d.status} onChange={(v) => edit((x) => { x.rules[r] = { ...x.rules[r], status: v }; })} />
                  <span className="ml-auto font-mono text-[11px] text-muted">{m ? `${m.edges} edges` : "—"}</span>
                  <button aria-label={`Delete rule ${r}`} className="p-1 text-muted hover:text-bad" onClick={() => edit((x) => { delete x.rules[r]; })}><Trash2 size={12} /></button>
                </div>
                <Field value={d.description ?? ""} onChange={(v) => edit((x) => { x.rules[r] = { ...x.rules[r], description: v }; })} />
                <div className="mt-1 grid gap-1 sm:grid-cols-2">
                  <label className="text-[11px] text-muted">from class
                    <select value={d.from} onChange={(e) => edit((x) => { x.rules[r] = { ...x.rules[r], from: e.target.value }; })}
                      className="ml-2 rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink">
                      {classNames.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </label>
                  <label className="text-[11px] text-muted">to class
                    <select value={d.to} onChange={(e) => edit((x) => { x.rules[r] = { ...x.rules[r], to: e.target.value }; })}
                      className="ml-2 rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink">
                      {classNames.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </label>
                </div>
                <div className="mt-1 text-[11px] text-muted">paths (one per line, relations separated by commas)</div>
                <PathsInput value={d.via} onCommit={(via) => edit((x) => { x.rules[r] = { ...x.rules[r], via }; })} />
                {m && m.sample.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {m.sample.map((p) => <span key={p} className="rounded bg-surface-2 px-1.5 py-px font-mono text-[10.5px] text-muted">{p}</span>)}
                    {m.edges > m.sample.length && <span className="font-mono text-[10.5px] text-muted">+{m.edges - m.sample.length} more</span>}
                  </div>
                )}
              </div>
            );
          })}
          <button className="btn self-start" onClick={() => edit((x) => {
            x.rules[uniq(x.rules, "new_rule")] = { from: classNames[0] ?? "", to: classNames[0] ?? "", via: [], status: "proposed" };
          })}><Plus size={12} /> rule</button>
        </div>
      )}

      {tab === "yaml" && (
        <div>
          <p className="mb-2 text-xs text-muted">
            The file itself ({doc.path}). Saving here replaces the whole schema; the previous version is kept under history/.
            {dirty && <span className="text-warn"> You have unsaved form edits: save or discard them before editing the YAML.</span>}
          </p>
          <textarea value={yamlText} onChange={(e) => setYamlText(e.target.value)} disabled={dirty} spellCheck={false}
            className="input h-[520px] font-mono text-xs leading-relaxed" />
          <div className="mt-2 flex gap-2">
            <button className="btn border-local/60" disabled={!yamlDirty || saving || dirty} onClick={() => save({ yaml: yamlText })}><Save size={12} /> Save YAML</button>
            <button className="btn" disabled={!yamlDirty} onClick={() => setYamlText(doc.yaml)}><RotateCcw size={12} /> Revert</button>
          </div>
        </div>
      )}

      {tab === "history" && <History metrics={metrics} overview={overview} doc={doc} />}
    </section>
  );
}

function uniq(obj: Record<string, unknown>, base: string) {
  let n = base, i = 2;
  while (n in obj) n = `${base}${i++}`;
  return n;
}

function PathsInput({ value, onCommit }: { value: string[][]; onCommit: (v: string[][]) => void }) {
  const text = (value ?? []).map((p) => p.join(", ")).join("\n");
  const [v, setV] = useState(text);
  useEffect(() => setV(text), [text]);
  return (
    <textarea value={v} onChange={(e) => setV(e.target.value)} rows={Math.max(2, (value ?? []).length)}
      onBlur={() => onCommit(v.split("\n").map(csv).filter((p) => p.length))}
      className="input mt-1 py-1 font-mono text-[11px]" />
  );
}

const PROP_TYPES: SchemaPropertyType[] = ["string", "integer", "number", "date", "boolean", "enum"];

/** Properties a class inherits from its parent groups (shown read-only under the class). */
function inheritedProps(s: SemanticSchema, cls: string): [string, SchemaProperty, string][] {
  const out: [string, SchemaProperty, string][] = [];
  const own = s.classes[cls]?.properties ?? {};
  const seen = new Set([cls]);
  let p = s.classes[cls]?.parent ?? undefined;
  while (p && s.classes[p] && !seen.has(p)) {
    seen.add(p);
    for (const [n, d] of Object.entries(s.classes[p]?.properties ?? {})) if (!(n in own)) out.push([n, d, p]);
    p = s.classes[p]?.parent ?? undefined;
  }
  return out;
}

/** A class's properties: the structured facts the graph build writes on its nodes. Each row shows
 * how many of the class's nodes in the selected dataset carry the property, and values of the
 * wrong type; attributes found on nodes but not declared can be added with one click. */
function PropertyEditor({ schema, cls, metrics, onChange }: {
  schema: SemanticSchema; cls: string; metrics?: ClassPropertyMetrics;
  onChange: (p: Record<string, SchemaProperty>) => void;
}) {
  const props = schema.classes[cls]?.properties ?? {};
  const inherited = inheritedProps(schema, cls);
  const set = (n: string, d: SchemaProperty) => onChange({ ...props, [n]: d });
  const sel = "rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink outline-none";
  const taxonomies = Object.keys(schema.taxonomies);
  const coverage = (n: string) => {
    const m = metrics?.properties[n];
    if (!m) return <span className="text-muted/60">—</span>;
    return (
      <div className="flex items-center gap-2" title={m.examples.length ? `e.g. ${m.examples.join(", ")}` : undefined}>
        <div className="w-20"><Bar value={m.count} max={m.total} tone={m.bad ? "bad" : m.count < m.total ? "warn" : "local"} /></div>
        <span className="tabular">{m.count}/{m.total}</span>
        {m.bad > 0 && <span className="text-bad" title={`e.g. ${m.bad_examples.join(", ")}`}>{m.bad} wrong type</span>}
      </div>
    );
  };
  return (
    <div className="text-xs">
      <p className="mb-2 text-muted">
        Properties are structured facts on each <b className="font-semibold text-ink">{cls}</b> node (a file's format, a chat's start date).
        Coverage is how many of this dataset's {metrics?.nodes ?? "—"} {cls} nodes have a value. Questions like "how many PDFs?" are
        answered from these, so a fact the graph needs to count or filter by belongs here, not only in the card text.
      </p>
      <table className="w-full min-w-[820px]">
        <thead><tr className="caption text-left text-[10px]">
          <th className={th}>property</th><th className={th}>type</th><th className={th}>allowed values</th>
          <th className={th}>description</th><th className={th}>source</th><th className={th}>in graph</th><th />
        </tr></thead>
        <tbody>
          {Object.entries(props).map(([n, d]) => (
            <tr key={n} className="border-t border-line/60">
              <td className={clsx(td, "w-36")}><NameInput value={n} className="font-mono" onCommit={(v) => onChange(renameKey(props, n, v))} /></td>
              <td className={clsx(td, "w-24")}>
                <select value={d.type ?? "string"} className={sel}
                  onChange={(e) => set(n, { ...d, type: e.target.value as SchemaPropertyType })}>
                  {PROP_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </td>
              <td className={clsx(td, "w-52")}>
                {d.type === "enum" ? (
                  <div className="flex flex-col gap-0.5">
                    <select value={d.taxonomy ?? ""} className={sel} title="Take the allowed values from a taxonomy's terms"
                      onChange={(e) => set(n, { ...d, taxonomy: e.target.value || undefined })}>
                      <option value="">values listed here</option>
                      {taxonomies.map((t) => <option key={t} value={t}>terms of {t}</option>)}
                    </select>
                    {!d.taxonomy && <ListInput value={d.values ?? []} placeholder="comma-separated values"
                      onCommit={(v) => set(n, { ...d, values: v.length ? v : undefined })} />}
                  </div>
                ) : <span className="px-1.5 text-muted/60">any {d.type ?? "string"}</span>}
              </td>
              <td className={td}><Field value={d.description ?? ""} onChange={(v) => set(n, { ...d, description: v })} placeholder="what this fact means" /></td>
              <td className={clsx(td, "w-44")}><Field mono value={d.source ?? ""} onChange={(v) => set(n, { ...d, source: v || undefined })} placeholder="where it comes from" /></td>
              <td className={clsx(td, "w-44 font-mono text-[11px]")}>{coverage(n)}</td>
              <td className={clsx(td, "w-8")}>
                <button aria-label={`Delete property ${n}`} className="p-1 text-muted hover:text-bad"
                  onClick={() => { const c = { ...props }; delete c[n]; onChange(c); }}><Trash2 size={12} /></button>
              </td>
            </tr>
          ))}
          {inherited.map(([n, d, from]) => (
            <tr key={`inh-${n}`} className="border-t border-line/60 text-muted">
              <td className={clsx(td, "px-2 font-mono")}>{n}</td>
              <td className={clsx(td, "px-2 font-mono text-[11px]")}>{d.type ?? "string"}</td>
              <td className={clsx(td, "px-2")} colSpan={3}>inherited from {from}: {d.description}</td>
              <td className={clsx(td, "font-mono text-[11px]")}>{coverage(n)}</td>
              <td />
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button className="btn" onClick={() => set(uniq(props, "new_property"), { type: "string", description: "" })}>
          <Plus size={12} /> property
        </button>
        {metrics && Object.keys(metrics.undeclared).length > 0 && (
          <>
            <span className="text-warn">On {cls} nodes but not declared · click to add:</span>
            {Object.entries(metrics.undeclared).map(([k, n]) => (
              <button key={k} title={`${n} node(s) have it`} onClick={() => set(k, { type: "string", description: "" })}
                className="rounded border border-warn/50 px-1.5 py-px font-mono text-[10.5px] text-warn hover:text-ink">
                {k} <span className="text-muted/60">{n}</span>
              </button>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

/** A taxonomy's settings: which node class its terms are, which records it labels, who assigns it. */
function TaxonomySettings({ name, t, classNames, onChange }: {
  name: string; t: SchemaTaxonomy; classNames: string[]; onChange: (patch: Partial<SchemaTaxonomy>) => void;
}) {
  const applies = t.applies_to ?? [];
  const sel = "rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink outline-none";
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-muted">
      <label className="flex items-center gap-1" title="The node class its terms become in the graph (Topic for subjects and tags)">
        terms are
        <select value={t.class ?? ""} onChange={(e) => onChange({ class: e.target.value || undefined })} className={sel}>
          <option value="">—</option>
          {classNames.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>
      <label className="flex items-center gap-1" title="llm: tag_docs.py picks terms. author: the items' own tags are matched to terms by name or alias.">
        assigned by
        <select value={t.assigned_by ?? "llm"} onChange={(e) => onChange({ assigned_by: e.target.value })} className={sel}>
          <option value="llm">llm (the classifier picks)</option>
          <option value="author">author (item tags, matched by alias)</option>
        </select>
      </label>
      {t.assigned_by === "author" ? (
        <span title="Authors choose their own tags, so there is nothing to limit">up to: not enforced (authors choose their tags)</span>
      ) : name === "doc_types" ? (
        <span title="The classifier is built to pick one type per file">up to 1 per record (the classifier picks exactly one)</span>
      ) : (
        <label className="flex items-center gap-1"
          title={name === "topics" ? "The classifier gives each record at most this many terms on its next run (empty: 4)" : "Most terms one record may get"}>
          up to
          <input type="number" min={1} value={t.max_per_record ?? ""} placeholder={name === "topics" ? "4" : ""} className={clsx(sel, "w-14")}
            onChange={(e) => onChange({ max_per_record: e.target.value ? Math.max(1, parseInt(e.target.value, 10) || 1) : undefined })} />
          per record
        </label>
      )}
      <div className="flex flex-wrap items-center gap-1" title="Which records this taxonomy labels (a parent class includes its children)">
        applies to
        {classNames.map((c) => {
          const on = applies.includes(c);
          return (
            <button key={c} aria-pressed={on}
              onClick={() => onChange({ applies_to: on ? applies.filter((x) => x !== c) : [...applies, c] })}
              className={clsx("rounded border px-1.5 py-px font-mono text-[10.5px]", on ? "border-local/60 bg-local/10 text-ink" : "border-line text-muted/70 hover:text-ink")}>
              {c}
            </button>
          );
        })}
        {(name === "topics" || name === "doc_types") && t.assigned_by !== "author" && (
          <span className="text-muted/80">(the classifier only reads Chat and File)</span>
        )}
      </div>
      {t.assigned_by !== "author" && name !== "topics" && name !== "doc_types" && (
        <p className="w-full text-warn">
          Nothing assigns this taxonomy yet: the classifier only runs topics and doc_types. Use "author" if these terms come from item tags.
        </p>
      )}
    </div>
  );
}

function TermTree({ terms, counts, max, hasMetrics, onChange, showAliases = false }: {
  terms: Record<string, SchemaTerm>; counts: Record<string, number>; max: number; hasMetrics: boolean;
  onChange: (t: Record<string, SchemaTerm>) => void; showAliases?: boolean;
}) {
  const set = (name: string, t: SchemaTerm) => onChange({ ...terms, [name]: t });
  const remove = (name: string) => { const c = { ...terms }; delete c[name]; onChange(c); };
  return (
    <div className="flex flex-col">
      {Object.entries(terms).map(([name, t]) => {
        const group = !!t.children && Object.keys(t.children).length > 0;
        return (
          <div key={name} className={clsx(group && "mb-1 mt-1")}>
            <div className="grid grid-cols-[minmax(120px,180px)_1fr_minmax(80px,140px)_auto] items-center gap-2">
              <NameInput value={name} onCommit={(v) => onChange(renameKey(terms, name, v))}
                className={group ? "font-semibold uppercase tracking-wider text-muted" : ""} />
              <div className="flex flex-col">
                <Field value={t.description ?? ""} onChange={(v) => set(name, { ...t, description: v })}
                placeholder={showAliases ? "note for people: what this term covers" : "what the LLM should pick this for"} />
                {!group && (showAliases || (t.aliases?.length ?? 0) > 0) && (
                  <div className="flex items-center gap-1 pl-1.5">
                    <span className="shrink-0 font-mono text-[10.5px] text-muted">aliases</span>
                    <ListInput value={t.aliases ?? []} placeholder="other words that mean this term, comma-separated"
                      onCommit={(v) => set(name, { ...t, aliases: v.length ? v : undefined })} />
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {hasMetrics && <Bar value={counts[name] ?? 0} max={max} />}
                <span className="w-6 text-right font-mono text-[11px] tabular text-muted">{hasMetrics ? counts[name] ?? 0 : ""}</span>
              </div>
              <div className="flex">
                <button title={group ? "Add a term to this group" : "Make this a group with a sub-term"} aria-label={`Add a term under ${name}`}
                  className="p-1 text-muted hover:text-ink"
                  onClick={() => set(name, { ...t, children: { ...(t.children ?? {}), [uniq(t.children ?? {}, "new-term")]: { description: "" } } })}>
                  <Plus size={12} />
                </button>
                <button aria-label={`Delete ${name}`} className="p-1 text-muted hover:text-bad" onClick={() => remove(name)}><Trash2 size={12} /></button>
              </div>
            </div>
            {group && (
              <div className="ml-4 border-l border-line pl-3">
                <TermTree terms={t.children!} counts={counts} max={max} hasMetrics={hasMetrics} showAliases={showAliases}
                  onChange={(children) => set(name, { ...t, children: Object.keys(children).length ? children : undefined })} />
              </div>
            )}
          </div>
        );
      })}
      <button className="btn mt-1 self-start" onClick={() => onChange({ ...terms, [uniq(terms, "new-term")]: { description: "" } })}>
        <Plus size={12} /> term
      </button>
    </div>
  );
}

function MapDetail({ schema, metrics, sel, onEdit }: {
  schema: SemanticSchema; metrics: SchemaMetrics | null; sel: MapSelection; onEdit: (t: Tab) => void;
}) {
  if (!sel) {
    return (
      <aside className="rounded-md border border-line p-3 text-xs text-muted">
        <p className="mb-2">{schema.description}</p>
        <p>Click a class or a relation for its definition and counts. Hover a relation to trace it.</p>
        {metrics && (
          <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px]">
            <dt>docs</dt><dd className="text-ink">{metrics.docs}</dd>
            <dt>nodes</dt><dd className="text-ink">{metrics.nodes}</dd>
            <dt>edges</dt><dd className="text-ink">{metrics.edges}</dd>
            <dt>graph built</dt><dd className="text-ink">{fmtTime(metrics.graph_built_at)}</dd>
            <dt>classified</dt><dd className="text-ink">{fmtTime(metrics.tagged_at)}</dd>
          </dl>
        )}
      </aside>
    );
  }
  if (sel.kind === "class") {
    const c = schema.classes[sel.name];
    const m = metrics?.classes[sel.name];
    const rels = Object.entries(schema.relations).filter(([, r]) => (r.from ?? []).includes(sel.name) || (r.to ?? []).includes(sel.name));
    const taxa = Object.entries(schema.taxonomies).filter(([, t]) => (t.applies_to ?? []).includes(sel.name));
    return (
      <aside className="rounded-md border border-line p-3 text-xs">
        <div className="mb-1 flex items-baseline justify-between">
          <h3 className="font-title text-sm font-semibold">{sel.name}</h3>
          <button className="text-[11px] text-muted hover:text-ink" onClick={() => onEdit("classes")}>edit</button>
        </div>
        {!c ? <p className="text-warn">Not in the schema: the graph has {metrics?.unknown_classes[sel.name]} node(s) of this type.</p> : (
          <>
            <p className="mb-2 text-muted">{c.description}</p>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px]">
              {c.parent && <><dt className="text-muted">parent</dt><dd>{c.parent}</dd></>}
              <dt className="text-muted">status</dt><dd>{c.abstract ? "group" : c.status ?? "implemented"}</dd>
              {c.key && <><dt className="text-muted">key</dt><dd>{c.key}</dd></>}
              {c.source && <><dt className="text-muted">source</dt><dd>{c.source}</dd></>}
              <dt className="text-muted">in graph</dt><dd>{m ? (c.abstract ? m.total : m.count) : "—"}</dd>
            </dl>
            {(() => {
              const own = Object.entries(c.properties ?? {}).map(([n, d]) => [n, d, sel.name] as [string, SchemaProperty, string]);
              const all = [...own, ...inheritedProps(schema, sel.name)];
              const pm = metrics?.properties?.[sel.name];
              if (!all.length && !pm) return null;
              return (
                <>
                  <div className="caption mb-1 mt-3 text-[10px]">properties</div>
                  <ul className="flex flex-col gap-0.5 font-mono text-[11px]">
                    {all.map(([n, d, from]) => {
                      const x = pm?.properties[n];
                      return (
                        <li key={n} className="flex justify-between gap-2" title={d.description}>
                          <span><span className="text-ink">{n}</span> <span className="text-muted">{d.type ?? "string"}{from !== sel.name ? ` (from ${from})` : ""}</span></span>
                          <span className={x?.bad ? "text-bad" : x && x.count < x.total ? "text-warn" : "text-muted"}>
                            {x ? `${x.count}/${x.total}${x.bad ? ` · ${x.bad} bad` : ""}` : "—"}
                          </span>
                        </li>
                      );
                    })}
                    {!all.length && <li className="text-muted">none declared</li>}
                  </ul>
                  {pm && Object.keys(pm.undeclared).length > 0 && (
                    <p className="mt-1 text-[11px] text-warn">on nodes, not declared: {Object.keys(pm.undeclared).join(", ")}</p>
                  )}
                </>
              );
            })()}
            {rels.length > 0 && <div className="caption mb-1 mt-3 text-[10px]">relations</div>}
            <ul className="flex flex-col gap-0.5 font-mono text-[11px]">
              {rels.map(([r, d]) => (
                <li key={r} className="flex justify-between gap-2">
                  <span><span className="text-muted">{(d.from ?? []).join("/")}</span> <span className="text-ink">{r}</span> <span className="text-muted">{(d.to ?? []).join("/")}</span></span>
                  <span className="text-muted">{metrics?.relations[r]?.count ?? "—"}</span>
                </li>
              ))}
            </ul>
            {taxa.length > 0 && <div className="caption mb-1 mt-3 text-[10px]">classified by</div>}
            {taxa.map(([t]) => {
              const tm = metrics?.taxonomies[t];
              return <div key={t} className="font-mono text-[11px]">{t} <span className="text-muted">{tm ? `${tm.assigned}/${tm.records}` : ""}</span></div>;
            })}
          </>
        )}
      </aside>
    );
  }
  const r = schema.relations[sel.name];
  const m = metrics?.relations[sel.name];
  return (
    <aside className="rounded-md border border-line p-3 text-xs">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="font-mono text-sm text-ink">{sel.name}</h3>
        <button className="text-[11px] text-muted hover:text-ink" onClick={() => onEdit("relations")}>edit</button>
      </div>
      {r && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px]">
          <dt className="text-muted">from</dt><dd>{(r.from ?? []).join(", ")}</dd>
          <dt className="text-muted">to</dt><dd>{(r.to ?? []).join(", ")}</dd>
          <dt className="text-muted">status</dt><dd>{r.status ?? "implemented"}</dd>
          {r.source && <><dt className="text-muted">source</dt><dd>{r.source}</dd></>}
          <dt className="text-muted">edges</dt><dd>{m?.count ?? "—"}</dd>
          <dt className="text-muted">break from/to</dt><dd className={m?.violations ? "text-bad" : ""}>{m?.violations ?? "—"}</dd>
        </dl>
      )}
      {m && Object.keys(m.pairs).length > 0 && (
        <>
          <div className="caption mb-1 mt-3 text-[10px]">in this dataset</div>
          <ul className="font-mono text-[11px]">
            {Object.entries(m.pairs).map(([p, n]) => <li key={p} className="flex justify-between"><span>{p}</span><span className="text-muted">{n}</span></li>)}
          </ul>
        </>
      )}
      {m && m.examples.length > 0 && (
        <>
          <div className="caption mb-1 mt-3 text-[10px] text-bad">breaking edges</div>
          <ul className="font-mono text-[10.5px] text-muted">{m.examples.map((e) => <li key={e}>{e}</li>)}</ul>
        </>
      )}
    </aside>
  );
}

function History({ metrics, overview, doc }: { metrics: SchemaMetrics | null; overview: SchemaDatasetRow[]; doc: SchemaDoc }) {
  const snaps = [...(metrics?.history ?? [])].reverse();
  const delta = (a?: number, b?: number) => (a == null || b == null || a === b ? "" : a > b ? ` +${a - b}` : ` ${a - b}`);
  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="caption mb-1">Ingests of {metrics?.dataset ?? "this dataset"}</div>
        <p className="mb-2 text-xs text-muted">One row each time the dataset's metadata graph was rebuilt or re-classified, with what changed since the row before.</p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead><tr className="caption text-left">
              <th className={th}>ingest</th><th className={clsx(th, "text-right")}>docs</th><th className={clsx(th, "text-right")}>nodes</th>
              <th className={clsx(th, "text-right")}>edges</th><th className={clsx(th, "text-right")}>breaking</th>
              <th className={th}>classified</th><th className={th}>what changed</th>
            </tr></thead>
            <tbody>
              {snaps.map((h, i) => {
                const prev = snaps[i + 1];
                // which ingest step made this row: tag_docs (re-classified) and/or build_metadata_graph
                const steps = prev ? [
                  ...(h.tagged_at !== prev.tagged_at ? ["re-classified"] : []),
                  ...(h.graph_built_at !== prev.graph_built_at ? ["graph rebuilt"] : []),
                ] : [];
                const changed = prev ? [
                  ...Object.keys({ ...h.classes, ...prev.classes })
                    .filter((c) => (h.classes[c] ?? 0) !== (prev.classes[c] ?? 0))
                    .map((c) => `${c}${delta(h.classes[c] ?? 0, prev.classes[c] ?? 0)}`),
                  ...Object.keys(h.taxonomies)
                    .filter((t) => h.taxonomies[t]?.assigned !== prev.taxonomies[t]?.assigned)
                    .map((t) => `${t} classified${delta(h.taxonomies[t]?.assigned, prev.taxonomies[t]?.assigned ?? 0)}`),
                ] : ["first measurement"];
                return (
                  <tr key={`${h.graph_built_at}-${h.tagged_at}`} className="border-t border-line font-mono text-xs">
                    <td className="px-2 py-1.5">{fmtTime(Math.max(h.graph_built_at ?? 0, h.tagged_at ?? 0) || null)}
                      {steps.length > 0 && <span className="ml-2 text-muted">{steps.join(" + ")}</span>}</td>
                    <td className="px-2 py-1.5 text-right tabular">{h.docs}<span className="text-good">{delta(h.docs, prev?.docs)}</span></td>
                    <td className="px-2 py-1.5 text-right tabular">{h.nodes}<span className="text-good">{delta(h.nodes, prev?.nodes)}</span></td>
                    <td className="px-2 py-1.5 text-right tabular">{h.edges}<span className="text-good">{delta(h.edges, prev?.edges)}</span></td>
                    <td className={clsx("px-2 py-1.5 text-right tabular", h.violations ? "text-bad" : "text-muted")}>{h.violations}</td>
                    <td className="px-2 py-1.5 text-muted">
                      {Object.entries(h.taxonomies).map(([t, v]) => `${t} ${v.assigned}/${v.records}`).join(" · ")}
                    </td>
                    <td className="px-2 py-1.5 text-muted">{changed.join(", ") || "—"}</td>
                  </tr>
                );
              })}
              {snaps.length === 0 && <tr><td colSpan={7} className="px-2 py-3 text-xs text-muted">No measurements yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div className="caption mb-1">All exports against the current schema</div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead><tr className="caption text-left">
              <th className={th}>dataset</th><th className={clsx(th, "text-right")}>docs</th><th className={clsx(th, "text-right")}>nodes</th>
              <th className={clsx(th, "text-right")}>edges</th><th className={th}>classified</th><th className={th}>conformance</th>
            </tr></thead>
            <tbody>
              {overview.map((o) => (
                <tr key={o.dataset} className="border-t border-line font-mono text-xs">
                  <td className="px-2 py-1.5 text-ink">{o.dataset}</td>
                  <td className="px-2 py-1.5 text-right tabular">{o.docs}</td>
                  <td className="px-2 py-1.5 text-right tabular">{o.has_graph ? o.nodes : "—"}</td>
                  <td className="px-2 py-1.5 text-right tabular">{o.has_graph ? o.edges : "—"}</td>
                  <td className="px-2 py-1.5 text-muted">
                    {Object.entries(o.coverage).map(([t, [a, n]]) => `${t} ${a}/${n}${o.stale.includes(t) ? " (stale)" : ""}`).join(" · ")}
                  </td>
                  <td className="px-2 py-1.5">
                    {!o.has_graph ? <span className="text-muted">no metadata graph</span>
                      : o.violations || o.unknown_classes || o.unknown_relations
                        ? <span className="text-warn">{o.violations} breaking · {o.unknown_classes} unknown types · {o.unknown_relations} unknown relations</span>
                        : <span className="inline-flex items-center gap-1 text-good"><Check size={12} /> conforms</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div className="caption mb-1">Schema versions</div>
        <ul className="font-mono text-xs text-muted">
          <li className="text-ink">v{doc.version} · current</li>
          {doc.history.map((h) => <li key={h.file}>v{h.version} · replaced {h.replaced_at.replace(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/, "$1-$2-$3 $4:$5")}</li>)}
        </ul>
      </div>
    </div>
  );
}
