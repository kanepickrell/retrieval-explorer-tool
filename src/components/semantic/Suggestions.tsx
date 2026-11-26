import { useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { Check, GitMerge, X } from "lucide-react";
import {
  getSchemaSuggestions, reviewSuggestion,
  type ReviewAction, type SchemaDoc, type SchemaSuggestions, type SchemaTerm, type SuggestionGroup,
} from "@/lib/api";

/**
 * Bottom-up vocabulary growth: terms the LLM classifier proposed when no existing term covered
 * a record's main subject (tag_docs.py), grouped and checked against the schema by the backend.
 * Approve adds a term, Merge makes the names aliases of an existing term, Reject keeps them
 * from being suggested again. Every action saves the schema as a new version; the affected
 * records are then re-checked with `tag_docs.py --incremental` (only yes/no questions about
 * the changed terms, every other tag kept).
 */

function walk(terms: Record<string, SchemaTerm>, out: { name: string; group: boolean }[] = []) {
  for (const [n, d] of Object.entries(terms ?? {})) {
    out.push({ name: n, group: !!d?.children && Object.keys(d.children).length > 0 });
    if (d?.children) walk(d.children, out);
  }
  return out;
}

export default function Suggestions({ datasetId, doc, dirty, exportPath, onSaved }: {
  datasetId: string; doc: SchemaDoc; dirty: boolean; exportPath: string; onSaved: (d: SchemaDoc) => void;
}) {
  const [data, setData] = useState<SchemaSuggestions | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [done, setDone] = useState<{ ok: boolean; text: string; rerun?: boolean } | null>(null);

  const load = useCallback(() => {
    getSchemaSuggestions(datasetId).then((d) => { setData(d); setErr(null); }).catch((e) => setErr(String(e?.message ?? e)));
  }, [datasetId]);
  useEffect(() => { load(); }, [load, doc.version]);

  async function act(key: string, action: ReviewAction, body: Parameters<typeof reviewSuggestion>[1], what: string) {
    setBusy(key); setDone(null);
    try {
      const d = await reviewSuggestion(action, body);
      onSaved(d);
      // reject only changes the review list; approve/merge change what the classifier assigns
      setDone({ ok: true, text: `${what} Saved as v${d.version}.`, rerun: action !== "reject" });
    } catch (e: any) {
      setDone({ ok: false, text: String(e?.message ?? e) });
    } finally {
      setBusy(null);
    }
  }

  if (err) return <p className="text-xs text-bad">{err}</p>;
  if (!data) return <p className="text-xs text-muted">Loading suggestions…</p>;

  const ready = data.groups.filter((g) => g.ready);
  const watching = data.groups.filter((g) => !g.ready);
  const locked = dirty ? "Save or discard the form edits first" : undefined;

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted">
        The classifier first names each record's subject (and a file's type) in its own words, then picks from the terms. When that
        label shares no word with any term, the record still gets the closest terms and the label becomes a proposal. Proposals from
        every export are grouped by meaning (one LLM call per tagging run) and checked against the schema.
        One is ready for review once {data.thresholds.min_records}+ records
        {data.thresholds.min_users > 1 ? ` from ${data.thresholds.min_users}+ people` : ""} propose it. Nothing changes until you approve it.
      </p>

      {done && (
        <div className={clsx("rounded-md border px-3 py-2 text-xs", done.ok ? "border-good/50 bg-good/10" : "border-bad/50 bg-bad/10 text-bad")}>
          <div className={done.ok ? "text-good" : ""}>{done.text}</div>
          {done.rerun && (
            <>
              <div className="mt-1 text-muted">Re-check the records this can affect (yes/no questions about the changed terms only), then rebuild the graph:</div>
              <code className="mt-1 block whitespace-pre-wrap font-mono text-[11px] text-ink">
                python server/ingest/tag_docs.py {exportPath} --incremental{"\n"}python server/ingest/build_metadata_graph.py {exportPath}
              </code>
            </>
          )}
        </div>
      )}

      <section>
        <h3 className="caption mb-2">Ready for review · {ready.length}</h3>
        {ready.length === 0 && <p className="text-xs text-muted">No suggestion has enough support yet.</p>}
        <div className="flex flex-col gap-3">
          {ready.map((g) => <GroupCard key={`${g.taxonomy}:${g.term}`} g={g} doc={doc} busy={busy} locked={locked} act={act} />)}
        </div>
      </section>

      {watching.length > 0 && (
        <section>
          <h3 className="caption mb-2">Watching · {watching.length} <span className="normal-case tracking-normal text-muted">(fewer than {data.thresholds.min_records} records so far)</span></h3>
          <div className="flex flex-col gap-2">
            {watching.map((g) => <GroupCard key={`${g.taxonomy}:${g.term}`} g={g} doc={doc} busy={busy} locked={locked} act={act} compact />)}
          </div>
        </section>
      )}

      <section>
        <h3 className="caption mb-2">Term health · this dataset</h3>
        {data.health.length === 0 ? <p className="text-xs text-muted">Nothing to flag (or the dataset's tags are stale, so term usage isn't measured).</p> : (
          <ul className="flex flex-col gap-1 text-xs">
            {data.health.map((h, i) => (
              <li key={i} className="flex flex-wrap items-baseline gap-2">
                <span className={clsx("rounded border px-1.5 py-px font-mono text-[10.5px]",
                  h.kind === "overloaded" ? "border-warn/60 text-warn" : h.kind === "gaps" ? "border-local/60 text-local" : "border-line text-muted")}>
                  {h.kind}
                </span>
                <span className="font-mono text-[11px] text-muted">{h.taxonomy}</span>
                {h.term && <span className="font-mono text-ink">{h.term}</span>}
                <span className="text-muted">{h.note}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {Object.entries(data.hub_tags).map(([t, tags]) => Object.keys(tags).length > 0 && (
        <section key={t}>
          <h3 className="caption mb-1">Hub tags not in {t}</h3>
          <p className="mb-2 text-xs text-muted">
            Free-form tags from Agent Hub items, linked in the graph but not terms. Add one from the taxonomies tab, or reject the ones that will never be terms. Rejected tags are hidden here and in the taxonomies tab.
          </p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(tags).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
              <span key={k} className="inline-flex items-center gap-1 rounded border border-line px-1.5 py-px font-mono text-[10.5px] text-muted">
                {k} <span className="text-muted/60">{n}</span>
                <button title={locked ?? `reject "${k}"`} disabled={!!busy || !!locked} className="hover:text-bad disabled:opacity-40"
                  onClick={() => act(`hub:${k}`, "reject", { taxonomy: t, names: [k], base_version: doc.version }, `Rejected ${k}.`)}>
                  <X size={10} />
                </button>
              </span>
            ))}
            <button className="btn ml-1 py-0 text-[10.5px]" disabled={!!busy || !!locked} title={locked}
              onClick={() => act(`hub:*`, "reject", { taxonomy: t, names: Object.keys(tags), base_version: doc.version }, `Rejected ${Object.keys(tags).length} Hub tags.`)}>
              reject all
            </button>
          </div>
        </section>
      ))}
    </div>
  );
}

function GroupCard({ g, doc, busy, locked, act, compact }: {
  g: SuggestionGroup; doc: SchemaDoc; busy: string | null; locked?: string; compact?: boolean;
  act: (key: string, a: ReviewAction, body: Parameters<typeof reviewSuggestion>[1], what: string) => void;
}) {
  const terms = useMemo(() => walk(doc.schema.taxonomies[g.taxonomy]?.terms ?? {}), [doc, g.taxonomy]);
  const groups = terms.filter((t) => t.group).map((t) => t.name);
  const leaves = terms.filter((t) => !t.group).map((t) => t.name);
  const [name, setName] = useState(g.term);
  const [desc, setDesc] = useState(g.description);
  const [parent, setParent] = useState(g.parent ?? "");
  const [into, setInto] = useState(g.nearest ?? leaves[0] ?? "");
  const [open, setOpen] = useState(!compact);
  const key = `${g.taxonomy}:${g.term}`;
  const names = [g.term, ...g.variants];
  const off = !!busy || !!locked;

  return (
    <div className={clsx("rounded-md border p-3", g.ready ? "border-local/40" : "border-line opacity-90")}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded border border-line px-1.5 py-px font-mono text-[10.5px] text-muted">{g.taxonomy}</span>
        <input value={name} onChange={(e) => setName(e.target.value)} aria-label="term name"
          className="min-w-[12rem] rounded border border-line bg-bg px-2 py-0.5 font-mono text-sm text-ink focus:border-local/60 focus:outline-none" />
        <span className="font-mono text-[11px] text-muted">{g.records} record{g.records === 1 ? "" : "s"}{g.users.length ? ` · ${g.users.join(", ")}` : ""}</span>
        {g.variants.length > 0 && <span className="font-mono text-[11px] text-muted">also proposed as {g.variants.join(", ")}</span>}
        <button className="ml-auto text-[11px] text-muted hover:text-ink" onClick={() => setOpen(!open)}>{open ? "hide" : "evidence"}</button>
      </div>
      {open && (
        <>
          <input value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="description the classifier will see"
            className="mt-2 w-full rounded border border-line bg-bg px-2 py-1 text-xs text-ink focus:border-local/60 focus:outline-none" />
          <ul className="mt-2 flex flex-col gap-0.5 text-xs">
            {g.evidence.map((e) => (
              <li key={`${e.dataset}:${e.doc}`} className="flex flex-wrap gap-x-2">
                <span className="text-ink">{e.title}</span>
                <span className="font-mono text-[11px] text-muted">{e.author ?? ""}</span>
                <span className="font-mono text-[11px] text-muted">assigned meanwhile: {e.picked.filter(Boolean).join(", ") || "—"}</span>
              </li>
            ))}
          </ul>
          {g.same_as ? (
            <p className="mt-1 text-[11px] text-warn">
              The grouping step reads this as the existing term <span className="font-mono">{g.same_as}</span>: merging is likely the right call.
            </p>
          ) : (
            <p className="mt-1 font-mono text-[11px] text-muted">
              closest existing term: {g.nearest ? `${g.nearest} (${Math.round(g.nearest_score * 100)}% shared words)` : "none"}
            </p>
          )}
        </>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        {groups.length > 0 && (
          <label className="flex items-center gap-1 text-muted">under
            <select value={parent} onChange={(e) => setParent(e.target.value)} className="rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink">
              <option value="">(top level)</option>
              {groups.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        )}
        <button className="btn border-good/50" disabled={off || !name.trim()} title={locked ?? "Add as a new term"}
          onClick={() => act(key, "approve", { taxonomy: g.taxonomy, term: name, description: desc, parent: parent || null,
                                               names: names.filter((n) => n !== name), base_version: doc.version }, `Added ${name} to ${g.taxonomy}.`)}>
          <Check size={12} /> {busy === key ? "Saving…" : "Approve"}
        </button>
        <span className="flex items-center gap-1">
          <button className="btn" disabled={off || !into} title={locked ?? `Make ${names.join(", ")} aliases of ${into}`}
            onClick={() => act(key, "merge", { taxonomy: g.taxonomy, names, into, base_version: doc.version }, `Merged into ${into}.`)}>
            <GitMerge size={12} /> Merge into
          </button>
          <select value={into} onChange={(e) => setInto(e.target.value)} className="rounded border border-line bg-bg px-1 py-0.5 font-mono text-[11px] text-ink">
            {leaves.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </span>
        <button className="btn hover:border-bad/50" disabled={off} title={locked ?? "Never suggest these names again"}
          onClick={() => act(key, "reject", { taxonomy: g.taxonomy, names, base_version: doc.version }, `Rejected ${g.term}.`)}>
          <X size={12} /> Reject
        </button>
      </div>
    </div>
  );
}
