"use client";

import { useMemo, useState, useTransition } from "react";
import type { Dive } from "@/lib/dives";
import {
  clearCalibrationSourceAction,
  setCalibrationSourceAction,
} from "./actions";

function diveLabel(dive: Dive): string {
  return `${dive.name?.trim() || `Dive ${dive.id}`} #${dive.id}`;
}

function diveDate(dive: Dive): string {
  if (!dive.dive_datetime) return "—";
  const date = new Date(dive.dive_datetime);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toISOString().slice(0, 10);
}

type RowProps = {
  dive: Dive;
  sources: Dive[];
  sourceLabel: (id: number) => string;
};

function CalibrationRow({ dive, sources, sourceLabel }: RowProps) {
  const persisted = dive.calibration_dive_id;
  const [selected, setSelected] = useState<number | "">(persisted ?? "");
  const [pending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const dirty = (selected === "" ? null : selected) !== persisted;

  function save() {
    setError(null);
    if (selected === "") return;
    const sourceId = selected;
    startTransition(async () => {
      const result = await setCalibrationSourceAction(dive.id, sourceId);
      if (!result.ok) setError(result.error);
    });
  }

  function clear() {
    setError(null);
    setSelected("");
    startTransition(async () => {
      const result = await clearCalibrationSourceAction(dive.id);
      if (!result.ok) setError(result.error);
    });
  }

  return (
    <tr className="border-t border-slate-200 dark:border-slate-800">
      <td className="py-2 pr-4 align-top">
        <div className="font-medium">{diveLabel(dive)}</div>
        <div className="text-xs text-slate-500">{diveDate(dive)}</div>
      </td>
      <td className="py-2 pr-4 align-top text-center">
        {dive.dive_slate_id != null ? (
          <span title="Has its own slate — self-calibrates">✓</span>
        ) : (
          <span className="text-slate-400">—</span>
        )}
      </td>
      <td className="py-2 pr-4 align-top">
        <select
          value={selected}
          disabled={pending}
          onChange={(event) =>
            setSelected(event.target.value === "" ? "" : Number(event.target.value))
          }
          className="w-full max-w-xs rounded-md border border-slate-300 bg-white px-2 py-1 text-sm shadow-sm disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900"
        >
          <option value="">— none (self-calibrate) —</option>
          {sources
            .filter((source) => source.id !== dive.id)
            .map((source) => (
              <option key={source.id} value={source.id}>
                {sourceLabel(source.id)}
              </option>
            ))}
        </select>
        {error ? (
          <div className="mt-1 text-xs text-red-600 dark:text-red-400">{error}</div>
        ) : null}
      </td>
      <td className="py-2 align-top">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={save}
            disabled={pending || !dirty || selected === ""}
            className="rounded-md border border-slate-300 bg-white px-3 py-1 text-sm font-medium shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
          >
            {pending ? "Saving…" : "Save"}
          </button>
          <button
            type="button"
            onClick={clear}
            disabled={pending || persisted == null}
            className="rounded-md border border-slate-300 bg-white px-3 py-1 text-sm font-medium shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
          >
            Clear
          </button>
        </div>
      </td>
    </tr>
  );
}

export function CalibrationLinks({ dives }: { dives: Dive[] }) {
  const [query, setQuery] = useState("");
  const [onlyNeedingLink, setOnlyNeedingLink] = useState(true);

  // Sources are slate-bearing dives — the ones stage 13 can calibrate and so
  // the only ones worth borrowing from.
  const sources = useMemo(
    () =>
      dives
        .filter((dive) => dive.dive_slate_id != null)
        .sort((a, b) => a.id - b.id),
    [dives],
  );

  const sourceLabelById = useMemo(() => {
    const map = new Map<number, string>();
    for (const dive of dives) {
      map.set(dive.id, `${diveLabel(dive)} — ${diveDate(dive)}`);
    }
    return map;
  }, [dives]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return dives
      .filter((dive) => {
        if (onlyNeedingLink && dive.dive_slate_id != null) return false;
        if (!needle) return true;
        return (
          diveLabel(dive).toLowerCase().includes(needle) ||
          String(dive.id).includes(needle)
        );
      })
      .sort((a, b) => {
        // Newest first when we have dates; fall back to id.
        const at = a.dive_datetime ? Date.parse(a.dive_datetime) : 0;
        const bt = b.dive_datetime ? Date.parse(b.dive_datetime) : 0;
        return bt - at || b.id - a.id;
      });
  }, [dives, query, onlyNeedingLink]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search dives by name or id…"
          className="w-64 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm shadow-sm dark:border-slate-700 dark:bg-slate-900"
        />
        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
          <input
            type="checkbox"
            checked={onlyNeedingLink}
            onChange={(event) => setOnlyNeedingLink(event.target.checked)}
          />
          Only dives without their own slate
        </label>
        <span className="text-xs text-slate-500">
          {rows.length} dive{rows.length === 1 ? "" : "s"} · {sources.length} slate
          dive{sources.length === 1 ? "" : "s"} available as sources
        </span>
      </div>

      {sources.length === 0 ? (
        <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
          No slate-bearing dives found to borrow calibration from.
        </p>
      ) : null}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="pb-2 pr-4 font-medium">Dive</th>
              <th className="pb-2 pr-4 text-center font-medium">Own slate</th>
              <th className="pb-2 pr-4 font-medium">Calibration source</th>
              <th className="pb-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((dive) => (
              <CalibrationRow
                key={dive.id}
                dive={dive}
                sources={sources}
                sourceLabel={(id) => sourceLabelById.get(id) ?? `Dive ${id}`}
              />
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={4} className="py-6 text-center text-slate-500">
                  No dives match.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
