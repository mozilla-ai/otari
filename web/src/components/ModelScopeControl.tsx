import { Button } from "@heroui/react";
import { useId, useMemo, useState } from "react";

import { useDiscoverableModels, useProviders } from "@/api/hooks";

// The per-key model access-list is a tri-state:
//   null  -> "any"   (unrestricted, the default)
//   []    -> "block" (deny all)
//   list  -> "only"  (restrict to these entries)
// A bare multi-select cannot tell "any" from "block" (both look empty), so the
// mode is an explicit 3-way choice. "Only selected" with no entries is an
// INCOMPLETE form (Save disabled), never a silent deny-all.
type Mode = "any" | "only" | "block";

function modeOf(value: string[] | null): Mode {
  if (value === null) return "any";
  if (value.length === 0) return "block";
  return "only";
}

// Client-side echo of the server's grammar so a bad entry never becomes a chip.
// The server is the source of truth (it also rejects alias names and unknown
// providers); this catches the common mistakes inline.
function entryError(raw: string): string | null {
  const entry = raw.trim();
  if (!entry) return "Enter a model.";
  const colon = entry.indexOf(":");
  if (colon <= 0 || colon === entry.length - 1) {
    return "Use instance:model, e.g. openai:gpt-4o, openai:*, or openai:gpt-4*";
  }
  const model = entry.slice(colon + 1);
  if (model.includes("*") && !/^[^*]*\*$/.test(model)) {
    return "Only a single trailing * is allowed (openai:* or openai:gpt-4*).";
  }
  return null;
}

export function ModelScopeControl({
  initial,
  onChange,
}: {
  initial: string[] | null;
  onChange: (value: string[] | null, valid: boolean) => void;
}) {
  const providers = useProviders();
  const discoverable = useDiscoverableModels();
  const [mode, setMode] = useState<Mode>(modeOf(initial));
  const [entries, setEntries] = useState<string[]>(initial ?? []);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const listId = useId();

  // Suggestions: a wildcard per configured provider, then every discoverable
  // model key (already canonical instance:model). Flat, not grouped.
  const suggestions = useMemo(() => {
    const set = new Set<string>();
    for (const p of providers.data?.providers ?? []) set.add(`${p.instance}:*`);
    for (const prov of discoverable.data?.providers ?? []) {
      for (const m of prov.models) set.add(m.key);
    }
    return [...set].sort();
  }, [providers.data, discoverable.data]);

  const emit = (nextMode: Mode, nextEntries: string[]) => {
    if (nextMode === "any") onChange(null, true);
    else if (nextMode === "block") onChange([], true);
    else onChange(nextEntries, nextEntries.length > 0);
  };

  const chooseMode = (next: Mode) => {
    setMode(next);
    setError(null);
    emit(next, entries);
  };

  const addDraft = () => {
    const err = entryError(draft);
    if (err) {
      setError(err);
      return;
    }
    const entry = draft.trim();
    const next = entries.includes(entry) ? entries : [...entries, entry];
    setEntries(next);
    setDraft("");
    setError(null);
    emit("only", next);
  };

  const removeEntry = (entry: string) => {
    const next = entries.filter((e) => e !== entry);
    setEntries(next);
    emit("only", next);
  };

  const modeButton = (value: Mode, label: string) => (
    <button
      type="button"
      aria-pressed={mode === value}
      onClick={() => chooseMode(value)}
      className={
        mode === value
          ? "rounded-md bg-white px-3 py-1.5 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
          : "rounded-md px-3 py-1.5 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
      }
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-col gap-3">
      <div>
        <span className="text-sm font-medium text-[var(--otari-ink)]">Model access</span>
        <p className="text-xs text-[var(--otari-muted)]">
          Which models this key may list and call. The master key is never restricted, so blocking a key cannot lock
          you out of the dashboard.
        </p>
      </div>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
        {modeButton("any", "Any model")}
        {modeButton("only", "Only selected")}
        {modeButton("block", "Block all")}
      </div>

      {mode === "block" ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          This key is refused for <strong>every</strong> model until you change its access.
        </div>
      ) : null}

      {mode === "only" ? (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-1.5">
            {entries.length === 0 ? (
              <span className="text-xs text-[var(--otari-muted)]">
                Add at least one model, or choose “Block all”.
              </span>
            ) : (
              entries.map((entry) => (
                <span
                  key={entry}
                  className="inline-flex items-center gap-1 rounded-full bg-[var(--otari-brand-tint)] px-2.5 py-1 font-mono text-xs text-[var(--otari-brand-dark)]"
                >
                  {entry}
                  <button
                    type="button"
                    aria-label={`Remove ${entry}`}
                    onClick={() => removeEntry(entry)}
                    className="text-[var(--otari-brand-dark)] hover:text-red-700"
                  >
                    ×
                  </button>
                </span>
              ))
            )}
          </div>
          <div className="flex items-start gap-2">
            <div className="flex flex-1 flex-col gap-1">
              <input
                aria-label="Add a model"
                list={listId}
                value={draft}
                placeholder="openai:gpt-4o or openai:*"
                onChange={(e) => {
                  setDraft(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addDraft();
                  }
                }}
                autoComplete="off"
                className="w-full rounded-lg border border-[var(--otari-line)] bg-[var(--otari-surface)] px-3 py-2 font-mono text-xs text-[var(--otari-ink)]"
              />
              <datalist id={listId}>
                {suggestions.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
              {error ? <span className="text-xs text-red-700">{error}</span> : null}
            </div>
            <Button size="sm" variant="outline" onPress={addDraft}>
              Add
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

// A compact chip describing a key's access, for the table row.
export function accessLabel(allowed: string[] | null): { text: string; tone: "muted" | "normal" | "danger" } {
  if (allowed === null) return { text: "All models", tone: "muted" };
  if (allowed.length === 0) return { text: "Blocked", tone: "danger" };
  return { text: `${allowed.length} model${allowed.length === 1 ? "" : "s"}`, tone: "normal" };
}
