import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react";
import { type ReactNode, useMemo, useState } from "react";

import { useAliases, useDiscoverableModels, useProviders } from "@/api/hooks";

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

interface CatalogOption {
  // The canonical value stored on the allow-list (instance:model or instance:*).
  id: string;
  // What the operator sees in the dropdown.
  label: string;
}

const MAX_VISIBLE = 50;

// The control is reused for two layers of the same allow-list grammar: a user's
// default and a key's (narrower) override. The wording differs between them, so
// the heading, help text, and the "any" mode label are parameterized. "any" means
// null on the wire: unrestricted for a user, but "inherit the owner's default" for
// a key (a key with no list of its own falls back to its user).
export function ModelScopeControl({
  initial,
  onChange,
  title = "Model access",
  description,
  anyLabel = "Any model",
}: {
  initial: string[] | null;
  onChange: (value: string[] | null, valid: boolean) => void;
  title?: ReactNode;
  description?: ReactNode;
  anyLabel?: string;
}) {
  const providers = useProviders();
  const discoverable = useDiscoverableModels();
  const aliases = useAliases();
  const [mode, setMode] = useState<Mode>(modeOf(initial));
  const [entries, setEntries] = useState<string[]>(initial ?? []);
  const [query, setQuery] = useState("");

  // The pickable catalog: a wildcard per configured provider, every discoverable
  // model (already canonical instance:model), and each alias resolved to its
  // target. Deduped by value, so an alias whose target is already discoverable
  // does not appear twice. This is a pick-from-list control, not free text: every
  // stored entry is a real, canonical selector the backend will accept.
  const catalog = useMemo<CatalogOption[]>(() => {
    const seen = new Set<string>();
    const options: CatalogOption[] = [];
    const add = (id: string, label: string) => {
      if (id && !seen.has(id)) {
        seen.add(id);
        options.push({ id, label });
      }
    };
    for (const p of providers.data?.providers ?? []) {
      add(`${p.instance}:*`, `${p.instance}:*  ·  all ${p.instance} models`);
    }
    for (const prov of discoverable.data?.providers ?? []) {
      for (const m of prov.models) add(m.key, m.key);
    }
    for (const a of aliases.data ?? []) add(a.target, `${a.name}  ·  alias`);
    return options;
  }, [providers.data, discoverable.data, aliases.data]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return catalog
      .filter((o) => !entries.includes(o.id))
      .filter((o) => !q || o.id.toLowerCase().includes(q) || o.label.toLowerCase().includes(q))
      .slice(0, MAX_VISIBLE);
  }, [catalog, entries, query]);

  const emit = (nextMode: Mode, nextEntries: string[]) => {
    if (nextMode === "any") onChange(null, true);
    else if (nextMode === "block") onChange([], true);
    else onChange(nextEntries, nextEntries.length > 0);
  };

  const chooseMode = (next: Mode) => {
    setMode(next);
    emit(next, entries);
  };

  const addEntry = (id: string) => {
    const next = entries.includes(id) ? entries : [...entries, id];
    setEntries(next);
    setQuery("");
    emit("only", next);
  };

  const removeEntry = (id: string) => {
    const next = entries.filter((e) => e !== id);
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

  const catalogEmpty = !discoverable.isLoading && !providers.isLoading && catalog.length === 0;

  return (
    <div className="flex flex-col gap-3">
      <div>
        <span className="text-sm font-medium text-[var(--otari-ink)]">{title}</span>
        <p className="text-xs text-[var(--otari-muted)]">
          {description ??
            "Which models this key may list and call. The master key is never restricted, so blocking a key cannot lock you out of the dashboard."}
        </p>
      </div>
      <div className="flex w-fit items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
        {modeButton("any", anyLabel)}
        {modeButton("only", "Only selected")}
        {modeButton("block", "Block all")}
      </div>

      {mode === "block" ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Blocked from <strong>every</strong> model until you change this access.
        </div>
      ) : null}

      {mode === "only" ? (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-1.5">
            {entries.length === 0 ? (
              <span className="text-xs text-[var(--otari-muted)]">
                Pick at least one model below, or choose “Block all”.
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
          {catalogEmpty ? (
            <span className="text-xs text-[var(--otari-muted)]">
              No providers or models discovered yet. Configure a provider first, then scope this key.
            </span>
          ) : (
            <ComboBox.Root
              allowsEmptyCollection
              menuTrigger="input"
              inputValue={query}
              onInputChange={setQuery}
              selectedKey={null}
              onSelectionChange={(key) => {
                if (key != null) addEntry(String(key));
              }}
              className="flex flex-col gap-1"
            >
              <ComboBox.InputGroup>
                <Input aria-label="Add a model" placeholder="Search providers, models, aliases…" autoComplete="off" />
                <ComboBox.Trigger />
              </ComboBox.InputGroup>
              <ComboBox.Popover>
                <ListBox items={visible} className="max-h-72 overflow-auto">
                  {(option: CatalogOption) => (
                    <ListBoxItem id={option.id} textValue={option.label}>
                      {option.label}
                    </ListBoxItem>
                  )}
                </ListBox>
              </ComboBox.Popover>
            </ComboBox.Root>
          )}
        </div>
      ) : null}
    </div>
  );
}

// A compact label describing a key's access, for the table row. Deliberately not
// a count: an entry like `openai:*` is one entry but many models, so a number
// would mislead. The exact entries are surfaced on hover / in the edit form.
export function accessLabel(allowed: string[] | null): { text: string; tone: "muted" | "normal" | "danger" } {
  if (allowed === null) return { text: "All models", tone: "muted" };
  if (allowed.length === 0) return { text: "No models", tone: "danger" };
  return { text: "Selected models", tone: "normal" };
}
