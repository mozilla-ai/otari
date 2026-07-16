import { Button, Card, Chip } from "@heroui/react";
import { type ReactNode, useMemo, useState } from "react";

import {
  useAliases,
  useCreateAlias,
  useDeleteAlias,
  useDeletePricing,
  useDiscoverableModels,
  useModels,
  usePricing,
  useSetPricing,
  useSettings,
} from "@/api/hooks";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, InfoBanner, PageHeader } from "@/components/ui";
import { formatCost, formatNumber } from "@/lib/format";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";

// `owned_by` the gateway stamps on a configured alias (ALIAS_OWNED_BY in
// src/gateway/api/routes/models.py). Aliases are display names declared in
// config.yml, not models, so they cannot be priced here: pricing, budgets, and
// usage all key on the resolved target, and the API rejects a price posted
// against an alias name.
const ALIAS_OWNED_BY = "otari";

// Where a row's price comes from. "configured" is a price in the database and is
// the only kind that can be edited or cleared; "default" is the bundled
// genai-prices fallback; "alias" is inherited from the alias's target; "none"
// means the model is metered at no cost.
type PriceSource = "configured" | "default" | "alias" | "none";

interface ModelRow {
  key: string;
  model: string;
  provider: string;
  // Set only for alias rows. A config.yml alias lives in a file this UI cannot
  // edit, so it is shown but not deletable.
  aliasSource?: "config" | "stored";
  // For an alias row, where its target's price comes from. The row's own
  // ``source`` is "alias", which would otherwise hide whether the alias resolves
  // to a configured or default-priced model.
  aliasPriceSource?: PriceSource;
  // True when a provider currently reports this model via discovery. Drives the
  // Discovered tab regardless of whether the model is also priced.
  isDiscovered?: boolean;
  inputPrice: number | null;
  outputPrice: number | null;
  source: PriceSource;
}

function isValidPrice(value: string): boolean {
  const parsed = Number(value);
  return value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0;
}

function MoneyInput({
  value,
  onChange,
  ariaLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  ariaLabel: string;
}) {
  return (
    <input
      type="number"
      step="any"
      min="0"
      inputMode="decimal"
      aria-label={ariaLabel}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-28 rounded-md border border-[var(--otari-line)] bg-white px-2 py-1 text-right text-sm tabular-nums focus:border-[var(--otari-brand)] focus:outline-none"
    />
  );
}

function SourceChip({ source }: { source: PriceSource }) {
  if (source === "configured") {
    return (
      <Chip size="sm" color="default">
        configured
      </Chip>
    );
  }
  if (source === "default") {
    return (
      <Chip size="sm" color="accent">
        default
      </Chip>
    );
  }
  if (source === "alias") {
    return (
      <Chip size="sm" color="accent">
        alias
      </Chip>
    );
  }
  return <span className="text-xs text-[var(--otari-muted)]">not priced</span>;
}

function ModelTableRow({ row }: { row: ModelRow }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const deleteAlias = useDeleteAlias();
  const [editing, setEditing] = useState(false);
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const startEdit = () => {
    // Seed from whatever rate is in effect, so overriding a default price is a
    // tweak rather than re-typing it.
    setInput(row.inputPrice == null ? "" : String(row.inputPrice));
    setOutput(row.outputPrice == null ? "" : String(row.outputPrice));
    setEditing(true);
  };

  const save = () => {
    if (!isValidPrice(input) || !isValidPrice(output)) {
      return;
    }
    setPricing.mutate(
      {
        model_key: row.key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      {
        onSuccess: () => {
          setEditing(false);
        },
      },
    );
  };

  return (
    <Tr>
      <Td className="font-medium break-all">{row.model}</Td>
      <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
      <Td>
        <SourceChip source={row.source} />
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.key}`} />
        ) : row.inputPrice == null ? (
          "—"
        ) : (
          formatCost(row.inputPrice)
        )}
      </Td>
      <Td className="text-right">
        {editing ? (
          <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.key}`} />
        ) : row.outputPrice == null ? (
          "—"
        ) : (
          formatCost(row.outputPrice)
        )}
      </Td>
      <Td className="text-right whitespace-nowrap">
        {row.source === "alias" ? (
          // An alias is priced through its target, and the API rejects a price
          // posted against the alias name, so pricing is never offered here. A
          // stored alias can still be removed; one from config.yml cannot,
          // because this UI does not own that file.
          row.aliasSource === "stored" ? (
            <span className="inline-flex items-center gap-2">
              <ConfirmButton
                confirmLabel="Delete"
                isPending={deleteAlias.isPending}
                onConfirm={() => deleteAlias.mutate(row.key)}
              >
                Delete
              </ConfirmButton>
              {deleteAlias.error ? <span className="text-xs text-red-700">{errorMessage(deleteAlias.error)}</span> : null}
            </span>
          ) : (
            <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
          )
        ) : editing ? (
          <span className="inline-flex items-center gap-2">
            <Button
              size="sm"
              variant="primary"
              isDisabled={setPricing.isPending || !isValidPrice(input) || !isValidPrice(output)}
              onPress={save}
            >
              Save
            </Button>
            <Button size="sm" variant="ghost" isDisabled={setPricing.isPending} onPress={() => setEditing(false)}>
              Cancel
            </Button>
          </span>
        ) : (
          <span className="inline-flex items-center gap-2">
            <Button size="sm" variant="outline" onPress={startEdit}>
              {row.source === "configured" ? "Edit" : "Set price"}
            </Button>
            {row.source === "configured" ? (
              <ConfirmButton
                confirmLabel="Clear"
                isPending={deletePricing.isPending}
                onConfirm={() => deletePricing.mutate(row.key)}
              >
                Clear
              </ConfirmButton>
            ) : null}
            {setPricing.error || deletePricing.error ? (
              <span className="text-xs text-red-700">{errorMessage(setPricing.error ?? deletePricing.error)}</span>
            ) : null}
          </span>
        )}
      </Td>
    </Tr>
  );
}

// Prices a model: one discovery lists, or one it cannot see (an unconfigured
// provider, a brand-new release), which is why the key stays free text.
function PriceModelForm({ onClose }: { onClose: () => void }) {
  const setPricing = useSetPricing();
  const [modelKey, setModelKey] = useState("");
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const canSubmit = modelKey.trim() !== "" && isValidPrice(input) && isValidPrice(output);

  const submit = () => {
    if (!canSubmit) {
      return;
    }
    const key = modelKey.trim();
    setPricing.mutate(
      {
        model_key: key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      {
        onSuccess: () => {
          onClose();
        },
      },
    );
  };

  return (
    <div className="flex flex-col gap-4">
      <ErrorBanner error={setPricing.error} />
      <div className="grid gap-4 sm:grid-cols-3">
        <ModelComboBox
          label="Model"
          value={modelKey}
          onChange={setModelKey}
          isRequired
          autoFocus
          description="Pick one your providers report, or type any provider:model key."
        />
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-[var(--otari-ink)]">Input $ / 1M</span>
          <MoneyInput value={input} onChange={setInput} ariaLabel="Input price per million" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-[var(--otari-ink)]">Output $ / 1M</span>
          <MoneyInput value={output} onChange={setOutput} ariaLabel="Output price per million" />
        </div>
      </div>
      <div>
        <Button variant="primary" isDisabled={!canSubmit || setPricing.isPending} onPress={submit}>
          {setPricing.isPending ? "Saving…" : "Save price"}
        </Button>
      </div>
    </div>
  );
}

// Creates a stored alias: a display name callers send as `model`, resolved to a
// real target. Pricing, budgets, and usage all key on the target, so an alias is
// never priced here.
function AddAliasForm({ onClose }: { onClose: () => void }) {
  const createAlias = useCreateAlias();
  const [name, setName] = useState("");
  const [target, setTarget] = useState("");

  // The gateway rejects a name carrying a selector delimiter, since it could
  // never be told apart from a real provider:model. Say so before the round trip.
  const nameHasDelimiter = /[:/]/.test(name);
  const canSubmit = name.trim() !== "" && target.trim() !== "" && !nameHasDelimiter;

  const submit = () => {
    if (!canSubmit) {
      return;
    }
    createAlias.mutate({ name: name.trim(), target: target.trim() }, { onSuccess: onClose });
  };

  return (
    <div className="flex flex-col gap-4">
      <ErrorBanner error={createAlias.error} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Alias name"
          value={name}
          onChange={setName}
          placeholder="fast-model"
          isRequired
          autoFocus
          description={
            nameHasDelimiter ? (
              <span className="text-red-700">
                An alias name cannot contain “:” or “/”, which would make it look like a provider:model key.
              </span>
            ) : (
              "What callers send as `model`."
            )
          }
        />
        <ModelComboBox
          label="Target"
          value={target}
          onChange={setTarget}
          isRequired
          description="The real model this resolves to. Callers never see it."
        />
      </div>
      <div>
        <Button variant="primary" isDisabled={!canSubmit || createAlias.isPending} onPress={submit}>
          {createAlias.isPending ? "Creating…" : "Create alias"}
        </Button>
      </div>
    </div>
  );
}

type AddTab = "model" | "alias";

function AddForm({
  onClose,
  initialMode = "model",
}: {
  onClose: () => void;
  initialMode?: AddTab;
}) {
  const [tab, setTab] = useState<AddTab>(initialMode);

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 rounded-lg bg-[var(--otari-bg)] p-1">
            {(
              [
                ["model", "Price a model"],
                ["alias", "Add an alias"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={tab === id}
                onClick={() => setTab(id)}
                className={
                  tab === id
                    ? "rounded-md bg-white px-3 py-1.5 text-sm font-medium text-[var(--otari-ink)] shadow-sm"
                    : "rounded-md px-3 py-1.5 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
                }
              >
                {label}
              </button>
            ))}
          </div>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>
        {tab === "model" ? <PriceModelForm onClose={onClose} /> : <AddAliasForm onClose={onClose} />}
      </Card.Content>
    </Card>
  );
}

function DefaultPricingBanner() {
  const settings = useSettings();
  if (!settings.data) {
    return null;
  }
  if (settings.data.default_pricing) {
    return (
      <InfoBanner tone="info">
        Default pricing is <strong>on</strong>: models without a configured price are metered using
        community-maintained rates (the bundled genai-prices dataset). Set a price to override the fallback.
      </InfoBanner>
    );
  }
  return (
    <InfoBanner tone="warning">
      Default pricing is <strong>off</strong>: only models with a configured price are metered.
      {settings.data.require_pricing
        ? " Requests for any other model are rejected (HTTP 402) because require_pricing is on."
        : " Other models are served without cost tracking."}
    </InfoBanner>
  );
}

const TAB_ITEMS = [
  { id: "discovered", label: "Discovered" },
  { id: "priced", label: "Priced" },
  { id: "aliases", label: "Aliases" },
] as const;

type Tab = (typeof TAB_ITEMS)[number]["id"];

// One page of rows. The Discovered tab alone can run past a hundred models, so
// every tab paginates at the same size rather than rendering an unbounded list.
const PAGE_SIZE = 25;

function TabBar({ tab, counts, onSelect }: { tab: Tab; counts: Record<Tab, number>; onSelect: (tab: Tab) => void }) {
  return (
    <div role="tablist" aria-label="Model categories" className="flex items-center gap-1 border-b border-[var(--otari-line)]">
      {TAB_ITEMS.map((item) => {
        const active = item.id === tab;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onSelect(item.id)}
            className={
              active
                ? "-mb-px border-b-2 border-[var(--otari-brand)] px-3 py-2 text-sm font-medium text-[var(--otari-ink)]"
                : "-mb-px border-b-2 border-transparent px-3 py-2 text-sm text-[var(--otari-muted)] hover:text-[var(--otari-ink)]"
            }
          >
            {item.label}
            <span className="ml-1.5 text-xs text-[var(--otari-muted)] tabular-nums">{counts[item.id]}</span>
          </button>
        );
      })}
    </div>
  );
}

function SearchInput({ value, onChange, placeholder }: { value: string; onChange: (value: string) => void; placeholder: string }) {
  return (
    <input
      type="search"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      aria-label={placeholder}
      className="w-full max-w-xs rounded-md border border-[var(--otari-line)] bg-white px-3 py-1.5 text-sm focus:border-[var(--otari-brand)] focus:outline-none"
    />
  );
}

function Pagination({ page, pageCount, total, onPage }: { page: number; pageCount: number; total: number; onPage: (page: number) => void }) {
  if (pageCount <= 1) {
    return null;
  }
  return (
    <div className="flex items-center justify-between px-1 pt-1 text-sm text-[var(--otari-muted)]">
      <span>
        Page {page + 1} of {pageCount} · {formatNumber(total)} model{total === 1 ? "" : "s"}
      </span>
      <span className="inline-flex gap-2">
        <Button size="sm" variant="outline" isDisabled={page === 0} onPress={() => onPage(page - 1)}>
          Prev
        </Button>
        <Button size="sm" variant="outline" isDisabled={page >= pageCount - 1} onPress={() => onPage(page + 1)}>
          Next
        </Button>
      </span>
    </div>
  );
}

// Header shared by the Discovered and Priced tabs; both carry the same columns,
// so one row renderer (ModelTableRow) serves them.
function ModelTable({
  rows,
  isLoading,
  empty,
}: {
  rows: ModelRow[];
  isLoading: boolean;
  empty: ReactNode;
}) {
  return (
    <Table>
      <THead>
        <Tr>
          <Th>Model</Th>
          <Th>Provider</Th>
          <Th>Price</Th>
          <Th className="text-right">Input $ / 1M</Th>
          <Th className="text-right">Output $ / 1M</Th>
          <Th className="text-right">Actions</Th>
        </Tr>
      </THead>
      <tbody>
        {isLoading ? (
          <LoadingRow colSpan={6} />
        ) : rows.length > 0 ? (
          rows.map((row) => <ModelTableRow key={row.key} row={row} />)
        ) : (
          <TableMessage colSpan={6}>{empty}</TableMessage>
        )}
      </tbody>
    </Table>
  );
}

interface AliasEntry {
  name: string;
  target: string;
  source: "config" | "stored";
  row?: ModelRow;
}

function AliasTableRow({ entry }: { entry: AliasEntry }) {
  const deleteAlias = useDeleteAlias();
  const row = entry.row;
  return (
    <Tr>
      <Td className="font-medium break-all">{entry.name}</Td>
      <Td className="break-all text-[var(--otari-muted)]">{entry.target}</Td>
      <Td>
        {row?.aliasPriceSource ? (
          <SourceChip source={row.aliasPriceSource} />
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">—</span>
        )}
      </Td>
      <Td className="text-right">{row?.inputPrice != null ? formatCost(row.inputPrice) : "—"}</Td>
      <Td className="text-right">{row?.outputPrice != null ? formatCost(row.outputPrice) : "—"}</Td>
      <Td className="text-right whitespace-nowrap">
        {entry.source === "stored" ? (
          <span className="inline-flex items-center gap-2">
            <ConfirmButton
              confirmLabel="Delete"
              isPending={deleteAlias.isPending}
              onConfirm={() => deleteAlias.mutate(entry.name)}
            >
              Delete
            </ConfirmButton>
            {deleteAlias.error ? <span className="text-xs text-red-700">{errorMessage(deleteAlias.error)}</span> : null}
          </span>
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">set in config.yml</span>
        )}
      </Td>
    </Tr>
  );
}

function AliasTable({ entries, isLoading }: { entries: AliasEntry[]; isLoading: boolean }) {
  return (
    <Table>
      <THead>
        <Tr>
          <Th>Alias</Th>
          <Th>Target</Th>
          <Th>Resolves to</Th>
          <Th className="text-right">Input $ / 1M</Th>
          <Th className="text-right">Output $ / 1M</Th>
          <Th className="text-right">Actions</Th>
        </Tr>
      </THead>
      <tbody>
        {isLoading ? (
          <LoadingRow colSpan={6} />
        ) : entries.length > 0 ? (
          entries.map((entry) => <AliasTableRow key={entry.name} entry={entry} />)
        ) : (
          <TableMessage colSpan={6}>No aliases yet. Use “Add” to create one.</TableMessage>
        )}
      </tbody>
    </Table>
  );
}

// The Discovered tab lists what the credentials can reach; a provider that could
// not be listed is called out so a short list is not mistaken for "nothing to
// offer". Mirrors the model picker's hint.
function DiscoveredErrors({ providers }: { providers: { provider: string; error: string | null }[] }) {
  if (providers.length === 0) {
    return null;
  }
  return (
    <InfoBanner tone="warning">
      Could not list {providers.map((provider) => provider.provider).join(", ")}. Check that provider's credentials in
      config.yml; its models are missing from the list below.
    </InfoBanner>
  );
}

export function ModelsPage() {
  const models = useModels();
  const pricing = usePricing();
  const aliases = useAliases();
  const discoverable = useDiscoverableModels();
  const [tab, setTab] = useState<Tab>("discovered");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [showForm, setShowForm] = useState(false);

  // Keys a provider currently reports, so a row can be tagged as discovered
  // regardless of whether it is also priced.
  const discoverableKeys = useMemo(
    () => new Set((discoverable.data?.providers ?? []).flatMap((provider) => provider.models.map((model) => model.key))),
    [discoverable.data],
  );

  // Changing tab or search resets to the first page; a page index left over from
  // a longer list would otherwise land past the end of a shorter one.
  const selectTab = (next: Tab) => {
    setTab(next);
    setSearch("");
    setPage(0);
  };
  const changeSearch = (value: string) => {
    setSearch(value);
    setPage(0);
  };

  const rows = useMemo<ModelRow[]>(() => {
    // A configured price wins over whatever the catalog reports, and is the only
    // thing this page can edit or clear.
    const configured = new Map(currentPricing(pricing.data ?? []).map((row) => [row.model_key, row]));
    const aliasSourceByName = new Map((aliases.data ?? []).map((alias) => [alias.name, alias.source]));
    const result: ModelRow[] = [];
    const seen = new Set<string>();

    const add = (key: string, model: string, provider: string, catalogRow?: ModelRow) => {
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      // An alias reports the price of whatever it resolves to. A pricing row
      // stored under the alias's own name is dead (nothing reads it), so it must
      // not be shown as this row's price.
      const isAlias = catalogRow?.source === "alias";
      const priced = isAlias ? undefined : configured.get(key);
      result.push({
        key,
        model,
        provider,
        aliasSource: isAlias ? aliasSourceByName.get(key) : undefined,
        aliasPriceSource: catalogRow?.aliasPriceSource,
        isDiscovered: discoverableKeys.has(key),
        inputPrice: priced ? priced.input_price_per_million : (catalogRow?.inputPrice ?? null),
        outputPrice: priced ? priced.output_price_per_million : (catalogRow?.outputPrice ?? null),
        source: priced ? "configured" : (catalogRow?.source ?? "none"),
      });
    };

    // The catalog first, so every configured/discovered/aliased model shows.
    for (const model of models.data?.data ?? []) {
      const isAlias = model.owned_by === ALIAS_OWNED_BY;
      // The genai-prices/DB pricing status of the model, computed the same way
      // for a real model (its own price) and an alias (its target's price, kept
      // as aliasPriceSource since the row's own source is "alias").
      const priceStatus: PriceSource =
        model.pricing_source === "default" ? "default" : model.pricing ? "configured" : "none";
      add(model.id, model.id, model.owned_by || providerFromModelKey(model.id), {
        key: model.id,
        model: model.id,
        provider: model.owned_by,
        aliasPriceSource: isAlias ? priceStatus : undefined,
        inputPrice: model.pricing?.input_price_per_million ?? null,
        outputPrice: model.pricing?.output_price_per_million ?? null,
        source: isAlias ? "alias" : priceStatus,
      });
    }
    // Then any configured price the catalog did not list (e.g. an alias target
    // the catalog withholds), so the Priced tab is complete.
    for (const key of configured.keys()) {
      add(key, key, providerFromModelKey(key));
    }

    return result.sort((a, b) => a.model.localeCompare(b.model));
  }, [models.data, pricing.data, aliases.data, discoverableKeys]);

  const rowsByKey = useMemo(() => new Map(rows.map((row) => [row.key, row])), [rows]);

  // Each tab is a lens over the same rows: a discovered model with a configured
  // price shows in both Discovered and Priced.
  const priced = useMemo(() => rows.filter((row) => row.source === "configured"), [rows]);
  const discovered = useMemo(() => {
    const seen = new Set<string>();
    const out: ModelRow[] = [];
    for (const provider of discoverable.data?.providers ?? []) {
      for (const model of provider.models) {
        if (seen.has(model.key)) {
          continue;
        }
        seen.add(model.key);
        // Use the enriched row (price) when the catalog produced one; otherwise a
        // bare discovered model the catalog did not enrich (model_discovery off).
        out.push(
          rowsByKey.get(model.key) ?? {
            key: model.key,
            // Full selector, matching how enriched rows label the Model column.
            model: model.key,
            provider: provider.provider,
            isDiscovered: true,
            inputPrice: null,
            outputPrice: null,
            source: "none",
          },
        );
      }
    }
    return out.sort((a, b) => a.key.localeCompare(b.key));
  }, [discoverable.data, rowsByKey]);
  const discoveredErrors = (discoverable.data?.providers ?? []).filter((provider) => !provider.ok);

  const aliasEntries = useMemo<AliasEntry[]>(
    () =>
      (aliases.data ?? [])
        .map((alias) => ({
          name: alias.name,
          target: alias.target,
          source: alias.source,
          row: rowsByKey.get(alias.name),
        }))
        .sort((a, b) => a.name.localeCompare(b.name)),
    [aliases.data, rowsByKey],
  );

  const counts: Record<Tab, number> = {
    discovered: discovered.length,
    priced: priced.length,
    aliases: aliasEntries.length,
  };

  const query = search.trim().toLowerCase();
  const modelMatches = (row: ModelRow) =>
    !query || row.key.toLowerCase().includes(query) || row.provider.toLowerCase().includes(query);
  const filteredModels = (tab === "priced" ? priced : discovered).filter(modelMatches);
  const filteredAliases = aliasEntries.filter(
    (entry) => !query || entry.name.toLowerCase().includes(query) || entry.target.toLowerCase().includes(query),
  );

  const total = tab === "aliases" ? filteredAliases.length : filteredModels.length;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * PAGE_SIZE;
  const pageModels = filteredModels.slice(start, start + PAGE_SIZE);
  const pageAliases = filteredAliases.slice(start, start + PAGE_SIZE);

  const rowsLoading = models.isLoading || pricing.isLoading;
  const emptyMessage: Record<Exclude<Tab, "aliases">, ReactNode> = {
    discovered: query
      ? "No discovered models match your search."
      : "No models discovered. Check provider credentials in config.yml, or add one under “Add”.",
    priced: query ? "No priced models match your search." : "No prices set. Use “Add” to price a model.",
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="Configure model pricing: set rates for discovered models, review what you have priced, and manage aliases."
        action={
          <Button variant="primary" onPress={() => setShowForm((open) => !open)}>
            {showForm ? "Hide form" : "Add"}
          </Button>
        }
      />

      <DefaultPricingBanner />

      {showForm ? (
        <AddForm onClose={() => setShowForm(false)} initialMode={tab === "aliases" ? "alias" : "model"} />
      ) : null}

      <ErrorBanner error={models.error ?? pricing.error ?? aliases.error} />

      <TabBar tab={tab} counts={counts} onSelect={selectTab} />

      <div className="flex flex-col gap-3">
        <SearchInput
          value={search}
          onChange={changeSearch}
          placeholder={tab === "aliases" ? "Search aliases…" : "Search models…"}
        />

        {tab === "discovered" ? <DiscoveredErrors providers={discoveredErrors} /> : null}

        {tab === "aliases" ? (
          <AliasTable entries={pageAliases} isLoading={aliases.isLoading} />
        ) : (
          <ModelTable
            rows={pageModels}
            isLoading={tab === "discovered" ? discoverable.isLoading : rowsLoading}
            empty={emptyMessage[tab]}
          />
        )}

        <Pagination page={clampedPage} pageCount={pageCount} total={total} onPage={setPage} />
      </div>
    </div>
  );
}
