import { Button, Card, Chip } from "@heroui/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";

import {
  useAliases,
  useCreateAlias,
  useDeleteAlias,
  useDeletePricing,
  useDiscoverableModels,
  useModels,
  usePricing,
  useProviders,
  useSetPricing,
  useSettings,
} from "@/api/hooks";
import type { ProviderCapabilities, ProviderInfo } from "@/api/types";
import { Field } from "@/components/Field";
import { ModelComboBox } from "@/components/ModelComboBox";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, InfoBanner, PageHeader } from "@/components/ui";
import { formatContext, formatCost, formatNumber } from "@/lib/format";
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

// Human labels for the provider capability flags the API returns. Order here is
// the order badges render in.
const CAPABILITY_LABELS: Record<keyof ProviderCapabilities, string> = {
  vision: "Vision",
  reasoning: "Reasoning",
  embeddings: "Embeddings",
  streaming: "Streaming",
  audio: "Audio",
  image_generation: "Image gen",
  pdf: "PDF",
  rerank: "Rerank",
  responses_api: "Responses API",
  moderation: "Moderation",
  list_models: "Lists models",
};

// Capabilities offered in the model filter. A narrower set than every flag: these
// are the ones an operator picks a model for. Capabilities are provider-level, so
// filtering by one keeps every model a matching provider serves.
const FILTERABLE_CAPABILITIES: (keyof ProviderCapabilities)[] = [
  "vision",
  "reasoning",
  "embeddings",
  "audio",
  "image_generation",
  "pdf",
  "rerank",
];

const CONTEXT_OPTIONS = [
  { value: "0", label: "Any context" },
  { value: "8000", label: "≥ 8K" },
  { value: "32000", label: "≥ 32K" },
  { value: "128000", label: "≥ 128K" },
  { value: "200000", label: "≥ 200K" },
  { value: "1000000", label: "≥ 1M" },
];

const PRICE_OPTIONS = [
  { value: "", label: "Any price" },
  { value: "1", label: "≤ $1 / 1M in" },
  { value: "3", label: "≤ $3 / 1M in" },
  { value: "10", label: "≤ $10 / 1M in" },
  { value: "30", label: "≤ $30 / 1M in" },
];

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
  // True when a provider currently reports this model via discovery, as opposed
  // to a model only present because it was priced or aliased by hand.
  isDiscovered?: boolean;
  // genai-prices context-window token limit, or null when unknown.
  contextWindow: number | null;
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

function ModelTableRow({ row, onProviderClick }: { row: ModelRow; onProviderClick: (provider: string) => void }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
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
      <Td>
        <button
          type="button"
          onClick={() => onProviderClick(row.provider)}
          className="text-[var(--otari-muted)] underline decoration-dotted underline-offset-2 hover:text-[var(--otari-ink)]"
        >
          {row.provider}
        </button>
      </Td>
      <Td className="text-right tabular-nums text-[var(--otari-muted)]">{formatContext(row.contextWindow)}</Td>
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
        {editing ? (
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

// A capability the provider exposes, one chip per true flag.
function CapabilityBadges({ capabilities }: { capabilities: ProviderCapabilities }) {
  const active = (Object.keys(CAPABILITY_LABELS) as (keyof ProviderCapabilities)[]).filter((key) => capabilities[key]);
  if (active.length === 0) {
    return <span className="text-sm text-[var(--otari-muted)]">No capabilities reported.</span>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {active.map((key) => (
        <Chip key={key} size="sm" color="default">
          {CAPABILITY_LABELS[key]}
        </Chip>
      ))}
    </div>
  );
}

// Provider detail: a right-side panel with the provider's capabilities, links,
// and how many models it currently reports. Built by hand (a fixed panel plus a
// backdrop) rather than with a component library so its dev-mode behavior is
// predictable and it stays consistent with the page's other hand-rolled UI.
function ProviderDrawer({
  provider,
  info,
  modelCount,
  discoveryError,
  onClose,
}: {
  provider: string;
  info: ProviderInfo | undefined;
  modelCount: number;
  discoveryError: string | null;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={`Provider ${provider}`}>
      <button type="button" aria-label="Close provider details" className="absolute inset-0 bg-black/30" onClick={onClose} />
      <div className="relative z-10 flex h-full w-full max-w-md flex-col gap-5 overflow-y-auto bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[var(--otari-ink)]">{info?.name ?? provider}</h2>
            <p className="text-sm text-[var(--otari-muted)]">
              {info && info.provider_type !== provider ? `${provider} · ${info.provider_type}` : provider}
            </p>
          </div>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>

        {info?.description ? <p className="text-sm text-[var(--otari-ink)]">{info.description}</p> : null}

        <div className="flex flex-col gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-[var(--otari-muted)]">Models</span>
          {discoveryError ? (
            <span className="text-sm text-red-700">Could not list models: {discoveryError}</span>
          ) : (
            <span className="text-sm text-[var(--otari-ink)]">
              {formatNumber(modelCount)} model{modelCount === 1 ? "" : "s"} reported
            </span>
          )}
        </div>

        {info ? (
          <div className="flex flex-col gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-[var(--otari-muted)]">Capabilities</span>
            <CapabilityBadges capabilities={info.capabilities} />
          </div>
        ) : (
          <span className="text-sm text-[var(--otari-muted)]">
            No bundled metadata for this provider. It is priced or aliased by hand, not one of the configured providers.
          </span>
        )}

        {info?.env_key ? (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-[var(--otari-muted)]">Credential</span>
            <code className="rounded bg-[var(--otari-bg)] px-2 py-1 text-sm text-[var(--otari-ink)]">{info.env_key}</code>
          </div>
        ) : null}

        {info && (info.doc_url || info.pricing_urls.length > 0) ? (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-[var(--otari-muted)]">Links</span>
            {info.doc_url ? (
              <a
                href={info.doc_url}
                target="_blank"
                rel="noreferrer"
                className="text-sm text-[var(--otari-brand)] underline underline-offset-2"
              >
                API documentation
              </a>
            ) : null}
            {info.pricing_urls.map((url) => (
              <a
                key={url}
                href={url}
                target="_blank"
                rel="noreferrer"
                className="text-sm break-all text-[var(--otari-brand)] underline underline-offset-2"
              >
                {url}
              </a>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

const TAB_ITEMS = [
  { id: "models", label: "Models" },
  { id: "aliases", label: "Aliases" },
] as const;

type Tab = (typeof TAB_ITEMS)[number]["id"];

// One page of rows. The model list alone can run past a hundred entries, so both
// tabs paginate at the same size rather than rendering an unbounded list.
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

function FilterSelect({
  ariaLabel,
  value,
  onChange,
  options,
}: {
  ariaLabel: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="rounded-md border border-[var(--otari-line)] bg-white px-2 py-1.5 text-sm text-[var(--otari-ink)] focus:border-[var(--otari-brand)] focus:outline-none"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
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

type SortCol = "model" | "context" | "input" | "output";
type SortDir = "asc" | "desc";

function SortableTh({
  label,
  col,
  sort,
  onSort,
  align = "left",
}: {
  label: string;
  col: SortCol;
  sort: { col: SortCol; dir: SortDir };
  onSort: (col: SortCol) => void;
  align?: "left" | "right";
}) {
  const active = sort.col === col;
  return (
    <Th className={align === "right" ? "text-right" : undefined}>
      <button
        type="button"
        onClick={() => onSort(col)}
        className={`inline-flex items-center gap-1 ${align === "right" ? "flex-row-reverse" : ""} hover:text-[var(--otari-ink)]`}
      >
        {label}
        <span className="text-[10px] text-[var(--otari-muted)]">{active ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}</span>
      </button>
    </Th>
  );
}

function ModelTable({
  rows,
  isLoading,
  empty,
  sort,
  onSort,
  onProviderClick,
}: {
  rows: ModelRow[];
  isLoading: boolean;
  empty: ReactNode;
  sort: { col: SortCol; dir: SortDir };
  onSort: (col: SortCol) => void;
  onProviderClick: (provider: string) => void;
}) {
  return (
    <Table>
      <THead>
        <Tr>
          <SortableTh label="Model" col="model" sort={sort} onSort={onSort} />
          <Th>Provider</Th>
          <SortableTh label="Context" col="context" sort={sort} onSort={onSort} align="right" />
          <Th>Price</Th>
          <SortableTh label="Input $ / 1M" col="input" sort={sort} onSort={onSort} align="right" />
          <SortableTh label="Output $ / 1M" col="output" sort={sort} onSort={onSort} align="right" />
          <Th className="text-right">Actions</Th>
        </Tr>
      </THead>
      <tbody>
        {isLoading ? (
          <LoadingRow colSpan={7} />
        ) : rows.length > 0 ? (
          rows.map((row) => <ModelTableRow key={row.key} row={row} onProviderClick={onProviderClick} />)
        ) : (
          <TableMessage colSpan={7}>{empty}</TableMessage>
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

// A provider that could not be listed is called out so a short list is not
// mistaken for "nothing to offer". Mirrors the model picker's hint.
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
  const providers = useProviders();

  const [tab, setTab] = useState<Tab>("models");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [showForm, setShowForm] = useState(false);
  const [openProvider, setOpenProvider] = useState<string | null>(null);
  const [sort, setSort] = useState<{ col: SortCol; dir: SortDir }>({ col: "model", dir: "asc" });

  // Filters (models tab only).
  const [providerFilter, setProviderFilter] = useState("all");
  const [pricingFilter, setPricingFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [capabilityFilter, setCapabilityFilter] = useState("all");
  const [minContext, setMinContext] = useState("0");
  const [maxInput, setMaxInput] = useState("");

  // Keys a provider currently reports, so a row can be tagged as discovered
  // regardless of whether it is also priced.
  const discoverableKeys = useMemo(
    () => new Set((discoverable.data?.providers ?? []).flatMap((provider) => provider.models.map((model) => model.key))),
    [discoverable.data],
  );

  // Capability flags per provider instance, for the capability filter and drawer.
  const providerCapabilities = useMemo(
    () => new Map((providers.data?.providers ?? []).map((info) => [info.instance, info.capabilities])),
    [providers.data],
  );

  const selectTab = (next: Tab) => {
    setTab(next);
    setSearch("");
    setPage(0);
  };
  const changeSearch = (value: string) => {
    setSearch(value);
    setPage(0);
  };
  const changeFilter = (setter: (value: string) => void) => (value: string) => {
    setter(value);
    setPage(0);
  };
  const onSort = (col: SortCol) => {
    setSort((current) =>
      current.col === col ? { col, dir: current.dir === "asc" ? "desc" : "asc" } : { col, dir: "asc" },
    );
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
        contextWindow: catalogRow?.contextWindow ?? null,
        inputPrice: priced ? priced.input_price_per_million : (catalogRow?.inputPrice ?? null),
        outputPrice: priced ? priced.output_price_per_million : (catalogRow?.outputPrice ?? null),
        source: priced ? "configured" : (catalogRow?.source ?? "none"),
      });
    };

    // The catalog first, so every configured/discovered/aliased model shows.
    for (const model of models.data?.data ?? []) {
      const isAlias = model.owned_by === ALIAS_OWNED_BY;
      const priceStatus: PriceSource =
        model.pricing_source === "default" ? "default" : model.pricing ? "configured" : "none";
      add(model.id, model.id, model.owned_by || providerFromModelKey(model.id), {
        key: model.id,
        model: model.id,
        provider: model.owned_by,
        aliasPriceSource: isAlias ? priceStatus : undefined,
        contextWindow: model.context_window,
        inputPrice: model.pricing?.input_price_per_million ?? null,
        outputPrice: model.pricing?.output_price_per_million ?? null,
        source: isAlias ? "alias" : priceStatus,
      });
    }
    // Then any configured price the catalog did not list (e.g. an alias target
    // the catalog withholds), so the priced set is complete.
    for (const key of configured.keys()) {
      add(key, key, providerFromModelKey(key));
    }

    return result;
  }, [models.data, pricing.data, aliases.data, discoverableKeys]);

  const rowsByKey = useMemo(() => new Map(rows.map((row) => [row.key, row])), [rows]);

  // The unified model list: every non-alias row, plus any model a provider
  // reports that the catalog did not enrich (model_discovery off). Aliases are
  // their own tab.
  const modelRows = useMemo<ModelRow[]>(() => {
    const out = rows.filter((row) => row.source !== "alias");
    const seen = new Set(out.map((row) => row.key));
    for (const provider of discoverable.data?.providers ?? []) {
      for (const model of provider.models) {
        if (seen.has(model.key)) {
          continue;
        }
        seen.add(model.key);
        out.push({
          key: model.key,
          model: model.key,
          provider: provider.provider,
          isDiscovered: true,
          contextWindow: null,
          inputPrice: null,
          outputPrice: null,
          source: "none",
        });
      }
    }
    return out;
  }, [rows, discoverable.data]);

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

  // Providers present in the model list, for the provider filter dropdown.
  const providerOptions = useMemo(() => {
    const names = Array.from(new Set(modelRows.map((row) => row.provider))).sort((a, b) => a.localeCompare(b));
    return [{ value: "all", label: "All providers" }, ...names.map((name) => ({ value: name, label: name }))];
  }, [modelRows]);

  const query = search.trim().toLowerCase();
  const minContextValue = Number(minContext) || 0;
  const maxInputValue = maxInput === "" ? Number.POSITIVE_INFINITY : Number(maxInput);

  const filteredModels = useMemo(() => {
    const matches = (row: ModelRow): boolean => {
      if (query && !row.key.toLowerCase().includes(query) && !row.provider.toLowerCase().includes(query)) {
        return false;
      }
      if (providerFilter !== "all" && row.provider !== providerFilter) {
        return false;
      }
      if (pricingFilter === "configured" && row.source !== "configured") {
        return false;
      }
      if (pricingFilter === "default" && row.source !== "default") {
        return false;
      }
      if (pricingFilter === "priced" && row.inputPrice == null) {
        return false;
      }
      if (pricingFilter === "unpriced" && row.inputPrice != null) {
        return false;
      }
      if (sourceFilter === "discovered" && !row.isDiscovered) {
        return false;
      }
      if (sourceFilter === "custom" && row.isDiscovered) {
        return false;
      }
      if (capabilityFilter !== "all") {
        const caps = providerCapabilities.get(row.provider);
        if (!caps || !caps[capabilityFilter as keyof ProviderCapabilities]) {
          return false;
        }
      }
      if (minContextValue > 0 && (row.contextWindow == null || row.contextWindow < minContextValue)) {
        return false;
      }
      if (maxInputValue !== Number.POSITIVE_INFINITY && (row.inputPrice == null || row.inputPrice > maxInputValue)) {
        return false;
      }
      return true;
    };

    const compare = (a: ModelRow, b: ModelRow): number => {
      const dir = sort.dir === "asc" ? 1 : -1;
      if (sort.col === "model") {
        return a.model.localeCompare(b.model) * dir;
      }
      const pick = (row: ModelRow) =>
        sort.col === "context" ? row.contextWindow : sort.col === "input" ? row.inputPrice : row.outputPrice;
      const av = pick(a);
      const bv = pick(b);
      // Unknown values sort last regardless of direction, so a column of numbers
      // is not interrupted by blanks.
      if (av == null && bv == null) {
        return a.model.localeCompare(b.model);
      }
      if (av == null) {
        return 1;
      }
      if (bv == null) {
        return -1;
      }
      return (av - bv) * dir || a.model.localeCompare(b.model);
    };

    return modelRows.filter(matches).sort(compare);
  }, [
    modelRows,
    query,
    providerFilter,
    pricingFilter,
    sourceFilter,
    capabilityFilter,
    minContextValue,
    maxInputValue,
    providerCapabilities,
    sort,
  ]);

  const filteredAliases = aliasEntries.filter(
    (entry) => !query || entry.name.toLowerCase().includes(query) || entry.target.toLowerCase().includes(query),
  );

  const counts: Record<Tab, number> = {
    models: modelRows.length,
    aliases: aliasEntries.length,
  };

  const total = tab === "aliases" ? filteredAliases.length : filteredModels.length;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * PAGE_SIZE;
  const pageModels = filteredModels.slice(start, start + PAGE_SIZE);
  const pageAliases = filteredAliases.slice(start, start + PAGE_SIZE);

  const modelsLoading = models.isLoading || pricing.isLoading || discoverable.isLoading;
  const emptyModels = query
    ? "No models match your filters."
    : "No models yet. Check provider credentials in config.yml, or use “Add” to price one.";

  const drawerProvider = openProvider;
  const drawerInfo = drawerProvider ? providers.data?.providers.find((info) => info.instance === drawerProvider) : undefined;
  const drawerDiscovery = drawerProvider
    ? (discoverable.data?.providers ?? []).find((provider) => provider.provider === drawerProvider)
    : undefined;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="See the models your providers can reach, set their pricing, and manage aliases."
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
        <div className="flex flex-wrap items-center gap-2">
          <SearchInput
            value={search}
            onChange={changeSearch}
            placeholder={tab === "aliases" ? "Search aliases…" : "Search models…"}
          />
          {tab === "models" ? (
            <>
              <FilterSelect
                ariaLabel="Filter by provider"
                value={providerFilter}
                onChange={changeFilter(setProviderFilter)}
                options={providerOptions}
              />
              <FilterSelect
                ariaLabel="Filter by pricing"
                value={pricingFilter}
                onChange={changeFilter(setPricingFilter)}
                options={[
                  { value: "all", label: "Any pricing" },
                  { value: "configured", label: "Custom price" },
                  { value: "default", label: "Default price" },
                  { value: "priced", label: "Priced" },
                  { value: "unpriced", label: "Unpriced" },
                ]}
              />
              <FilterSelect
                ariaLabel="Filter by source"
                value={sourceFilter}
                onChange={changeFilter(setSourceFilter)}
                options={[
                  { value: "all", label: "Any source" },
                  { value: "discovered", label: "Discovered" },
                  { value: "custom", label: "Custom (not discovered)" },
                ]}
              />
              <FilterSelect
                ariaLabel="Filter by capability"
                value={capabilityFilter}
                onChange={changeFilter(setCapabilityFilter)}
                options={[
                  { value: "all", label: "Any capability" },
                  ...FILTERABLE_CAPABILITIES.map((key) => ({ value: key, label: CAPABILITY_LABELS[key] })),
                ]}
              />
              <FilterSelect
                ariaLabel="Minimum context window"
                value={minContext}
                onChange={changeFilter(setMinContext)}
                options={CONTEXT_OPTIONS}
              />
              <FilterSelect
                ariaLabel="Maximum input price"
                value={maxInput}
                onChange={changeFilter(setMaxInput)}
                options={PRICE_OPTIONS}
              />
            </>
          ) : null}
        </div>

        {tab === "models" ? <DiscoveredErrors providers={discoveredErrors} /> : null}

        {tab === "aliases" ? (
          <AliasTable entries={pageAliases} isLoading={aliases.isLoading} />
        ) : (
          <ModelTable
            rows={pageModels}
            isLoading={modelsLoading}
            empty={emptyModels}
            sort={sort}
            onSort={onSort}
            onProviderClick={setOpenProvider}
          />
        )}

        <Pagination page={clampedPage} pageCount={pageCount} total={total} onPage={setPage} />
      </div>

      {drawerProvider ? (
        <ProviderDrawer
          provider={drawerProvider}
          info={drawerInfo}
          modelCount={drawerDiscovery?.models.length ?? 0}
          discoveryError={drawerDiscovery && !drawerDiscovery.ok ? drawerDiscovery.error : null}
          onClose={() => setOpenProvider(null)}
        />
      ) : null}
    </div>
  );
}
