import { Button, Card, Chip } from "@heroui/react";
import { Fragment, type ReactNode, useEffect, useId, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  useDeletePricing,
  useDiscoverableModels,
  useModelMetadata,
  useModels,
  usePricing,
  useProviders,
  useSetPricing,
  useSettings,
} from "@/api/hooks";
import type { ModelMetadata, ProviderCapabilities, ProviderInfo } from "@/api/types";
import { ModelComboBox } from "@/components/ModelComboBox";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ConfirmButton, ErrorBanner, errorMessage, FilterSelect, InfoBanner, PageHeader } from "@/components/ui";
import { formatContext, formatCost, formatNumber, formatReleaseDate } from "@/lib/format";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";

// `owned_by` the gateway stamps on a configured alias (ALIAS_OWNED_BY in
// src/gateway/api/routes/models.py). Aliases are display names declared in
// config.yml, not models, so they cannot be priced here.
const ALIAS_OWNED_BY = "otari";

// Where a row's price comes from. "configured" is a DB price and the only kind
// editable here; "default" is the genai-prices fallback; "alias" is inherited
// from a target; "none" means metered at no cost.
type PriceSource = "configured" | "default" | "alias" | "none";

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

// Capability filter options. These test the model's own metadata (models.dev),
// so they line up with the per-model Features chips shown in the table, not the
// coarser provider-level capabilities. Picking "Vision" narrows to models whose
// metadata actually reports image input, matching what the chips display.
const MODEL_FILTER_CAPABILITIES: { value: string; label: string; test: (metadata: ModelMetadata) => boolean }[] = [
  { value: "vision", label: "Vision", test: (m) => m.input_modalities.includes("image") },
  { value: "tool_call", label: "Tool calling", test: (m) => Boolean(m.tool_call) },
  { value: "reasoning", label: "Reasoning", test: (m) => Boolean(m.reasoning) },
  { value: "structured_output", label: "Structured output", test: (m) => Boolean(m.structured_output) },
  { value: "attachment", label: "Attachments", test: (m) => Boolean(m.attachment) },
  { value: "audio", label: "Audio", test: (m) => m.input_modalities.includes("audio") },
  { value: "pdf", label: "PDF", test: (m) => m.input_modalities.includes("pdf") },
];

// Per-model capability flags from models.dev, labelled for the detail panel.
const MODEL_CAPABILITY_LABELS: { key: keyof ModelMetadata; label: string }[] = [
  { key: "reasoning", label: "Reasoning" },
  { key: "tool_call", label: "Tool calling" },
  { key: "structured_output", label: "Structured output" },
  { key: "attachment", label: "Attachments" },
  { key: "temperature", label: "Temperature" },
];

const MODALITY_LABELS: Record<string, string> = {
  text: "Text",
  image: "Image",
  audio: "Audio",
  video: "Video",
  pdf: "PDF",
};

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

// Newness windows for the release-date filter, in days back from today. Rows
// with no known release date are excluded once a window is active.
const RELEASE_OPTIONS = [
  { value: "all", label: "Any release date" },
  { value: "365", label: "Past year" },
  { value: "730", label: "Past 2 years" },
  { value: "1095", label: "Past 3 years" },
];
const RELEASE_WINDOW_MS = 24 * 60 * 60 * 1000;

interface ModelRow {
  key: string;
  model: string;
  provider: string;
  isDiscovered?: boolean;
  contextWindow: number | null;
  releaseDate?: string | null;
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
  if (source === "default" || source === "alias") {
    return (
      <Chip size="sm" color="accent">
        {source}
      </Chip>
    );
  }
  return <span className="text-xs text-[var(--otari-muted)]">not priced</span>;
}

// Prices a model the picker lists, or one it cannot see (an unconfigured
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
    setPricing.mutate(
      {
        model_key: modelKey.trim(),
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
      },
      { onSuccess: onClose },
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
      <div className="flex gap-2">
        <Button variant="primary" isDisabled={!canSubmit || setPricing.isPending} onPress={submit}>
          {setPricing.isPending ? "Saving…" : "Save price"}
        </Button>
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

// A small info affordance: an "i" bubble that reveals a tooltip on hover or
// keyboard focus. Hand-rolled to match the dashboard's other bare primitives
// (HeroUI v3 ships no tooltip we use here).
function InfoTooltip({
  label,
  tone = "info",
  children,
}: {
  label: string;
  tone?: "info" | "warning";
  children: ReactNode;
}) {
  // Unique per instance: several tooltips render at once, so a shared id would
  // make aria-describedby resolve to the wrong (or ambiguous) tooltip.
  const tipId = useId();
  return (
    <span className="group relative inline-flex items-center font-normal normal-case">
      <button
        type="button"
        aria-label={label}
        aria-describedby={tipId}
        className={`inline-flex h-4 w-4 items-center justify-center rounded-full border text-[10px] leading-none ${
          tone === "warning"
            ? "border-[#c2843a] text-[#b45309]"
            : "border-[var(--otari-line)] text-[var(--otari-muted)] hover:border-[var(--otari-brand)] hover:text-[var(--otari-brand)]"
        }`}
      >
        i
      </button>
      <span
        id={tipId}
        role="tooltip"
        className="pointer-events-none absolute top-full right-0 z-20 mt-1.5 w-72 rounded-lg border border-[var(--otari-line)] bg-[var(--otari-surface)] px-3 py-2 text-left text-xs font-normal whitespace-normal break-words text-[var(--otari-ink)] opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {children}
      </span>
    </span>
  );
}

// The default-pricing explanation, surfaced from the pricing column headers
// instead of a persistent banner. Tone shifts to a warning when metering is off.
function PricingInfo() {
  const settings = useSettings();
  if (!settings.data) {
    return null;
  }
  if (settings.data.default_pricing) {
    return (
      <InfoTooltip label="How unpriced models are metered" tone="info">
        Default pricing is on: models without a configured price are metered using community-maintained rates (the
        bundled genai-prices dataset). Set a price to override the fallback.
      </InfoTooltip>
    );
  }
  return (
    <InfoTooltip label="How unpriced models are metered" tone="warning">
      Default pricing is off: only models with a configured price are metered.
      {settings.data.require_pricing
        ? " Requests for any other model are rejected (HTTP 402) because require_pricing is on."
        : " Other models are served without cost tracking."}
    </InfoTooltip>
  );
}

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

// Small labelled key/value used throughout the detail panel.
function Spec({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-xs text-[var(--otari-muted)]">{label}</span>
      <span className="text-right text-sm text-[var(--otari-ink)] tabular-nums">{value}</span>
    </div>
  );
}

function PanelSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-semibold uppercase tracking-wide text-[var(--otari-muted)]">{title}</span>
      {children}
    </div>
  );
}

// Price editor inside the detail panel: the selected model's rate, plus
// set/edit/clear. Reset per model via a `key` on the element that mounts it.
function PanelPriceEditor({ row }: { row: ModelRow }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const [editing, setEditing] = useState(false);
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");

  const startEdit = () => {
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
      { onSuccess: () => setEditing(false) },
    );
  };

  if (editing) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-[var(--otari-muted)]">Input $ / 1M</span>
          <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.key}`} />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-[var(--otari-muted)]">Output $ / 1M</span>
          <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.key}`} />
        </div>
        <div className="flex items-center gap-2">
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
        </div>
        {setPricing.error ? <span className="text-xs text-red-700">{errorMessage(setPricing.error)}</span> : null}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <Spec label="Input" value={row.inputPrice == null ? "—" : `${formatCost(row.inputPrice)} / 1M`} />
      <Spec label="Output" value={row.outputPrice == null ? "—" : `${formatCost(row.outputPrice)} / 1M`} />
      <div className="flex items-center gap-2 pt-1">
        <Button size="sm" variant="outline" onPress={startEdit}>
          {row.source === "configured" ? "Edit price" : "Set price"}
        </Button>
        {row.source === "configured" ? (
          <>
            <ConfirmButton
              confirmLabel="Reset"
              isPending={deletePricing.isPending}
              onConfirm={() => deletePricing.mutate(row.key)}
            >
              Reset
            </ConfirmButton>
            <InfoTooltip label="What reset does">
              Removes the custom price. The model reverts to the default rate (genai-prices) when default pricing is on,
              otherwise it is metered at no cost.
            </InfoTooltip>
          </>
        ) : null}
        {deletePricing.error ? <span className="text-xs text-red-700">{errorMessage(deletePricing.error)}</span> : null}
      </div>
    </div>
  );
}

function ProviderBlock({
  provider,
  info,
  modelCount,
  discoveryError,
}: {
  provider: string;
  info: ProviderInfo | undefined;
  modelCount: number;
  discoveryError: string | null;
}) {
  return (
    <PanelSection title="Provider">
      <div className="flex flex-col gap-3 rounded-lg border border-[var(--otari-line)] p-3">
        <div>
          <div className="text-sm font-medium text-[var(--otari-ink)]">{info?.name ?? provider}</div>
          <div className="text-xs text-[var(--otari-muted)]">
            {info && info.provider_type !== provider ? `${provider} · ${info.provider_type}` : provider}
          </div>
        </div>
        {discoveryError ? (
          <span className="text-xs text-red-700">Could not list models: {discoveryError}</span>
        ) : (
          <span className="text-xs text-[var(--otari-muted)]">
            {formatNumber(modelCount)} model{modelCount === 1 ? "" : "s"} reported
          </span>
        )}
        {info ? <CapabilityBadges capabilities={info.capabilities} /> : null}
        {info?.env_key ? (
          <code className="w-fit rounded bg-[var(--otari-bg)] px-2 py-0.5 text-xs text-[var(--otari-ink)]">
            {info.env_key}
          </code>
        ) : null}
        {info && (info.doc_url || info.pricing_urls.length > 0) ? (
          <div className="flex flex-col gap-1">
            {info.doc_url ? (
              <a
                href={info.doc_url}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-[var(--otari-brand)] underline underline-offset-2"
              >
                API documentation
              </a>
            ) : null}
            {info.pricing_urls.slice(0, 2).map((url) => (
              <a
                key={url}
                href={url}
                target="_blank"
                rel="noreferrer"
                className="text-xs break-all text-[var(--otari-brand)] underline underline-offset-2"
              >
                Pricing page
              </a>
            ))}
          </div>
        ) : null}
      </div>
    </PanelSection>
  );
}

// The persistent detail panel beside the table. Shows the selected model's
// pricing, specs, modalities, capabilities, and its provider; lets you page
// through the current list; and offers to alias the model.
function ModelDetailPanel({
  row,
  metadata,
  metadataAvailable,
  providerInfo,
  providerModelCount,
  providerDiscoveryError,
  onMakeAlias,
}: {
  row: ModelRow | null;
  metadata: ModelMetadata | undefined;
  metadataAvailable: boolean;
  providerInfo: ProviderInfo | undefined;
  providerModelCount: number;
  providerDiscoveryError: string | null;
  onMakeAlias: (key: string) => void;
}) {
  if (!row) {
    return (
      <Card>
        <Card.Content className="p-5 text-sm text-[var(--otari-muted)]">
          Select a model to see its pricing, context window, modalities, and capabilities.
        </Card.Content>
      </Card>
    );
  }

  const inputModalities = metadata?.input_modalities ?? [];
  const outputModalities = metadata?.output_modalities ?? [];
  const activeCaps = MODEL_CAPABILITY_LABELS.filter(({ key }) => metadata?.[key]);

  return (
    <Card>
      <Card.Content className="flex flex-col gap-5 p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold break-all text-[var(--otari-ink)]">{row.model}</h2>
              {metadata?.deprecated ? (
                <Chip size="sm" color="danger">
                  deprecated
                </Chip>
              ) : null}
            </div>
            {metadata?.family ? <p className="text-xs text-[var(--otari-muted)]">{metadata.family}</p> : null}
          </div>
        </div>

        {metadata?.description ? <p className="text-sm text-[var(--otari-ink)]">{metadata.description}</p> : null}

        <PanelSection title="Pricing">
          <div className="flex items-center gap-2">
            <SourceChip source={row.source} />
            {row.isDiscovered ? null : <span className="text-xs text-[var(--otari-muted)]">not discovered</span>}
          </div>
          <PanelPriceEditor key={row.key} row={row} />
          <Button size="sm" variant="outline" onPress={() => onMakeAlias(row.key)}>
            Make an alias
          </Button>
        </PanelSection>

        <PanelSection title="Specs">
          <Spec label="Context window" value={formatContext(row.contextWindow)} />
          <Spec label="Max output" value={formatContext(metadata?.max_output_tokens ?? null)} />
          <Spec label="Knowledge cutoff" value={metadata?.knowledge_cutoff ?? "—"} />
          <Spec label="Released" value={formatReleaseDate(metadata?.release_date)} />
          <Spec label="Open weights" value={metadata ? (metadata.open_weights ? "Yes" : "No") : "—"} />
        </PanelSection>

        <PanelSection title="Modalities">
          {inputModalities.length === 0 && outputModalities.length === 0 ? (
            <span className="text-sm text-[var(--otari-muted)]">Unknown.</span>
          ) : (
            <div className="flex flex-col gap-1.5 text-xs text-[var(--otari-muted)]">
              <div className="flex flex-wrap items-center gap-1">
                <span>In:</span>
                {inputModalities.map((m) => (
                  <Chip key={m} size="sm" color="default">
                    {MODALITY_LABELS[m] ?? m}
                  </Chip>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-1">
                <span>Out:</span>
                {outputModalities.map((m) => (
                  <Chip key={m} size="sm" color="default">
                    {MODALITY_LABELS[m] ?? m}
                  </Chip>
                ))}
              </div>
            </div>
          )}
        </PanelSection>

        <PanelSection title="Capabilities">
          {activeCaps.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {activeCaps.map(({ key, label }) => (
                <Chip key={key} size="sm" color="default">
                  {label}
                </Chip>
              ))}
            </div>
          ) : (
            <span className="text-sm text-[var(--otari-muted)]">
              {metadataAvailable
                ? "None reported."
                : "Extended metadata unavailable (models.dev disabled or unreachable)."}
            </span>
          )}
        </PanelSection>

        {/* Skip the provider pane for a custom / self-hosted instance (its name is
            not a stock provider): the bundled capabilities and doc links describe
            the underlying implementation, not the operator's endpoint. */}
        {providerInfo && providerInfo.provider_type === row.provider ? (
          <ProviderBlock
            provider={row.provider}
            info={providerInfo}
            modelCount={providerModelCount}
            discoveryError={providerDiscoveryError}
          />
        ) : null}
      </Card.Content>
    </Card>
  );
}

const PAGE_SIZE = 25;

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

type SortCol = "model" | "context" | "released" | "input" | "output";
type SortDir = "asc" | "desc";
type Sort = { col: SortCol; dir: SortDir };

// Newest models first by default; the choice is remembered across refreshes.
const SORT_STORAGE_KEY = "otari.dashboard.modelsSort";
const DEFAULT_SORT: Sort = { col: "model", dir: "asc" };
const SORT_COLS: SortCol[] = ["model", "context", "released", "input", "output"];

function readStoredSort(): Sort {
  if (typeof window === "undefined") return DEFAULT_SORT;
  try {
    const raw = window.localStorage.getItem(SORT_STORAGE_KEY);
    if (!raw) return DEFAULT_SORT;
    const parsed = JSON.parse(raw) as { col?: unknown; dir?: unknown };
    if (SORT_COLS.includes(parsed.col as SortCol) && (parsed.dir === "asc" || parsed.dir === "desc")) {
      return { col: parsed.col as SortCol, dir: parsed.dir };
    }
  } catch {
    // Ignore malformed storage and fall back to the default.
  }
  return DEFAULT_SORT;
}

function SortableTh({
  label,
  col,
  sort,
  onSort,
  align = "left",
  info,
}: {
  label: string;
  col: SortCol;
  sort: { col: SortCol; dir: SortDir };
  onSort: (col: SortCol) => void;
  align?: "left" | "right";
  info?: ReactNode;
}) {
  const active = sort.col === col;
  return (
    <Th
      className={align === "right" ? "text-right" : undefined}
      ariaSort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <span className={`inline-flex items-center gap-1.5 ${align === "right" ? "flex-row-reverse" : ""}`}>
        <button
          type="button"
          onClick={() => onSort(col)}
          className={`inline-flex items-center gap-1 ${align === "right" ? "flex-row-reverse" : ""} hover:text-[var(--otari-ink)]`}
        >
          {label}
          <span className="text-[10px] text-[var(--otari-muted)]">
            {active ? (sort.dir === "asc" ? "▲" : "▼") : "↕"}
          </span>
        </button>
        {info}
      </span>
    </Th>
  );
}

// Compact inline price editor shown in a sub-row under a model, so you can
// price it without leaving the list.
function InlinePriceForm({ row, onClose }: { row: ModelRow; onClose: () => void }) {
  const setPricing = useSetPricing();
  const deletePricing = useDeletePricing();
  const [input, setInput] = useState(row.inputPrice == null ? "" : String(row.inputPrice));
  const [output, setOutput] = useState(row.outputPrice == null ? "" : String(row.outputPrice));

  const save = () => {
    if (!isValidPrice(input) || !isValidPrice(output)) return;
    setPricing.mutate(
      { model_key: row.key, input_price_per_million: Number(input), output_price_per_million: Number(output) },
      { onSuccess: onClose },
    );
  };

  return (
    <div className="flex flex-wrap items-center gap-3 px-4 py-3">
      <span className="text-xs font-medium break-all text-[var(--otari-muted)]">{row.key}</span>
      <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
        Input $ / 1M
        <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.key}`} />
      </label>
      <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
        Output $ / 1M
        <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.key}`} />
      </label>
      <Button
        size="sm"
        variant="primary"
        isDisabled={setPricing.isPending || !isValidPrice(input) || !isValidPrice(output)}
        onPress={save}
      >
        {setPricing.isPending ? "Saving…" : "Save"}
      </Button>
      <Button size="sm" variant="ghost" isDisabled={setPricing.isPending} onPress={onClose}>
        Cancel
      </Button>
      {row.source === "configured" ? (
        <span className="inline-flex items-center gap-1">
          <ConfirmButton
            confirmLabel="Reset"
            isPending={deletePricing.isPending}
            onConfirm={() => deletePricing.mutate(row.key, { onSuccess: onClose })}
          >
            Reset
          </ConfirmButton>
          <InfoTooltip label="What reset does">
            Removes the custom price. The model reverts to the default rate (genai-prices) when default pricing is on,
            otherwise it is metered at no cost.
          </InfoTooltip>
        </span>
      ) : null}
      {setPricing.error || deletePricing.error ? (
        <span className="text-xs text-red-700">{errorMessage(setPricing.error ?? deletePricing.error)}</span>
      ) : null}
    </div>
  );
}

function ModelTable({
  rows,
  isLoading,
  empty,
  sort,
  onSort,
  selectedKey,
  onSelect,
  pricingKey,
  onSetPricingKey,
}: {
  rows: ModelRow[];
  isLoading: boolean;
  empty: ReactNode;
  sort: { col: SortCol; dir: SortDir };
  onSort: (col: SortCol) => void;
  selectedKey: string | null;
  onSelect: (key: string) => void;
  pricingKey: string | null;
  onSetPricingKey: (key: string | null) => void;
}) {
  return (
    <Table>
      <THead>
        <Tr>
          <SortableTh label="Model" col="model" sort={sort} onSort={onSort} />
          <Th>Provider</Th>
          <SortableTh label="Context" col="context" sort={sort} onSort={onSort} align="right" />
          <SortableTh
            label="Input $ / 1M"
            col="input"
            sort={sort}
            onSort={onSort}
            align="right"
            info={<PricingInfo />}
          />
          <SortableTh label="Output $ / 1M" col="output" sort={sort} onSort={onSort} align="right" />
          <Th className="text-right">Price</Th>
        </Tr>
      </THead>
      <tbody>
        {isLoading ? (
          <LoadingRow colSpan={6} />
        ) : rows.length > 0 ? (
          rows.map((row) => (
            <Fragment key={row.key}>
              <Tr onClick={() => onSelect(row.key)} selected={row.key === selectedKey}>
                <Td className="font-medium break-all">{row.model}</Td>
                <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
                <Td className="text-right tabular-nums text-[var(--otari-muted)]">
                  {formatContext(row.contextWindow)}
                </Td>
                <Td className="text-right tabular-nums">
                  {row.inputPrice == null ? "—" : formatCost(row.inputPrice)}
                </Td>
                <Td className="text-right tabular-nums">
                  {row.outputPrice == null ? "—" : formatCost(row.outputPrice)}
                </Td>
                <Td className="text-right">
                  <button
                    type="button"
                    className="text-xs font-medium text-[var(--otari-brand-dark)] hover:underline"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSetPricingKey(pricingKey === row.key ? null : row.key);
                    }}
                  >
                    {row.source === "configured" ? "Edit" : "Set price"}
                  </button>
                </Td>
              </Tr>
              {pricingKey === row.key ? (
                <tr className="border-b border-[var(--otari-line)] bg-[var(--otari-bg)] last:border-b-0">
                  <td colSpan={6}>
                    <InlinePriceForm row={row} onClose={() => onSetPricingKey(null)} />
                  </td>
                </tr>
              ) : null}
            </Fragment>
          ))
        ) : (
          <TableMessage colSpan={6}>{empty}</TableMessage>
        )}
      </tbody>
    </Table>
  );
}

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
  const navigate = useNavigate();
  const models = useModels();
  const pricing = usePricing();
  const discoverable = useDiscoverableModels();
  const providers = useProviders();
  const metadata = useModelMetadata();

  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [showPriceForm, setShowPriceForm] = useState(false);
  const [pricingKey, setPricingKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [sort, setSort] = useState<Sort>(readStoredSort);

  useEffect(() => {
    try {
      window.localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort));
    } catch {
      // Ignore storage errors (e.g. disabled localStorage); sort still applies.
    }
  }, [sort]);

  const [providerFilter, setProviderFilter] = useState("all");
  const [pricingFilter, setPricingFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [capabilityFilter, setCapabilityFilter] = useState("all");
  const [minContext, setMinContext] = useState("0");
  const [maxInput, setMaxInput] = useState("");
  const [releaseFilter, setReleaseFilter] = useState("all");

  const metadataByKey = metadata.data?.models ?? {};
  const metadataAvailable = metadata.data?.available ?? false;

  const discoverableKeys = useMemo(
    () => new Set((discoverable.data?.providers ?? []).flatMap((provider) => provider.models.map((model) => model.key))),
    [discoverable.data],
  );

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
      current.col === col
        ? { col, dir: current.dir === "asc" ? "desc" : "asc" }
        : { col, dir: col === "released" ? "desc" : "asc" },
    );
    setPage(0);
  };

  const rows = useMemo<ModelRow[]>(() => {
    const configured = new Map(currentPricing(pricing.data ?? []).map((row) => [row.model_key, row]));
    const result: ModelRow[] = [];
    const seen = new Set<string>();

    const add = (key: string, model: string, provider: string, catalogRow?: ModelRow) => {
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      const priced = configured.get(key);
      result.push({
        key,
        model,
        provider,
        isDiscovered: discoverableKeys.has(key),
        contextWindow: catalogRow?.contextWindow ?? null,
        inputPrice: priced ? priced.input_price_per_million : (catalogRow?.inputPrice ?? null),
        outputPrice: priced ? priced.output_price_per_million : (catalogRow?.outputPrice ?? null),
        source: priced ? "configured" : (catalogRow?.source ?? "none"),
      });
    };

    for (const model of models.data?.data ?? []) {
      // Aliases are managed on their own page, not part of the model catalogue.
      if (model.owned_by === ALIAS_OWNED_BY) {
        continue;
      }
      const priceStatus: PriceSource =
        model.pricing_source === "default" ? "default" : model.pricing ? "configured" : "none";
      add(model.id, model.id, model.owned_by || providerFromModelKey(model.id), {
        key: model.id,
        model: model.id,
        provider: model.owned_by,
        contextWindow: model.context_window,
        inputPrice: model.pricing?.input_price_per_million ?? null,
        outputPrice: model.pricing?.output_price_per_million ?? null,
        source: priceStatus,
      });
    }
    for (const key of configured.keys()) {
      add(key, key, providerFromModelKey(key));
    }

    return result;
  }, [models.data, pricing.data, discoverableKeys]);

  const rowsByKey = useMemo(() => new Map(rows.map((row) => [row.key, row])), [rows]);

  // The model list, with the context window filled from models.dev when the
  // catalog did not supply one. Aliases live on their own page.
  const modelRows = useMemo<ModelRow[]>(() => {
    const out = rows.map((row) => ({
        ...row,
        contextWindow: row.contextWindow ?? (metadataByKey[row.key]?.context_window ?? null),
        releaseDate: metadataByKey[row.key]?.release_date ?? null,
      }));
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
          contextWindow: metadataByKey[model.key]?.context_window ?? null,
          releaseDate: metadataByKey[model.key]?.release_date ?? null,
          inputPrice: null,
          outputPrice: null,
          source: "none",
        });
      }
    }
    return out;
  }, [rows, discoverable.data, metadataByKey]);

  const discoveredErrors = (discoverable.data?.providers ?? []).filter((provider) => !provider.ok);

  const providerOptions = useMemo(() => {
    const names = Array.from(new Set(modelRows.map((row) => row.provider))).sort((a, b) => a.localeCompare(b));
    return [{ value: "all", label: "All providers" }, ...names.map((name) => ({ value: name, label: name }))];
  }, [modelRows]);

  const query = search.trim().toLowerCase();
  const minContextValue = Number(minContext) || 0;
  const maxInputValue = maxInput === "" ? Number.POSITIVE_INFINITY : Number(maxInput);
  const releaseCutoff = releaseFilter === "all" ? null : Date.now() - Number(releaseFilter) * RELEASE_WINDOW_MS;

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
        const capability = MODEL_FILTER_CAPABILITIES.find((entry) => entry.value === capabilityFilter);
        const meta = metadataByKey[row.key];
        if (!capability || !meta || !capability.test(meta)) {
          return false;
        }
      }
      if (minContextValue > 0 && (row.contextWindow == null || row.contextWindow < minContextValue)) {
        return false;
      }
      if (maxInputValue !== Number.POSITIVE_INFINITY && (row.inputPrice == null || row.inputPrice > maxInputValue)) {
        return false;
      }
      if (releaseCutoff != null) {
        const released = row.releaseDate ? Date.parse(row.releaseDate) : Number.NaN;
        if (Number.isNaN(released) || released < releaseCutoff) {
          return false;
        }
      }
      return true;
    };

    const compare = (a: ModelRow, b: ModelRow): number => {
      const dir = sort.dir === "asc" ? 1 : -1;
      if (sort.col === "model") {
        return a.model.localeCompare(b.model) * dir;
      }
      if (sort.col === "released") {
        // models.dev dates are ISO (YYYY-MM-DD), so lexical order is chronological.
        const ad = a.releaseDate ?? null;
        const bd = b.releaseDate ?? null;
        if (!ad && !bd) {
          return a.model.localeCompare(b.model);
        }
        if (!ad) {
          return 1;
        }
        if (!bd) {
          return -1;
        }
        return (ad < bd ? -1 : ad > bd ? 1 : 0) * dir || a.model.localeCompare(b.model);
      }
      const pick = (row: ModelRow) =>
        sort.col === "context" ? row.contextWindow : sort.col === "input" ? row.inputPrice : row.outputPrice;
      const av = pick(a);
      const bv = pick(b);
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
    releaseCutoff,
    metadataByKey,
    sort,
  ]);

  const total = filteredModels.length;
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * PAGE_SIZE;
  const pageModels = filteredModels.slice(start, start + PAGE_SIZE);

  const modelsLoading = models.isLoading || pricing.isLoading || discoverable.isLoading;
  const hasActiveFilters =
    query !== "" ||
    providerFilter !== "all" ||
    pricingFilter !== "all" ||
    sourceFilter !== "all" ||
    capabilityFilter !== "all" ||
    minContext !== "0" ||
    maxInput !== "" ||
    releaseFilter !== "all";
  const emptyModels = hasActiveFilters
    ? "No models match your filters."
    : "No models yet. Add a provider on the Providers page, or price a model below.";

  // The row selected in the table, resolved against the filtered list (falling
  // back to any known row) to fill the detail panel.
  const selectedRow = selectedKey
    ? (filteredModels.find((row) => row.key === selectedKey) ?? rowsByKey.get(selectedKey) ?? null)
    : null;
  const selectedProvider = selectedRow?.provider;
  const selectedProviderInfo = selectedProvider
    ? providers.data?.providers.find((info) => info.instance === selectedProvider)
    : undefined;
  const selectedProviderDiscovery = selectedProvider
    ? (discoverable.data?.providers ?? []).find((provider) => provider.provider === selectedProvider)
    : undefined;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="Every model your providers can serve. Set a price on any model so budgets and usage tracking work."
      />

      <ErrorBanner
        error={models.error ?? pricing.error ?? discoverable.error ?? providers.error ?? metadata.error}
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
          <div className="flex min-w-0 flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <SearchInput value={search} onChange={changeSearch} placeholder="Search models…" />
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
                  ...MODEL_FILTER_CAPABILITIES.map((entry) => ({ value: entry.value, label: entry.label })),
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
              <FilterSelect
                ariaLabel="Filter by release date"
                value={releaseFilter}
                onChange={changeFilter(setReleaseFilter)}
                options={RELEASE_OPTIONS}
              />
            </div>

            <DiscoveredErrors providers={discoveredErrors} />

            <ModelTable
              rows={pageModels}
              isLoading={modelsLoading}
              empty={emptyModels}
              sort={sort}
              onSort={onSort}
              selectedKey={selectedKey}
              onSelect={setSelectedKey}
              pricingKey={pricingKey}
              onSetPricingKey={setPricingKey}
            />

            <div className="flex flex-wrap items-center justify-between gap-2">
              <Pagination page={clampedPage} pageCount={pageCount} total={total} onPage={setPage} />
              <button
                type="button"
                className="text-xs font-medium text-[var(--otari-brand-dark)]"
                onClick={() => setShowPriceForm((open) => !open)}
              >
                {showPriceForm ? "Hide" : "+ Price a model not listed here"}
              </button>
            </div>

            {showPriceForm ? (
              <Card>
                <Card.Content className="flex flex-col gap-4 p-5">
                  <div className="text-sm font-semibold text-[var(--otari-ink)]">Price a model not listed here</div>
                  <PriceModelForm onClose={() => setShowPriceForm(false)} />
                </Card.Content>
              </Card>
            ) : null}
          </div>

          <aside className="lg:sticky lg:top-4">
            <ModelDetailPanel
              row={selectedRow}
              metadata={selectedRow ? metadataByKey[selectedRow.key] : undefined}
              metadataAvailable={metadataAvailable}
              providerInfo={selectedProviderInfo}
              providerModelCount={selectedProviderDiscovery?.models.length ?? 0}
              providerDiscoveryError={
                selectedProviderDiscovery && !selectedProviderDiscovery.ok ? selectedProviderDiscovery.error : null
              }
              onMakeAlias={(key) => navigate(`/aliases?target=${encodeURIComponent(key)}`)}
            />
          </aside>
        </div>
    </div>
  );
}
