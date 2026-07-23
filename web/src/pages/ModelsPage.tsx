import { Button, Card, Chip } from "@heroui/react";
import { Fragment, type ReactNode, useEffect, useId, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import {
  useDeletePricing,
  useDiscoverableModels,
  useModelMetadata,
  useModels,
  usePricing,
  useSetPricing,
  useSettings,
} from "@/api/hooks";
import type { ModelMetadata, PricingTier } from "@/api/types";
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

// Capability filter options. These test the model's own metadata (models.dev),
// so they line up with the per-model Features chips shown in the table, not the
// coarser provider-level capabilities. Picking "Vision" narrows to models whose
// metadata actually reports image input, matching what the chips display.
const MODEL_FILTER_CAPABILITIES: { value: string; label: string; test: (metadata: ModelMetadata) => boolean }[] = [
  { value: "vision", label: "Vision", test: (m) => Array.isArray(m.input_modalities) && m.input_modalities.includes("image") },
  { value: "tool_call", label: "Tool calling", test: (m) => Boolean(m.tool_call) },
  { value: "reasoning", label: "Reasoning", test: (m) => Boolean(m.reasoning) },
  { value: "structured_output", label: "Structured output", test: (m) => Boolean(m.structured_output) },
  { value: "attachment", label: "Attachments", test: (m) => Boolean(m.attachment) },
  { value: "audio", label: "Audio", test: (m) => Array.isArray(m.input_modalities) && m.input_modalities.includes("audio") },
  { value: "pdf", label: "PDF", test: (m) => Array.isArray(m.input_modalities) && m.input_modalities.includes("pdf") },
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

const PRICE_COMPARISON_OPTIONS = [
  { value: "", label: "Base prices" },
  { value: "8000", label: "Compare at 8K" },
  { value: "128000", label: "Compare at 128K" },
  { value: "200000", label: "Compare at 200K" },
  { value: "500000", label: "Compare at 500K" },
  { value: "1000000", label: "Compare at 1M" },
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
  cacheReadPrice: number | null;
  cacheWritePrice: number | null;
  cacheWrite1hPrice: number | null;
  pricingTiers: PricingTier[];
  source: PriceSource;
}

type EffectiveRates = Pick<
  ModelRow,
  "inputPrice" | "outputPrice" | "cacheReadPrice" | "cacheWritePrice" | "cacheWrite1hPrice"
>;

function displayModelName(selector: string, provider: string): string {
  const instancePrefix = `${provider}:`;
  return selector.startsWith(instancePrefix) ? selector.slice(instancePrefix.length) : selector;
}

function effectiveRatesAtContext(row: ModelRow, contextTokens: number | null): EffectiveRates {
  const rates: EffectiveRates = {
    inputPrice: row.inputPrice,
    outputPrice: row.outputPrice,
    cacheReadPrice: row.cacheReadPrice,
    cacheWritePrice: row.cacheWritePrice,
    cacheWrite1hPrice: row.cacheWrite1hPrice,
  };
  if (contextTokens == null) {
    return rates;
  }
  const tier = row.pricingTiers
    .filter((candidate) => candidate.min_input_tokens <= contextTokens)
    .sort((a, b) => b.min_input_tokens - a.min_input_tokens)[0];
  if (!tier) {
    return rates;
  }
  return {
    inputPrice: tier.input_price_per_million ?? rates.inputPrice,
    outputPrice: tier.output_price_per_million ?? rates.outputPrice,
    cacheReadPrice: tier.cache_read_price_per_million ?? rates.cacheReadPrice,
    cacheWritePrice: tier.cache_write_price_per_million ?? rates.cacheWritePrice,
    cacheWrite1hPrice: tier.cache_write_1h_price_per_million ?? rates.cacheWrite1hPrice,
  };
}

function isValidPrice(value: string): boolean {
  const parsed = Number(value);
  return value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0;
}

// Cache rates are optional: an empty field means "no cache price" (cache tokens
// then bill at the input rate). Empty is valid; any set value must be >= 0.
function isValidOptionalPrice(value: string): boolean {
  if (value.trim() === "") {
    return true;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0;
}

function optionalPrice(value: string): number | null {
  return value.trim() === "" ? null : Number(value);
}

type EditablePricingTier = {
  id: number;
  minInputTokens: string;
  input: string;
  output: string;
  cacheRead: string;
  cacheWrite: string;
  cacheWrite1h: string;
};

function editableTiers(tiers: PricingTier[]): EditablePricingTier[] {
  return tiers.map((tier, index) => ({
    id: index,
    minInputTokens: String(tier.min_input_tokens),
    input: tier.input_price_per_million == null ? "" : String(tier.input_price_per_million),
    output: tier.output_price_per_million == null ? "" : String(tier.output_price_per_million),
    cacheRead: tier.cache_read_price_per_million == null ? "" : String(tier.cache_read_price_per_million),
    cacheWrite: tier.cache_write_price_per_million == null ? "" : String(tier.cache_write_price_per_million),
    cacheWrite1h: tier.cache_write_1h_price_per_million == null ? "" : String(tier.cache_write_1h_price_per_million),
  }));
}

function validTiers(tiers: EditablePricingTier[]): boolean {
  const thresholds = new Set<number>();
  return tiers.every((tier) => {
    const threshold = Number(tier.minInputTokens);
    const hasOverride = [tier.input, tier.output, tier.cacheRead, tier.cacheWrite, tier.cacheWrite1h].some(
      (value) => value.trim() !== "",
    );
    if (!Number.isInteger(threshold) || threshold <= 0 || thresholds.has(threshold) || !hasOverride) {
      return false;
    }
    thresholds.add(threshold);
    return [tier.input, tier.output, tier.cacheRead, tier.cacheWrite, tier.cacheWrite1h].every(isValidOptionalPrice);
  });
}

function pricingTiers(tiers: EditablePricingTier[]): PricingTier[] {
  return tiers.map((tier) => ({
    min_input_tokens: Number(tier.minInputTokens),
    ...(tier.input.trim() === "" ? {} : { input_price_per_million: Number(tier.input) }),
    ...(tier.output.trim() === "" ? {} : { output_price_per_million: Number(tier.output) }),
    ...(tier.cacheRead.trim() === "" ? {} : { cache_read_price_per_million: Number(tier.cacheRead) }),
    ...(tier.cacheWrite.trim() === "" ? {} : { cache_write_price_per_million: Number(tier.cacheWrite) }),
    ...(tier.cacheWrite1h.trim() === "" ? {} : { cache_write_1h_price_per_million: Number(tier.cacheWrite1h) }),
  }));
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

function PricingTierEditor({
  tiers,
  onChange,
}: {
  tiers: EditablePricingTier[];
  onChange: (tiers: EditablePricingTier[]) => void;
}) {
  const update = (id: number, field: keyof EditablePricingTier, value: string) => {
    onChange(tiers.map((tier) => (tier.id === id ? { ...tier, [field]: value } : tier)));
  };
  const add = () => {
    const nextId = tiers.reduce((max, tier) => Math.max(max, tier.id), -1) + 1;
    onChange([
      ...tiers,
      { id: nextId, minInputTokens: "128000", input: "", output: "", cacheRead: "", cacheWrite: "", cacheWrite1h: "" },
    ]);
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-[var(--otari-line)] p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-medium text-[var(--otari-ink)]">Long-context price tiers</div>
          <p className="text-xs text-[var(--otari-muted)]">At a threshold, listed rates replace the base rate for the whole request.</p>
        </div>
        <Button size="sm" variant="outline" onPress={add}>
          Add tier
        </Button>
      </div>
      {tiers.map((tier) => (
        <div key={tier.id} className="flex flex-wrap items-end gap-2 border-t border-[var(--otari-line)] pt-2">
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            Context ≥ tokens
            <input
              type="number"
              min="1"
              step="1"
              inputMode="numeric"
              aria-label="Tier context threshold"
              value={tier.minInputTokens}
              onChange={(event) => update(tier.id, "minInputTokens", event.target.value)}
              className="w-28 rounded-md border border-[var(--otari-line)] bg-white px-2 py-1 text-right text-sm tabular-nums focus:border-[var(--otari-brand)] focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            Input
            <MoneyInput value={tier.input} onChange={(value) => update(tier.id, "input", value)} ariaLabel="Tier input price" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            Output
            <MoneyInput value={tier.output} onChange={(value) => update(tier.id, "output", value)} ariaLabel="Tier output price" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            Cache read
            <MoneyInput value={tier.cacheRead} onChange={(value) => update(tier.id, "cacheRead", value)} ariaLabel="Tier cache read price" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            Cache write
            <MoneyInput value={tier.cacheWrite} onChange={(value) => update(tier.id, "cacheWrite", value)} ariaLabel="Tier cache write price" />
          </label>
          <label className="flex flex-col gap-1 text-xs text-[var(--otari-muted)]">
            1h write
            <MoneyInput value={tier.cacheWrite1h} onChange={(value) => update(tier.id, "cacheWrite1h", value)} ariaLabel="Tier 1 hour cache write price" />
          </label>
          <Button size="sm" variant="ghost" onPress={() => onChange(tiers.filter((item) => item.id !== tier.id))}>
            Remove
          </Button>
        </div>
      ))}
    </div>
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
  const [cacheRead, setCacheRead] = useState("");
  const [cacheWrite, setCacheWrite] = useState("");
  const [cacheWrite1h, setCacheWrite1h] = useState("");
  const [tiers, setTiers] = useState<EditablePricingTier[]>([]);

  const startEdit = () => {
    setInput(row.inputPrice == null ? "" : String(row.inputPrice));
    setOutput(row.outputPrice == null ? "" : String(row.outputPrice));
    setCacheRead(row.cacheReadPrice == null ? "" : String(row.cacheReadPrice));
    setCacheWrite(row.cacheWritePrice == null ? "" : String(row.cacheWritePrice));
    setCacheWrite1h(row.cacheWrite1hPrice == null ? "" : String(row.cacheWrite1hPrice));
    setTiers(editableTiers(row.pricingTiers));
    setEditing(true);
  };

  const canSave =
    isValidPrice(input) &&
    isValidPrice(output) &&
    isValidOptionalPrice(cacheRead) &&
    isValidOptionalPrice(cacheWrite) &&
    isValidOptionalPrice(cacheWrite1h) &&
    validTiers(tiers);

  const save = () => {
    if (!canSave) {
      return;
    }
    setPricing.mutate(
      {
        model_key: row.key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
        cache_read_price_per_million: optionalPrice(cacheRead),
        cache_write_price_per_million: optionalPrice(cacheWrite),
        cache_write_1h_price_per_million: optionalPrice(cacheWrite1h),
        pricing_tiers: pricingTiers(tiers),
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
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-[var(--otari-muted)]">Cache read $ / 1M</span>
          <MoneyInput value={cacheRead} onChange={setCacheRead} ariaLabel={`Cache read price for ${row.key}`} />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-[var(--otari-muted)]">Cache write $ / 1M</span>
          <MoneyInput value={cacheWrite} onChange={setCacheWrite} ariaLabel={`Cache write price for ${row.key}`} />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-[var(--otari-muted)]">1h cache write $ / 1M</span>
          <MoneyInput value={cacheWrite1h} onChange={setCacheWrite1h} ariaLabel={`1 hour cache write price for ${row.key}`} />
        </div>
        <PricingTierEditor tiers={tiers} onChange={setTiers} />
        <div className="flex items-center gap-2">
          <Button size="sm" variant="primary" isDisabled={setPricing.isPending || !canSave} onPress={save}>
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
      <Spec label="Cache read" value={row.cacheReadPrice == null ? "—" : `${formatCost(row.cacheReadPrice)} / 1M`} />
      <Spec label="Cache write" value={row.cacheWritePrice == null ? "—" : `${formatCost(row.cacheWritePrice)} / 1M`} />
      <Spec label="1h cache write" value={row.cacheWrite1hPrice == null ? "—" : `${formatCost(row.cacheWrite1hPrice)} / 1M`} />
      <Spec label="Context tiers" value={row.pricingTiers.length ? `${row.pricingTiers.length} configured` : "—"} />
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

// The persistent detail panel beside the table shows the selected model's
// pricing, specs, modalities, and capabilities; it also offers to alias it.
function ModelDetailPanel({
  row,
  metadata,
  metadataAvailable,
  onMakeAlias,
  onClose,
}: {
  row: ModelRow;
  metadata: ModelMetadata | undefined;
  metadataAvailable: boolean;
  onMakeAlias: (key: string) => void;
  onClose: () => void;
}) {
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
            <p className="mt-1 text-xs break-all text-[var(--otari-muted)]">
              Selector: <code>{row.key}</code>
            </p>
            {metadata?.family ? <p className="text-xs text-[var(--otari-muted)]">{metadata.family}</p> : null}
          </div>
          <button
            type="button"
            aria-label="Close model details"
            onClick={onClose}
            className="-mt-1 -mr-1 shrink-0 rounded-md px-1.5 py-0.5 text-lg leading-none text-[var(--otari-muted)] hover:bg-[var(--otari-bg)] hover:text-[var(--otari-ink)]"
          >
            ✕
          </button>
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

      </Card.Content>
    </Card>
  );
}

const DEFAULT_PAGE_SIZE = 15;
const PAGE_SIZE_OPTIONS = [
  { value: "15", label: "15 per page" },
  { value: "25", label: "25 per page" },
  { value: "50", label: "50 per page" },
];

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

function Pagination({
  page,
  pageCount,
  total,
  pageSize,
  onPage,
  onPageSize,
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onPage: (page: number) => void;
  onPageSize: (pageSize: number) => void;
}) {
  return (
    <div className="flex items-center justify-between px-1 pt-1 text-sm text-[var(--otari-muted)]">
      <span>
        {pageCount > 1 ? `Page ${page + 1} of ${pageCount} · ` : ""}
        {formatNumber(total)} model{total === 1 ? "" : "s"}
      </span>
      <span className="inline-flex gap-2">
        <FilterSelect
          ariaLabel="Rows per page"
          value={String(pageSize)}
          onChange={(value) => onPageSize(Number(value))}
          options={PAGE_SIZE_OPTIONS}
        />
        {pageCount > 1 ? (
          <>
            <Button size="sm" variant="outline" isDisabled={page === 0} onPress={() => onPage(page - 1)}>
              Prev
            </Button>
            <Button size="sm" variant="outline" isDisabled={page >= pageCount - 1} onPress={() => onPage(page + 1)}>
              Next
            </Button>
          </>
        ) : null}
      </span>
    </div>
  );
}

type SortCol = "model" | "released" | "input" | "output";
type SortDir = "asc" | "desc";
type Sort = { col: SortCol; dir: SortDir };

// Newest models first by default; the choice is remembered across refreshes.
const SORT_STORAGE_KEY = "otari.dashboard.modelsSort";
const DEFAULT_SORT: Sort = { col: "model", dir: "asc" };
const SORT_COLS: SortCol[] = ["model", "released", "input", "output"];

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
  const [cacheRead, setCacheRead] = useState(row.cacheReadPrice == null ? "" : String(row.cacheReadPrice));
  const [cacheWrite, setCacheWrite] = useState(row.cacheWritePrice == null ? "" : String(row.cacheWritePrice));
  const [cacheWrite1h, setCacheWrite1h] = useState(row.cacheWrite1hPrice == null ? "" : String(row.cacheWrite1hPrice));
  const [tiers, setTiers] = useState<EditablePricingTier[]>(editableTiers(row.pricingTiers));

  const canSave =
    isValidPrice(input) &&
    isValidPrice(output) &&
    isValidOptionalPrice(cacheRead) &&
    isValidOptionalPrice(cacheWrite) &&
    isValidOptionalPrice(cacheWrite1h) &&
    validTiers(tiers);

  const save = () => {
    if (!canSave) return;
    setPricing.mutate(
      {
        model_key: row.key,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
        cache_read_price_per_million: optionalPrice(cacheRead),
        cache_write_price_per_million: optionalPrice(cacheWrite),
        cache_write_1h_price_per_million: optionalPrice(cacheWrite1h),
        pricing_tiers: pricingTiers(tiers),
      },
      { onSuccess: onClose },
    );
  };

  return (
    <div className="flex flex-col gap-3 px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium break-all text-[var(--otari-muted)]">{row.key}</span>
        <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
          Input $ / 1M
          <MoneyInput value={input} onChange={setInput} ariaLabel={`Input price for ${row.key}`} />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
          Output $ / 1M
          <MoneyInput value={output} onChange={setOutput} ariaLabel={`Output price for ${row.key}`} />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
          Cache read $ / 1M
          <MoneyInput value={cacheRead} onChange={setCacheRead} ariaLabel={`Cache read price for ${row.key}`} />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
          Cache write $ / 1M
          <MoneyInput value={cacheWrite} onChange={setCacheWrite} ariaLabel={`Cache write price for ${row.key}`} />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-[var(--otari-muted)]">
          1h cache write $ / 1M
          <MoneyInput value={cacheWrite1h} onChange={setCacheWrite1h} ariaLabel={`1 hour cache write price for ${row.key}`} />
        </label>
        <Button size="sm" variant="primary" isDisabled={setPricing.isPending || !canSave} onPress={save}>
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
      <PricingTierEditor tiers={tiers} onChange={setTiers} />
    </div>
  );
}

// A price pair (input/output or cache read/write) shown as two clickable
// numbers in one cell. Clicking either opens the row's inline editor, so the
// table stays narrow without hiding the edit affordance. Stops propagation so
// the click edits the price rather than only selecting the row.
function PriceCell({
  primary,
  secondary,
  rowKey,
  primaryLabel,
  secondaryLabel,
  onEdit,
}: {
  primary: number | null;
  secondary: number | null;
  rowKey: string;
  primaryLabel: string;
  secondaryLabel: string;
  onEdit: () => void;
}) {
  const number = (value: number | null, label: string) => (
    <button
      type="button"
      aria-label={`Edit ${label} price for ${rowKey}`}
      className="tabular-nums hover:text-[var(--otari-brand-dark)] hover:underline"
      onClick={(event) => {
        event.stopPropagation();
        onEdit();
      }}
    >
      {value == null ? "—" : formatCost(value)}
    </button>
  );
  return (
    <span className="inline-flex items-center justify-end gap-1">
      {number(primary, primaryLabel)}
      <span className="text-[var(--otari-muted)]">/</span>
      {number(secondary, secondaryLabel)}
    </span>
  );
}

function CachingCell({ rates, rowKey, onEdit }: { rates: EffectiveRates; rowKey: string; onEdit: () => void }) {
  const entries = [
    rates.cacheReadPrice == null ? null : `R ${formatCost(rates.cacheReadPrice)}`,
    rates.cacheWritePrice == null ? null : `W ${formatCost(rates.cacheWritePrice)}`,
    rates.cacheWrite1hPrice == null ? null : `1h ${formatCost(rates.cacheWrite1hPrice)}`,
  ].filter((entry): entry is string => entry !== null);
  return (
    <button
      type="button"
      aria-label={`Edit caching price for ${rowKey}`}
      className="max-w-44 text-right text-xs leading-5 text-[var(--otari-muted)] hover:text-[var(--otari-brand-dark)] hover:underline"
      onClick={(event) => {
        event.stopPropagation();
        onEdit();
      }}
    >
      {entries.length > 0 ? entries.join(" · ") : "Input-rate fallback"}
    </button>
  );
}

function PricingPolicyCell({ row, onEdit }: { row: ModelRow; onEdit: () => void }) {
  const thresholds = [...row.pricingTiers]
    .sort((a, b) => a.min_input_tokens - b.min_input_tokens)
    .map((tier) => formatContext(tier.min_input_tokens))
  const label =
    thresholds.length === 0
      ? "Base only"
      : `${thresholds.length} tier${thresholds.length === 1 ? "" : "s"} · ≥ ${thresholds.join(", ")}`;
  return (
    <button
      type="button"
      aria-label={`Edit pricing policy for ${row.key}`}
      className="max-w-40 text-right text-xs leading-5 text-[var(--otari-muted)] hover:text-[var(--otari-brand-dark)] hover:underline"
      onClick={(event) => {
        event.stopPropagation();
        onEdit();
      }}
    >
      {label}
    </button>
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
  comparisonContextTokens,
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
  comparisonContextTokens: number | null;
}) {
  const comparisonLabel = comparisonContextTokens == null ? "Base" : `at ${formatContext(comparisonContextTokens)}`;
  return (
    <Table>
      <THead>
        <Tr>
          <SortableTh label="Model" col="model" sort={sort} onSort={onSort} />
          <Th>Provider</Th>
          <SortableTh
            label={`${comparisonLabel} in / out $ / 1M`}
            col="input"
            sort={sort}
            onSort={onSort}
            align="right"
            info={<PricingInfo />}
          />
          <Th className="text-right">Caching {comparisonContextTokens == null ? "policy" : comparisonLabel}</Th>
          <Th className="text-right">Pricing policy</Th>
        </Tr>
      </THead>
      <tbody>
        {isLoading ? (
          <LoadingRow colSpan={5} />
        ) : rows.length > 0 ? (
          rows.map((row) => {
            const edit = () => onSetPricingKey(pricingKey === row.key ? null : row.key);
            const rates = effectiveRatesAtContext(row, comparisonContextTokens);
            return (
              <Fragment key={row.key}>
                <Tr onClick={() => onSelect(row.key)} selected={row.key === selectedKey}>
                  <Td className="font-medium break-all">
                    {row.model}
                    <span className="sr-only">{row.key}</span>
                  </Td>
                  <Td className="text-[var(--otari-muted)]">{row.provider}</Td>
                  <Td className="text-right">
                    <PriceCell
                      primary={rates.inputPrice}
                      secondary={rates.outputPrice}
                      rowKey={row.key}
                      primaryLabel="input"
                      secondaryLabel="output"
                      onEdit={edit}
                    />
                  </Td>
                  <Td className="text-right">
                    <CachingCell rates={rates} rowKey={row.key} onEdit={edit} />
                  </Td>
                  <Td className="text-right">
                    <PricingPolicyCell row={row} onEdit={edit} />
                  </Td>
                </Tr>
                {pricingKey === row.key ? (
                  <tr className="border-b border-[var(--otari-line)] bg-[var(--otari-bg)] last:border-b-0">
                    <td colSpan={5}>
                      <InlinePriceForm row={row} onClose={() => onSetPricingKey(null)} />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })
        ) : (
          <TableMessage colSpan={5}>{empty}</TableMessage>
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
  const [searchParams] = useSearchParams();
  const models = useModels();
  const pricing = usePricing();
  const discoverable = useDiscoverableModels();
  const metadata = useModelMetadata();

  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
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

  // A provider clicked on the Providers page arrives as ?provider=<instance>,
  // pre-selecting that provider's filter so the list shows only their models.
  // An empty param (e.g. /models?provider=) collapses to "all"; an unknown one
  // is reset below once the catalogue loads.
  const [providerFilter, setProviderFilter] = useState(searchParams.get("provider") || "all");
  const [pricingFilter, setPricingFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [capabilityFilter, setCapabilityFilter] = useState("all");
  const [minContext, setMinContext] = useState("0");
  const [maxInput, setMaxInput] = useState("");
  const [releaseFilter, setReleaseFilter] = useState("all");
  const [comparisonContext, setComparisonContext] = useState("");

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
        model: displayModelName(model, provider),
        provider,
        isDiscovered: discoverableKeys.has(key),
        contextWindow: catalogRow?.contextWindow ?? null,
        inputPrice: priced ? priced.input_price_per_million : (catalogRow?.inputPrice ?? null),
        outputPrice: priced ? priced.output_price_per_million : (catalogRow?.outputPrice ?? null),
        cacheReadPrice: priced ? priced.cache_read_price_per_million : (catalogRow?.cacheReadPrice ?? null),
        cacheWritePrice: priced ? priced.cache_write_price_per_million : (catalogRow?.cacheWritePrice ?? null),
        cacheWrite1hPrice: priced ? (priced.cache_write_1h_price_per_million ?? null) : (catalogRow?.cacheWrite1hPrice ?? null),
        pricingTiers: priced ? (priced.pricing_tiers ?? []) : (catalogRow?.pricingTiers ?? []),
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
        cacheReadPrice: model.pricing?.cache_read_price_per_million ?? null,
        cacheWritePrice: model.pricing?.cache_write_price_per_million ?? null,
        cacheWrite1hPrice: model.pricing?.cache_write_1h_price_per_million ?? null,
        pricingTiers: model.pricing?.pricing_tiers ?? [],
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
          model: displayModelName(model.key, provider.provider),
          provider: provider.provider,
          isDiscovered: true,
          contextWindow: metadataByKey[model.key]?.context_window ?? null,
          releaseDate: metadataByKey[model.key]?.release_date ?? null,
          inputPrice: null,
          outputPrice: null,
          cacheReadPrice: null,
          cacheWritePrice: null,
          cacheWrite1hPrice: null,
          pricingTiers: [],
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

  // A ?provider= value seeded from the URL may name a provider with no models,
  // or a stale/misspelled one. The native <select> would then render blank and
  // the table would show zero rows, a confusing dead end. Once the catalogue has
  // loaded (options beyond "all"), drop an unknown value back to "all".
  useEffect(() => {
    if (providerFilter === "all" || providerOptions.length <= 1) {
      return;
    }
    if (!providerOptions.some((option) => option.value === providerFilter)) {
      setProviderFilter("all");
    }
  }, [providerOptions, providerFilter]);

  const query = search.trim().toLowerCase();
  const minContextValue = Number(minContext) || 0;
  const maxInputValue = maxInput === "" ? Number.POSITIVE_INFINITY : Number(maxInput);
  const comparisonContextTokens = comparisonContext === "" ? null : Number(comparisonContext);
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
      const pick = (row: ModelRow) => (sort.col === "input" ? row.inputPrice : row.outputPrice);
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
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const clampedPage = Math.min(page, pageCount - 1);
  const start = clampedPage * pageSize;
  const pageModels = filteredModels.slice(start, start + pageSize);

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
  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Models"
        description="Every model your providers can serve. Set a price on any model so budgets and usage tracking work."
      />

      <ErrorBanner
        error={models.error ?? pricing.error ?? discoverable.error ?? metadata.error}
      />

      <div
        className={`grid gap-4 lg:items-start ${
          selectedRow ? "lg:grid-cols-[minmax(0,1fr)_360px]" : "grid-cols-1"
        }`}
      >
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
                ariaLabel="Compare prices at context"
                value={comparisonContext}
                onChange={setComparisonContext}
                options={PRICE_COMPARISON_OPTIONS}
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
              comparisonContextTokens={comparisonContextTokens}
            />

            <div className="flex flex-wrap items-center justify-between gap-2">
              <Pagination
                page={clampedPage}
                pageCount={pageCount}
                total={total}
                pageSize={pageSize}
                onPage={setPage}
                onPageSize={(size) => {
                  setPageSize(size);
                  setPage(0);
                }}
              />
            </div>
          </div>

          {selectedRow ? (
            <aside className="lg:sticky lg:top-4">
              <ModelDetailPanel
                row={selectedRow}
                metadata={metadataByKey[selectedRow.key]}
                metadataAvailable={metadataAvailable}
                onMakeAlias={(key) => navigate(`/aliases?target=${encodeURIComponent(key)}`)}
                onClose={() => setSelectedKey(null)}
              />
            </aside>
          ) : null}
        </div>
    </div>
  );
}
