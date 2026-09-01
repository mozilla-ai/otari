// biome-ignore-all lint/a11y/noLabelWithoutControl: every label here wraps MoneyInput, which renders the input and takes its own ariaLabel. The rule cannot see through a component, and it still works on this file's native inputs.
import { Button } from "@heroui/react"
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useState,
} from "react"
import type { Selection, SortDescriptor } from "react-aria-components"
import type { DiscoverableProvider, ModelMetadata, PricingTier } from "@/client"
import { isPricingTier } from "@/client"
import { currentPricing, providerFromModelKey } from "@/features/models/pricing"
import {
  isValidModelKey,
  type ManualRates,
  SetPriceDialog,
} from "@/features/models/SetPriceDialog"
import {
  useDeletePricing,
  useDiscoverableModels,
  useModelMetadata,
  useModels,
  usePricing,
  useSetPricing,
  useSettings,
} from "@/shared/api/hooks"
import { BulkActionBar } from "@/shared/components/BulkActionBar"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Dot, Section, TableScrollFrame } from "@/shared/components/surface"
import { TablePagination } from "@/shared/components/TablePagination"
import {
  ConfirmButton,
  CopyableValue,
  ErrorBanner,
  errorMessage,
  FilterSelect,
} from "@/shared/components/ui"
import {
  formatContext,
  formatCost,
  formatReleaseDate,
} from "@/shared/helpers/format"
import {
  resolveSelectedIds,
  useTableSelection,
} from "@/shared/helpers/tableSelection"
import { useUrlValue } from "@/shared/helpers/urlState"

// `owned_by` the gateway stamps on a configured alias (ALIAS_OWNED_BY in
// src/gateway/api/routes/models.py). Aliases are display names declared in
// config.yml, not models, so they cannot be priced here.
// Provider segment of a gateway-run tool's pricing key; see the loop in `rows`.
const GATEWAY_TOOL_PRICING_PROVIDER = "otari"

const ALIAS_OWNED_BY = "otari"

// Where a row's price comes from. "configured" is a DB price and the only kind
// editable here; "default" is the genai-prices fallback; "alias" is inherited
// from a target; "none" means metered at no cost. "unknown" is the one case the
// dashboard cannot answer: a model discovery reports that the catalog did not
// list, with default pricing on. The gateway would meter it at the fallback rate,
// but only `GET /v1/models` computes that rate and this row did not come from
// there, so claiming "not priced" here would be a guess that reads as a fact.
type PriceSource = "configured" | "default" | "alias" | "none" | "unknown"

// Capability filter options. These test the model's own metadata (models.dev),
// so they line up with the per-model Features chips shown in the table, not the
// coarser provider-level capabilities. Picking "Vision" narrows to models whose
// metadata actually reports image input, matching what the chips display.
const MODEL_FILTER_CAPABILITIES: {
  value: string
  label: string
  test: (metadata: ModelMetadata) => boolean
}[] = [
  {
    value: "vision",
    label: "Vision",
    test: (m) =>
      Array.isArray(m.input_modalities) && m.input_modalities.includes("image"),
  },
  {
    value: "tool_call",
    label: "Tool calling",
    test: (m) => Boolean(m.tool_call),
  },
  { value: "reasoning", label: "Reasoning", test: (m) => Boolean(m.reasoning) },
  {
    value: "structured_output",
    label: "Structured output",
    test: (m) => Boolean(m.structured_output),
  },
  {
    value: "attachment",
    label: "Attachments",
    test: (m) => Boolean(m.attachment),
  },
  {
    value: "audio",
    label: "Audio",
    test: (m) =>
      Array.isArray(m.input_modalities) && m.input_modalities.includes("audio"),
  },
  {
    value: "pdf",
    label: "PDF",
    test: (m) =>
      Array.isArray(m.input_modalities) && m.input_modalities.includes("pdf"),
  },
]

// Per-model capability flags from models.dev, labeled for the detail panel.
const MODEL_CAPABILITY_LABELS: { key: keyof ModelMetadata; label: string }[] = [
  { key: "reasoning", label: "Reasoning" },
  { key: "tool_call", label: "Tool calling" },
  { key: "structured_output", label: "Structured output" },
  { key: "attachment", label: "Attachments" },
  { key: "temperature", label: "Temperature" },
]

const MODALITY_LABELS: Record<string, string> = {
  text: "Text",
  image: "Image",
  audio: "Audio",
  video: "Video",
  pdf: "PDF",
}

const CONTEXT_OPTIONS = [
  { value: "0", label: "Any context" },
  { value: "8000", label: "≥ 8K" },
  { value: "32000", label: "≥ 32K" },
  { value: "128000", label: "≥ 128K" },
  { value: "200000", label: "≥ 200K" },
  { value: "1000000", label: "≥ 1M" },
]

const PRICE_OPTIONS = [
  { value: "", label: "Any price" },
  { value: "1", label: "≤ $1 / 1M in" },
  { value: "3", label: "≤ $3 / 1M in" },
  { value: "10", label: "≤ $10 / 1M in" },
  { value: "30", label: "≤ $30 / 1M in" },
]

const PRICE_COMPARISON_OPTIONS = [
  { value: "", label: "Base prices" },
  { value: "8000", label: "Compare at 8K" },
  { value: "128000", label: "Compare at 128K" },
  { value: "200000", label: "Compare at 200K" },
  { value: "500000", label: "Compare at 500K" },
  { value: "1000000", label: "Compare at 1M" },
]

// Newness windows for the release-date filter, in days back from today. Rows
// with no known release date are excluded once a window is active.
const RELEASE_OPTIONS = [
  { value: "all", label: "Any release date" },
  { value: "365", label: "Past year" },
  { value: "730", label: "Past 2 years" },
  { value: "1095", label: "Past 3 years" },
]
const RELEASE_WINDOW_MS = 24 * 60 * 60 * 1000

interface ModelRow {
  key: string
  model: string
  provider: string
  isDiscovered?: boolean
  contextWindow: number | null
  releaseDate?: string | null
  inputPrice: number | null
  outputPrice: number | null
  cacheReadPrice: number | null
  cacheWritePrice: number | null
  cacheWrite1hPrice: number | null
  pricingTiers: PricingTier[]
  source: PriceSource
  /** What the model accepts and emits, for the Modalities lane. */
  inputModalities?: string[]
  outputModalities?: string[]
}

type EffectiveRates = Pick<
  ModelRow,
  | "inputPrice"
  | "outputPrice"
  | "cacheReadPrice"
  | "cacheWritePrice"
  | "cacheWrite1hPrice"
>

// Stable row-key getter so DataTable's per-row cache holds across re-renders.
const getModelRowKey = (row: ModelRow): string => row.key

function displayModelName(selector: string, provider: string): string {
  const instancePrefix = `${provider}:`
  return selector.startsWith(instancePrefix)
    ? selector.slice(instancePrefix.length)
    : selector
}

function effectiveRatesAtContext(
  row: ModelRow,
  contextTokens: number | null,
): EffectiveRates {
  const rates: EffectiveRates = {
    inputPrice: row.inputPrice,
    outputPrice: row.outputPrice,
    cacheReadPrice: row.cacheReadPrice,
    cacheWritePrice: row.cacheWritePrice,
    cacheWrite1hPrice: row.cacheWrite1hPrice,
  }
  if (contextTokens == null) {
    return rates
  }
  const tier = row.pricingTiers
    .filter((candidate) => candidate.min_input_tokens <= contextTokens)
    .sort((a, b) => b.min_input_tokens - a.min_input_tokens)[0]
  if (!tier) {
    return rates
  }
  return {
    inputPrice: tier.input_price_per_million ?? rates.inputPrice,
    outputPrice: tier.output_price_per_million ?? rates.outputPrice,
    cacheReadPrice: tier.cache_read_price_per_million ?? rates.cacheReadPrice,
    cacheWritePrice:
      tier.cache_write_price_per_million ?? rates.cacheWritePrice,
    cacheWrite1hPrice:
      tier.cache_write_1h_price_per_million ?? rates.cacheWrite1hPrice,
  }
}

function isValidPrice(value: string): boolean {
  const parsed = Number(value)
  return value.trim() !== "" && Number.isFinite(parsed) && parsed >= 0
}

// Cache rates are optional: an empty field means "no cache price" (cache tokens
// then bill at the input rate). Empty is valid; any set value must be >= 0.
function isValidOptionalPrice(value: string): boolean {
  if (value.trim() === "") {
    return true
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0
}

function optionalPrice(value: string): number | null {
  return value.trim() === "" ? null : Number(value)
}

type EditablePricingTier = {
  id: number
  minInputTokens: string
  input: string
  output: string
  cacheRead: string
  cacheWrite: string
  cacheWrite1h: string
}

function editableTiers(tiers: PricingTier[]): EditablePricingTier[] {
  return tiers.map((tier, index) => ({
    id: index,
    minInputTokens: String(tier.min_input_tokens),
    input:
      tier.input_price_per_million == null
        ? ""
        : String(tier.input_price_per_million),
    output:
      tier.output_price_per_million == null
        ? ""
        : String(tier.output_price_per_million),
    cacheRead:
      tier.cache_read_price_per_million == null
        ? ""
        : String(tier.cache_read_price_per_million),
    cacheWrite:
      tier.cache_write_price_per_million == null
        ? ""
        : String(tier.cache_write_price_per_million),
    cacheWrite1h:
      tier.cache_write_1h_price_per_million == null
        ? ""
        : String(tier.cache_write_1h_price_per_million),
  }))
}

function validTiers(tiers: EditablePricingTier[]): boolean {
  const thresholds = new Set<number>()
  return tiers.every((tier) => {
    const threshold = Number(tier.minInputTokens)
    const hasOverride = [
      tier.input,
      tier.output,
      tier.cacheRead,
      tier.cacheWrite,
      tier.cacheWrite1h,
    ].some((value) => value.trim() !== "")
    if (
      !Number.isInteger(threshold) ||
      threshold <= 0 ||
      thresholds.has(threshold) ||
      !hasOverride
    ) {
      return false
    }
    thresholds.add(threshold)
    return [
      tier.input,
      tier.output,
      tier.cacheRead,
      tier.cacheWrite,
      tier.cacheWrite1h,
    ].every(isValidOptionalPrice)
  })
}

function pricingTiers(tiers: EditablePricingTier[]): PricingTier[] {
  return tiers.map((tier) => ({
    min_input_tokens: Number(tier.minInputTokens),
    ...(tier.input.trim() === ""
      ? {}
      : { input_price_per_million: Number(tier.input) }),
    ...(tier.output.trim() === ""
      ? {}
      : { output_price_per_million: Number(tier.output) }),
    ...(tier.cacheRead.trim() === ""
      ? {}
      : { cache_read_price_per_million: Number(tier.cacheRead) }),
    ...(tier.cacheWrite.trim() === ""
      ? {}
      : { cache_write_price_per_million: Number(tier.cacheWrite) }),
    ...(tier.cacheWrite1h.trim() === ""
      ? {}
      : { cache_write_1h_price_per_million: Number(tier.cacheWrite1h) }),
  }))
}

function MoneyInput({
  value,
  onChange,
  ariaLabel,
}: {
  value: string
  onChange: (value: string) => void
  ariaLabel: string
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
      className="w-28 rounded-md border border-field-border bg-field px-2 py-1 text-right text-sm tabular-nums focus:border-accent focus:outline-none"
    />
  )
}

function PricingTierEditor({
  tiers,
  onChange,
}: {
  tiers: EditablePricingTier[]
  onChange: (tiers: EditablePricingTier[]) => void
}) {
  const update = (
    id: number,
    field: keyof EditablePricingTier,
    value: string,
  ) => {
    onChange(
      tiers.map((tier) =>
        tier.id === id ? { ...tier, [field]: value } : tier,
      ),
    )
  }
  const add = () => {
    const nextId = tiers.reduce((max, tier) => Math.max(max, tier.id), -1) + 1
    onChange([
      ...tiers,
      {
        id: nextId,
        minInputTokens: "128000",
        input: "",
        output: "",
        cacheRead: "",
        cacheWrite: "",
        cacheWrite1h: "",
      },
    ])
  }

  return (
    <div className="flex flex-col gap-2 border border-control-border p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-xs font-medium text-foreground">
            Long-context price tiers
          </div>
          <p className="text-xs text-muted">
            At a threshold, listed rates replace the base rate for the whole
            request.
          </p>
        </div>
        <Button size="sm" variant="outline" onPress={add}>
          Add tier
        </Button>
      </div>
      {tiers.map((tier) => (
        <div
          key={tier.id}
          className="flex flex-wrap items-end gap-2 border-t border-border pt-2"
        >
          <label className="flex flex-col gap-1 text-xs text-muted">
            Context ≥ tokens
            <input
              type="number"
              min="1"
              step="1"
              inputMode="numeric"
              aria-label="Tier context threshold"
              value={tier.minInputTokens}
              onChange={(event) =>
                update(tier.id, "minInputTokens", event.target.value)
              }
              className="w-28 rounded-md border border-field-border bg-field px-2 py-1 text-right text-sm tabular-nums focus:border-accent focus:outline-none"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            Input
            <MoneyInput
              value={tier.input}
              onChange={(value) => update(tier.id, "input", value)}
              ariaLabel="Tier input price"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            Output
            <MoneyInput
              value={tier.output}
              onChange={(value) => update(tier.id, "output", value)}
              ariaLabel="Tier output price"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            Cache read
            <MoneyInput
              value={tier.cacheRead}
              onChange={(value) => update(tier.id, "cacheRead", value)}
              ariaLabel="Tier cache read price"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            Cache write
            <MoneyInput
              value={tier.cacheWrite}
              onChange={(value) => update(tier.id, "cacheWrite", value)}
              ariaLabel="Tier cache write price"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            1h write
            <MoneyInput
              value={tier.cacheWrite1h}
              onChange={(value) => update(tier.id, "cacheWrite1h", value)}
              ariaLabel="Tier 1 hour cache write price"
            />
          </label>
          <Button
            size="sm"
            variant="ghost"
            onPress={() =>
              onChange(tiers.filter((item) => item.id !== tier.id))
            }
          >
            Remove
          </Button>
        </div>
      ))}
    </div>
  )
}

/**
 * Where a model's price came from, as a dot and a word.
 *
 * `configured` is the only affirmative one, and it takes the accent dot with
 * foreground ink because it is the one that means somebody set this
 * deliberately. A catalog default or an alias is a fact rather than a decision
 * and reads muted; an unpriced model is the absence of one and reads subtle.
 * The accent is the data ink here, not a way to say "look at this".
 */
/** One row of a category set: a subtle group label, then its members. */
function ModalitySet({ label, values }: { label: string; values: string[] }) {
  if (values.length === 0) return null
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 font-mono text-[11px] tracking-[0.1em] uppercase">
      <span className="text-subtle">{label}</span>
      {values.map((m) => (
        <span key={m} className="text-muted">
          {MODALITY_LABELS[m] ?? m}
        </span>
      ))}
    </div>
  )
}

function SourceMark({ source }: { source: PriceSource }) {
  const { word, dot, ink } =
    source === "configured"
      ? { word: "Custom", dot: "bg-accent", ink: "text-foreground" }
      : source === "default" || source === "alias"
        ? { word: source, dot: "bg-surface-subtle", ink: "text-muted" }
        : source === "unknown"
          ? {
              word: "rate unknown",
              dot: "bg-surface-subtle",
              ink: "text-subtle",
            }
          : { word: "not priced", dot: "bg-surface-subtle", ink: "text-subtle" }
  return (
    <span className={`flex items-center gap-2 font-mono text-[13px] ${ink}`}>
      <Dot className={dot} />
      {word.toUpperCase()}
    </span>
  )
}

// A small info affordance: an "i" bubble that reveals a tooltip on hover or
// keyboard focus. Hand-rolled to match the dashboard's other bare primitives
// (HeroUI v3 ships no tooltip we use here).
function InfoTooltip({
  label,
  tone = "info",
  children,
}: {
  label: string
  tone?: "info" | "warning"
  children: ReactNode
}) {
  // Unique per instance: several tooltips render at once, so a shared id would
  // make aria-describedby resolve to the wrong (or ambiguous) tooltip.
  const tipId = useId()
  return (
    <span className="group relative inline-flex items-center font-normal normal-case">
      <button
        type="button"
        aria-label={label}
        aria-describedby={tipId}
        className={`inline-flex h-4 w-4 items-center justify-center rounded-full border text-[10px] leading-none ${
          tone === "warning"
            ? "border-warning text-warning"
            : "border-border text-muted hover:border-accent hover:text-link"
        }`}
      >
        i
      </button>
      <span
        id={tipId}
        role="tooltip"
        className="pointer-events-none absolute top-full right-0 z-20 mt-1.5 w-72 border border-control-border bg-surface px-3 py-2 text-left text-xs font-normal whitespace-normal break-words text-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {children}
      </span>
    </span>
  )
}

// The default-pricing explanation, surfaced from the pricing column headers
// instead of a persistent banner. Tone shifts to a warning when metering is off.
function PricingInfo() {
  const settings = useSettings()
  if (!settings.data) {
    return null
  }
  if (settings.data.default_pricing) {
    return (
      <InfoTooltip label="How unpriced models are metered" tone="info">
        Default pricing is on: models without a configured price are metered
        using community-maintained rates (the bundled genai-prices dataset). Set
        a price to override the fallback.
      </InfoTooltip>
    )
  }
  return (
    <InfoTooltip label="How unpriced models are metered" tone="warning">
      Default pricing is off: only models with a configured price are metered.
      {settings.data.require_pricing
        ? " Requests for any other model are rejected (HTTP 402) because require_pricing is on."
        : " Other models are served without cost tracking."}
    </InfoTooltip>
  )
}

// Small labeled key/value used throughout the detail panel.
function Spec({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-xs text-muted">{label}</span>
      <span className="text-right text-sm text-foreground tabular-nums">
        {value}
      </span>
    </div>
  )
}

function PanelSection({
  title,
  children,
}: {
  title: string
  children: ReactNode
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-overline">{title}</span>
      {children}
    </div>
  )
}

// Price editor inside the detail panel: the selected model's rate, plus
// set/edit/clear. Reset per model via a `key` on the element that mounts it.
function PanelPriceEditor({ row }: { row: ModelRow }) {
  const setPricing = useSetPricing()
  const deletePricing = useDeletePricing()
  const [editing, setEditing] = useState(false)
  const [input, setInput] = useState("")
  const [output, setOutput] = useState("")
  const [cacheRead, setCacheRead] = useState("")
  const [cacheWrite, setCacheWrite] = useState("")
  const [cacheWrite1h, setCacheWrite1h] = useState("")
  const [tiers, setTiers] = useState<EditablePricingTier[]>([])

  const startEdit = () => {
    setInput(row.inputPrice == null ? "" : String(row.inputPrice))
    setOutput(row.outputPrice == null ? "" : String(row.outputPrice))
    setCacheRead(row.cacheReadPrice == null ? "" : String(row.cacheReadPrice))
    setCacheWrite(
      row.cacheWritePrice == null ? "" : String(row.cacheWritePrice),
    )
    setCacheWrite1h(
      row.cacheWrite1hPrice == null ? "" : String(row.cacheWrite1hPrice),
    )
    setTiers(editableTiers(row.pricingTiers))
    setEditing(true)
  }

  const canSave =
    isValidPrice(input) &&
    isValidPrice(output) &&
    isValidOptionalPrice(cacheRead) &&
    isValidOptionalPrice(cacheWrite) &&
    isValidOptionalPrice(cacheWrite1h) &&
    validTiers(tiers)

  const save = () => {
    if (!canSave) {
      return
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
    )
  }

  if (editing) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted">Input $ / 1M</span>
          <MoneyInput
            value={input}
            onChange={setInput}
            ariaLabel={`Input price for ${row.key}`}
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted">Output $ / 1M</span>
          <MoneyInput
            value={output}
            onChange={setOutput}
            ariaLabel={`Output price for ${row.key}`}
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted">Cache read $ / 1M</span>
          <MoneyInput
            value={cacheRead}
            onChange={setCacheRead}
            ariaLabel={`Cache read price for ${row.key}`}
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted">Cache write $ / 1M</span>
          <MoneyInput
            value={cacheWrite}
            onChange={setCacheWrite}
            ariaLabel={`Cache write price for ${row.key}`}
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted">1h cache write $ / 1M</span>
          <MoneyInput
            value={cacheWrite1h}
            onChange={setCacheWrite1h}
            ariaLabel={`1 hour cache write price for ${row.key}`}
          />
        </div>
        <PricingTierEditor tiers={tiers} onChange={setTiers} />
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="primary"
            isDisabled={setPricing.isPending || !canSave}
            onPress={save}
          >
            Save
          </Button>
          <Button
            size="sm"
            variant="ghost"
            isDisabled={setPricing.isPending}
            onPress={() => setEditing(false)}
          >
            Cancel
          </Button>
        </div>
        {setPricing.error ? (
          <span className="text-xs text-danger">
            {errorMessage(setPricing.error)}
          </span>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <Spec
        label="Input"
        value={
          row.inputPrice == null ? "—" : `${formatCost(row.inputPrice)} / 1M`
        }
      />
      <Spec
        label="Output"
        value={
          row.outputPrice == null ? "—" : `${formatCost(row.outputPrice)} / 1M`
        }
      />
      <Spec
        label="Cache read"
        value={
          row.cacheReadPrice == null
            ? "—"
            : `${formatCost(row.cacheReadPrice)} / 1M`
        }
      />
      <Spec
        label="Cache write"
        value={
          row.cacheWritePrice == null
            ? "—"
            : `${formatCost(row.cacheWritePrice)} / 1M`
        }
      />
      <Spec
        label="1h cache write"
        value={
          row.cacheWrite1hPrice == null
            ? "—"
            : `${formatCost(row.cacheWrite1hPrice)} / 1M`
        }
      />
      <Spec
        label="Context tiers"
        value={
          row.pricingTiers.length
            ? `${row.pricingTiers.length} configured`
            : "—"
        }
      />
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
              Removes the custom price. The model reverts to the default rate
              (genai-prices) when default pricing is on, otherwise it is metered
              at no cost.
            </InfoTooltip>
          </>
        ) : null}
        {deletePricing.error ? (
          <span className="text-xs text-danger">
            {errorMessage(deletePricing.error)}
          </span>
        ) : null}
      </div>
    </div>
  )
}

// The persistent detail panel beside the table shows the selected model's
// pricing, specs, modalities, and capabilities.
function ModelDetailPanel({
  row,
  metadata,
  metadataAvailable,
  onClose,
}: {
  row: ModelRow
  metadata: ModelMetadata | undefined
  metadataAvailable: boolean
  onClose: () => void
}) {
  const inputModalities = metadata?.input_modalities ?? []
  const outputModalities = metadata?.output_modalities ?? []
  const activeCaps = MODEL_CAPABILITY_LABELS.filter(
    ({ key }) => metadata?.[key],
  )

  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-5"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-title break-all">{row.model}</h2>
            {metadata?.deprecated ? (
              <span className="flex items-center gap-2 font-mono text-[13px] text-danger">
                <Dot className="bg-danger" />
                DEPRECATED
              </span>
            ) : null}
          </div>
          <p className="mt-1 text-xs break-all text-muted">
            Selector:{" "}
            <CopyableValue value={row.key} label="model id">
              <code>{row.key}</code>
            </CopyableValue>
          </p>
          {metadata?.family ? (
            <p className="text-xs text-muted">{metadata.family}</p>
          ) : null}
        </div>
        <button
          type="button"
          aria-label="Close model details"
          onClick={onClose}
          className="-mt-1 -mr-1 shrink-0 rounded-md px-1.5 py-0.5 text-lg leading-none text-muted hover:bg-surface-alt hover:text-foreground"
        >
          ✕
        </button>
      </div>

      {metadata?.description ? (
        <p className="text-sm text-foreground">{metadata.description}</p>
      ) : null}

      <PanelSection title="Pricing">
        <div className="flex items-center gap-2">
          <SourceMark source={row.source} />
          {row.isDiscovered ? null : (
            <span className="text-xs text-muted">not discovered</span>
          )}
        </div>
        <PanelPriceEditor key={row.key} row={row} />
      </PanelSection>

      <PanelSection title="Specs">
        <Spec label="Context window" value={formatContext(row.contextWindow)} />
        <Spec
          label="Max output"
          value={formatContext(metadata?.max_output_tokens ?? null)}
        />
        <Spec
          label="Knowledge cutoff"
          value={metadata?.knowledge_cutoff ?? "—"}
        />
        <Spec
          label="Released"
          value={formatReleaseDate(metadata?.release_date)}
        />
        <Spec
          label="Open weights"
          value={metadata ? (metadata.open_weights ? "Yes" : "No") : "—"}
        />
      </PanelSection>

      <PanelSection title="Modalities">
        {inputModalities.length === 0 && outputModalities.length === 0 ? (
          <span className="text-sm text-muted">Unknown.</span>
        ) : (
          /* A category set, not chips: these are the kinds a model accepts
               and emits, and a list of kinds is read as a list. Mono uppercase
               with the group label on the subtle rung, so `In` and `Out` name
               the two sets without competing with their members. */
          <div className="flex flex-col gap-2">
            <ModalitySet label="In" values={inputModalities} />
            <ModalitySet label="Out" values={outputModalities} />
          </div>
        )}
      </PanelSection>

      <PanelSection title="Capabilities">
        {activeCaps.length > 0 ? (
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[11px] tracking-[0.1em] text-muted uppercase">
            {activeCaps.map(({ key, label }) => (
              <span key={key}>{label}</span>
            ))}
          </div>
        ) : (
          <span className="text-sm text-muted">
            {metadataAvailable
              ? "None reported."
              : "Extended metadata unavailable (models.dev disabled or unreachable)."}
          </span>
        )}
      </PanelSection>
    </Section>
  )
}

const DEFAULT_PAGE_SIZE = 15
function SearchInput({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  return (
    <input
      type="search"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      aria-label={placeholder}
      className="w-full max-w-xs rounded-md border border-field-border bg-field px-3 py-1.5 text-sm focus:border-accent focus:outline-none"
    />
  )
}

type SortCol = "model" | "released" | "input" | "output"
type SortDir = "asc" | "desc"
type Sort = { col: SortCol; dir: SortDir }

// Newest models first by default; the choice is remembered across refreshes.
const SORT_STORAGE_KEY = "otari.dashboard.modelsSort"
const DEFAULT_SORT: Sort = { col: "model", dir: "asc" }
const SORT_COLS: SortCol[] = ["model", "released", "input", "output"]

function readStoredSort(): Sort {
  if (typeof window === "undefined") return DEFAULT_SORT
  try {
    const raw = window.localStorage.getItem(SORT_STORAGE_KEY)
    if (!raw) return DEFAULT_SORT
    const parsed = JSON.parse(raw) as { col?: unknown; dir?: unknown }
    if (
      SORT_COLS.includes(parsed.col as SortCol) &&
      (parsed.dir === "asc" || parsed.dir === "desc")
    ) {
      return { col: parsed.col as SortCol, dir: parsed.dir }
    }
  } catch {
    // Ignore malformed storage and fall back to the default.
  }
  return DEFAULT_SORT
}

// Compact inline price editor shown in a sub-row under a model, so you can
// price it without leaving the list.
function InlinePriceForm({
  row,
  onClose,
}: {
  row: ModelRow
  onClose: () => void
}) {
  const setPricing = useSetPricing()
  const deletePricing = useDeletePricing()
  const [input, setInput] = useState(
    row.inputPrice == null ? "" : String(row.inputPrice),
  )
  const [output, setOutput] = useState(
    row.outputPrice == null ? "" : String(row.outputPrice),
  )
  const [cacheRead, setCacheRead] = useState(
    row.cacheReadPrice == null ? "" : String(row.cacheReadPrice),
  )
  const [cacheWrite, setCacheWrite] = useState(
    row.cacheWritePrice == null ? "" : String(row.cacheWritePrice),
  )
  const [cacheWrite1h, setCacheWrite1h] = useState(
    row.cacheWrite1hPrice == null ? "" : String(row.cacheWrite1hPrice),
  )
  const [tiers, setTiers] = useState<EditablePricingTier[]>(
    editableTiers(row.pricingTiers),
  )

  const canSave =
    isValidPrice(input) &&
    isValidPrice(output) &&
    isValidOptionalPrice(cacheRead) &&
    isValidOptionalPrice(cacheWrite) &&
    isValidOptionalPrice(cacheWrite1h) &&
    validTiers(tiers)

  const save = () => {
    if (!canSave) return
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
    )
  }

  return (
    <div className="flex flex-col gap-3 px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium break-all text-muted">
          {row.key}
        </span>
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Input $ / 1M
          <MoneyInput
            value={input}
            onChange={setInput}
            ariaLabel={`Input price for ${row.key}`}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Output $ / 1M
          <MoneyInput
            value={output}
            onChange={setOutput}
            ariaLabel={`Output price for ${row.key}`}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Cache read $ / 1M
          <MoneyInput
            value={cacheRead}
            onChange={setCacheRead}
            ariaLabel={`Cache read price for ${row.key}`}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Cache write $ / 1M
          <MoneyInput
            value={cacheWrite}
            onChange={setCacheWrite}
            ariaLabel={`Cache write price for ${row.key}`}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-muted">
          1h cache write $ / 1M
          <MoneyInput
            value={cacheWrite1h}
            onChange={setCacheWrite1h}
            ariaLabel={`1 hour cache write price for ${row.key}`}
          />
        </label>
        <Button
          size="sm"
          variant="primary"
          isDisabled={setPricing.isPending || !canSave}
          onPress={save}
        >
          {setPricing.isPending ? "Saving…" : "Save"}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          isDisabled={setPricing.isPending}
          onPress={onClose}
        >
          Cancel
        </Button>
        {row.source === "configured" ? (
          <span className="inline-flex items-center gap-1">
            <ConfirmButton
              confirmLabel="Reset"
              isPending={deletePricing.isPending}
              onConfirm={() =>
                deletePricing.mutate(row.key, { onSuccess: onClose })
              }
            >
              Reset
            </ConfirmButton>
            <InfoTooltip label="What reset does">
              Removes the custom price. The model reverts to the default rate
              (genai-prices) when default pricing is on, otherwise it is metered
              at no cost.
            </InfoTooltip>
          </span>
        ) : null}
        {setPricing.error || deletePricing.error ? (
          <span className="text-xs text-danger">
            {errorMessage(setPricing.error ?? deletePricing.error)}
          </span>
        ) : null}
      </div>
      <PricingTierEditor tiers={tiers} onChange={setTiers} />
    </div>
  )
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
  primary: number | null
  secondary: number | null
  rowKey: string
  primaryLabel: string
  secondaryLabel: string
  onEdit: () => void
}) {
  // Each half gets the same fixed width, so the slash sits in one column all the
  // way down and the two numbers can be read as two columns rather than as one
  // ragged string. Without it the pair is right-aligned as a whole, which puts
  // every row's slash somewhere different and makes the input prices, the half
  // an operator actually scans, impossible to compare. `min-w` rather than `w`:
  // an unusually long price pushes its own row wider instead of being clipped.
  const number = (
    value: number | null,
    label: string,
    align: "text-right" | "text-left",
  ) => (
    <button
      type="button"
      aria-label={`Edit ${label} price for ${rowKey}`}
      className={`min-w-[7ch] tabular-nums ${align} hover:text-link hover:underline`}
      onClick={(event) => {
        event.stopPropagation()
        onEdit()
      }}
    >
      {value == null ? "—" : formatCost(value)}
    </button>
  )
  return (
    <span className="inline-flex items-center justify-end gap-1">
      {number(primary, primaryLabel, "text-right")}
      <span className="text-muted">/</span>
      {number(secondary, secondaryLabel, "text-left")}
    </span>
  )
}

function CachingCell({
  rates,
  rowKey,
  onEdit,
}: {
  rates: EffectiveRates
  rowKey: string
  onEdit: () => void
}) {
  const entries = [
    rates.cacheReadPrice == null
      ? null
      : { key: "R", value: formatCost(rates.cacheReadPrice) },
    rates.cacheWritePrice == null
      ? null
      : { key: "W", value: formatCost(rates.cacheWritePrice) },
    rates.cacheWrite1hPrice == null
      ? null
      : { key: "1h", value: formatCost(rates.cacheWrite1hPrice) },
  ].filter((entry): entry is { key: string; value: string } => entry !== null)
  return (
    <button
      type="button"
      aria-label={`Edit caching price for ${rowKey}`}
      className="max-w-44 text-right text-xs leading-5 text-muted hover:text-link hover:underline"
      onClick={(event) => {
        event.stopPropagation()
        onEdit()
      }}
    >
      {/* The same fixed-width treatment as the input/output pair next door, for
          the same reason: joined into one string these read as prose and the
          read and write prices cannot be compared down the column. */}
      {entries.length > 0 ? (
        <span className="inline-flex items-center justify-end">
          {entries.map((entry, index) => (
            <span key={entry.key} className="inline-flex items-center gap-1">
              {index > 0 ? <span className="px-1">·</span> : null}
              <span className="text-subtle">{entry.key}</span>
              <span className="min-w-[7ch] text-right tabular-nums">
                {entry.value}
              </span>
            </span>
          ))}
        </span>
      ) : (
        "Input-rate fallback"
      )}
    </button>
  )
}

function PricingPolicyCell({
  row,
  onEdit,
}: {
  row: ModelRow
  onEdit: () => void
}) {
  const thresholds = [...row.pricingTiers]
    .sort((a, b) => a.min_input_tokens - b.min_input_tokens)
    .map((tier) => formatContext(tier.min_input_tokens))
  const label =
    thresholds.length === 0
      ? "Base only"
      : `${thresholds.length} tier${thresholds.length === 1 ? "" : "s"} · ≥ ${thresholds.join(", ")}`
  return (
    <button
      type="button"
      aria-label={`Edit pricing policy for ${row.key}`}
      className="max-w-40 text-left font-mono text-[13px] leading-5 text-muted hover:text-foreground"
      onClick={(event) => {
        event.stopPropagation()
        onEdit()
      }}
    >
      {label}
    </button>
  )
}

function ModelTable({
  rows,
  isLoading,
  empty,
  sortDescriptor,
  onSortChange,
  selectedKey,
  onSelect,
  onEditPricing,
  comparisonContextTokens,
  selectedKeys,
  onSelectionChange,
}: {
  rows: ModelRow[]
  isLoading: boolean
  empty: ReactNode
  sortDescriptor: SortDescriptor
  onSortChange: (descriptor: SortDescriptor) => void
  selectedKey: string | null
  onSelect: (key: string) => void
  onEditPricing: (key: string) => void
  comparisonContextTokens: number | null
  selectedKeys: Selection
  onSelectionChange: (keys: Selection) => void
}) {
  // Memoized on their real inputs so DataTable's per-row cache holds across
  // selection clicks (see the DataTable docstring). rowClassName intentionally
  // depends on selectedKey: the drilled-row highlight must invalidate the
  // cache when the selection target changes.
  const columns = useMemo<DataTableColumn<ModelRow>[]>(() => {
    const comparisonLabel =
      comparisonContextTokens == null
        ? "Base"
        : `at ${formatContext(comparisonContextTokens)}`
    return [
      {
        id: "model",
        header: "Model",
        isRowHeader: true,
        allowsSorting: true,
        // The copy yields the provider-qualified key, which is what a caller sends
        // as `model`; the visible text drops the provider prefix.
        cell: (row) => (
          <CopyableValue
            value={row.key}
            label="model id"
            className="font-medium break-all"
          >
            {row.model}
            {/* `sr-only` is rendered (clipped), not hidden, so its text joins a range
              selection: without `select-none` a drag across this cell yields
              "gpt-4oopenai:gpt-4o". An own declaration outranks the inherited
              `user-select: text`, the same reasoning `select-text` uses above. */}
            <span className="sr-only select-none">{row.key}</span>
          </CopyableValue>
        ),
      },
      {
        id: "provider",
        header: "Provider",
        cell: (row) => <span className="text-muted">{row.provider}</span>,
      },
      {
        id: "modalities",
        header: "Modalities",
        // The category set's one instance in a table, which is why the artboard
        // puts it here rather than only in the detail panel: this is the form's
        // rendered specimen. An em dash where the metadata service reported
        // nothing, because unknown is an absence rather than "none".
        cell: (row) => {
          const input = row.inputModalities ?? []
          const output = row.outputModalities ?? []
          if (input.length === 0 && output.length === 0) {
            return <span className="text-muted">—</span>
          }
          return (
            <div className="flex flex-col gap-1">
              <ModalitySet label="In" values={input} />
              <ModalitySet label="Out" values={output} />
            </div>
          )
        },
      },
      {
        id: "input",
        header: (
          <span className="inline-flex items-center gap-1">
            {`${comparisonLabel} in / out / 1M`}
            <PricingInfo />
          </span>
        ),
        align: "end",
        allowsSorting: true,
        cell: (row) => {
          const rates = effectiveRatesAtContext(row, comparisonContextTokens)
          return (
            <PriceCell
              primary={rates.inputPrice}
              secondary={rates.outputPrice}
              rowKey={row.key}
              primaryLabel="input"
              secondaryLabel="output"
              onEdit={() => onEditPricing(row.key)}
            />
          )
        },
      },
      {
        id: "caching",
        header:
          comparisonContextTokens == null
            ? "Caching"
            : `Caching ${comparisonLabel}`,
        align: "end",
        cell: (row) => (
          <CachingCell
            rates={effectiveRatesAtContext(row, comparisonContextTokens)}
            rowKey={row.key}
            onEdit={() => onEditPricing(row.key)}
          />
        ),
      },
      {
        id: "policy",
        // Deliberately not shortened, unlike `Caching` beside it. `Pricing`
        // alone sits two columns from `Base in / out / 1M` and reads as a
        // second price column rather than as the policy that produced the
        // price, so it wraps instead. See the header comment in globals.css on
        // the overline spec: uppercase costs width, and a wrapped header costs
        // a glance where a misleading one costs a wrong conclusion.
        header: "Pricing policy",
        // Left-aligned, unlike the two numeric lanes before it, because it is a
        // status rather than a quantity. That adjacency is the one the
        // separation rule exists for: a right-aligned number and a left-aligned
        // label land their content on the same pixel, so `globals.css` gives
        // this lane explicit room on its left.
        cell: (row) => (
          <PricingPolicyCell row={row} onEdit={() => onEditPricing(row.key)} />
        ),
      },
    ]
  }, [comparisonContextTokens, onEditPricing])

  const rowClassName = useCallback(
    (row: ModelRow) =>
      row.key === selectedKey ? "bg-primary-subtle" : undefined,
    [selectedKey],
  )

  return (
    <TableScrollFrame className="otari-models-table">
      <DataTable
        ariaLabel="Models"
        columns={columns}
        rows={rows}
        getRowKey={getModelRowKey}
        isLoading={isLoading}
        emptyContent={empty}
        selectionMode="multiple"
        selectedKeys={selectedKeys}
        onSelectionChange={onSelectionChange}
        sortDescriptor={sortDescriptor}
        onSortChange={onSortChange}
        onRowAction={onSelect}
        rowClassName={rowClassName}
      />
    </TableScrollFrame>
  )
}

// A provider whose backend has no model-listing endpoint is not misconfigured, so
// it gets its own line: pointing the operator at their credentials would send them
// after a problem that isn't there (issue #447).
function DiscoveredErrors({
  providers,
  onPriceModel,
}: {
  // Typed from the API rather than restated: the discovery response carries
  // more per-provider detail than this banner reads, and `error` is optional on
  // the wire, not required-nullable as this once assumed.
  providers: DiscoverableProvider[]
  /** Opens the hand-pricing dialog seeded with the given key (may be a bare prefix). */
  onPriceModel: (prefill: string) => void
}) {
  if (providers.length === 0) {
    return null
  }
  const unreachable = providers.filter(
    (provider) => !provider.discovery_unsupported,
  )
  const noDiscovery = providers.filter(
    (provider) => provider.discovery_unsupported,
  )
  const names = (list: typeof providers) =>
    list.map((provider) => provider.provider).join(", ")
  const one = (list: typeof providers) => list.length === 1
  return (
    // The Sheet-2 banner form: a square dot and muted prose between rules, no
    // fill and no radius. Danger rather than caution because an unlistable
    // provider means models are missing from the table under it, which is a
    // wrong answer rather than a caveat.
    <Section
      className="border-y border-border py-3"
      contentClassName="flex items-start gap-3 text-sm text-muted"
    >
      <Dot className="mt-2 bg-danger" />
      <div className="flex flex-col gap-1">
        {unreachable.length > 0 ? (
          <span className="block">
            Could not list {names(unreachable)}. Check{" "}
            {one(unreachable) ? "that provider's" : "those providers'"}{" "}
            credentials in config.yml; {one(unreachable) ? "its" : "their"}{" "}
            models are missing from the list below.
          </span>
        ) : null}
        {noDiscovery.length > 0 ? (
          <span className="block">
            {names(noDiscovery)} {one(noDiscovery) ? "does" : "do"} not offer
            model discovery, so {one(noDiscovery) ? "its" : "their"} models are
            missing from the list below.{" "}
            {one(noDiscovery) ? "The provider" : "They"} may still serve
            requests. Price a model by its selector to meter it here, or declare
            the model ids {one(noDiscovery) ? "it serves" : "they serve"} under
            the <code>models:</code> key in config.yml to list them all.
            {/* Seeded with the prefix only when it is unambiguous: with several
              providers lacking discovery, guessing one would put the operator on
              the wrong provider without saying so. */}
            <Button
              size="sm"
              variant="outline"
              className="mt-2"
              onPress={() =>
                onPriceModel(
                  one(noDiscovery) ? `${noDiscovery[0].provider}:` : "",
                )
              }
            >
              Price a model
            </Button>
          </span>
        ) : null}
      </div>
    </Section>
  )
}

export function ModelsPage() {
  const models = useModels()
  const pricing = usePricing()
  const discoverable = useDiscoverableModels()
  const metadata = useModelMetadata()
  const settings = useSettings()

  const setPricing = useSetPricing()
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [pricingKey, setPricingKey] = useState<string | null>(null)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [sort, setSort] = useState<Sort>(readStoredSort)
  const selection = useTableSelection()
  const [bulkPriceOpen, setBulkPriceOpen] = useState(false)
  const [bulkPending, setBulkPending] = useState(false)
  const [bulkError, setBulkError] = useState<unknown>(undefined)
  // Non-null while the hand-pricing dialog is open, holding the key it opened
  // with: a searched selector, a provider prefix, or "" for a blank field.
  const [customPriceKey, setCustomPriceKey] = useState<string | null>(null)
  const [customPending, setCustomPending] = useState(false)
  const [customError, setCustomError] = useState<unknown>(undefined)

  useEffect(() => {
    try {
      window.localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify(sort))
    } catch {
      // Ignore storage errors (e.g. disabled localStorage); sort still applies.
    }
  }, [sort])

  const providerParam = useUrlValue("provider")
  // A provider clicked on the Providers page arrives as ?provider=<instance>,
  // pre-selecting that provider's filter so the list shows only their models.
  // An empty param (e.g. /models?provider=) collapses to "all"; an unknown one
  // is reset below once the catalog loads.
  const [providerFilter, setProviderFilter] = useState(providerParam || "all")
  const [pricingFilter, setPricingFilter] = useState("all")
  const [sourceFilter, setSourceFilter] = useState("all")
  const [capabilityFilter, setCapabilityFilter] = useState("all")
  const [minContext, setMinContext] = useState("0")
  const [maxInput, setMaxInput] = useState("")
  const [releaseFilter, setReleaseFilter] = useState("all")
  const [comparisonContext, setComparisonContext] = useState("")

  const metadataByKey = metadata.data?.models ?? {}
  const metadataAvailable = metadata.data?.available ?? false

  const discoverableKeys = useMemo(
    () =>
      new Set(
        (discoverable.data?.providers ?? []).flatMap((provider) =>
          provider.models.map((model) => model.key),
        ),
      ),
    [discoverable.data],
  )

  const changeSearch = (value: string) => {
    setSearch(value)
    setPage(0)
  }
  const changeFilter = (setter: (value: string) => void) => (value: string) => {
    setter(value)
    setPage(0)
  }

  const rows = useMemo<ModelRow[]>(() => {
    const configured = new Map(
      currentPricing(pricing.data ?? []).map((row) => [row.model_key, row]),
    )
    const result: ModelRow[] = []
    const seen = new Set<string>()

    const add = (
      key: string,
      model: string,
      provider: string,
      catalogRow?: ModelRow,
    ) => {
      if (seen.has(key)) {
        return
      }
      seen.add(key)
      const priced = configured.get(key)
      result.push({
        key,
        model: displayModelName(model, provider),
        provider,
        isDiscovered: discoverableKeys.has(key),
        contextWindow: catalogRow?.contextWindow ?? null,
        inputPrice: priced
          ? priced.input_price_per_million
          : (catalogRow?.inputPrice ?? null),
        outputPrice: priced
          ? priced.output_price_per_million
          : (catalogRow?.outputPrice ?? null),
        cacheReadPrice: priced
          ? priced.cache_read_price_per_million
          : (catalogRow?.cacheReadPrice ?? null),
        cacheWritePrice: priced
          ? priced.cache_write_price_per_million
          : (catalogRow?.cacheWritePrice ?? null),
        cacheWrite1hPrice: priced
          ? (priced.cache_write_1h_price_per_million ?? null)
          : (catalogRow?.cacheWrite1hPrice ?? null),
        pricingTiers: priced
          ? (priced.pricing_tiers ?? [])
          : (catalogRow?.pricingTiers ?? []),
        source: priced ? "configured" : (catalogRow?.source ?? "none"),
      })
    }

    for (const model of models.data?.data ?? []) {
      // Aliases are managed on their own page, not part of the model catalog.
      if (model.owned_by === ALIAS_OWNED_BY) {
        continue
      }
      const priceStatus: PriceSource =
        model.pricing_source === "default"
          ? "default"
          : model.pricing
            ? "configured"
            : "none"
      add(
        model.id,
        model.id,
        model.owned_by || providerFromModelKey(model.id),
        {
          key: model.id,
          model: model.id,
          provider: model.owned_by,
          contextWindow: model.context_window ?? null,
          inputPrice: model.pricing?.input_price_per_million ?? null,
          outputPrice: model.pricing?.output_price_per_million ?? null,
          cacheReadPrice: model.pricing?.cache_read_price_per_million ?? null,
          cacheWritePrice: model.pricing?.cache_write_price_per_million ?? null,
          cacheWrite1hPrice:
            model.pricing?.cache_write_1h_price_per_million ?? null,
          pricingTiers: (model.pricing?.pricing_tiers ?? []).filter(
            isPricingTier,
          ),
          source: priceStatus,
        },
      )
    }
    for (const key of configured.keys()) {
      // Gateway-run tools (otari:web_search, otari:code_execution) are priced per
      // call, not per million tokens, so they are managed on the Tools & Guardrails
      // page. Listing them here would put a per-call rate under columns labeled
      // "$ / 1M" and drag them into the price filters and the comparison chart,
      // where a rate stored as 10000 would rank as the costliest "model" served.
      if (key.startsWith(`${GATEWAY_TOOL_PRICING_PROVIDER}:`)) {
        continue
      }
      add(key, key, providerFromModelKey(key))
    }

    return result
  }, [models.data, pricing.data, discoverableKeys])

  const rowsByKey = useMemo(
    () => new Map(rows.map((row) => [row.key, row])),
    [rows],
  )

  // The model list, with the context window filled from models.dev when the
  // catalog did not supply one. Aliases live on their own page.
  const modelRows = useMemo<ModelRow[]>(() => {
    const out = rows.map((row) => ({
      ...row,
      contextWindow:
        row.contextWindow ?? metadataByKey[row.key]?.context_window ?? null,
      releaseDate: metadataByKey[row.key]?.release_date ?? null,
      inputModalities: metadataByKey[row.key]?.input_modalities ?? undefined,
      outputModalities: metadataByKey[row.key]?.output_modalities ?? undefined,
    }))
    const seen = new Set(out.map((row) => row.key))
    // These rows exist because discovery reaches a model the catalog did not list:
    // an alias target, which is withheld, or any discovered model at all while
    // `model_discovery` is off. With the genai-prices fallback on, such a model is
    // still metered, so "not priced" would be wrong; the rate is simply not knowable
    // from here. With the fallback off, unpriced really does mean unpriced.
    //
    // Tested against `false` rather than for truth so that the claim waits for the
    // answer: while settings are still loading the flag is undefined, and "not priced"
    // is the one of the two labels that would be asserting something.
    const unpricedSource: PriceSource =
      settings.data?.default_pricing === false ? "none" : "unknown"
    for (const provider of discoverable.data?.providers ?? []) {
      for (const model of provider.models) {
        if (seen.has(model.key)) {
          continue
        }
        seen.add(model.key)
        out.push({
          key: model.key,
          model: displayModelName(model.key, provider.provider),
          provider: provider.provider,
          isDiscovered: true,
          contextWindow: metadataByKey[model.key]?.context_window ?? null,
          releaseDate: metadataByKey[model.key]?.release_date ?? null,
          inputModalities:
            metadataByKey[model.key]?.input_modalities ?? undefined,
          outputModalities:
            metadataByKey[model.key]?.output_modalities ?? undefined,
          inputPrice: null,
          outputPrice: null,
          cacheReadPrice: null,
          cacheWritePrice: null,
          cacheWrite1hPrice: null,
          pricingTiers: [],
          source: unpricedSource,
        })
      }
    }
    return out
  }, [rows, discoverable.data, metadataByKey, settings.data?.default_pricing])

  const discoveredErrors = (discoverable.data?.providers ?? []).filter(
    (provider) => !provider.ok,
  )

  const providerOptions = useMemo(() => {
    const names = Array.from(
      new Set(modelRows.map((row) => row.provider)),
    ).sort((a, b) => a.localeCompare(b))
    return [
      { value: "all", label: "All providers" },
      ...names.map((name) => ({ value: name, label: name })),
    ]
  }, [modelRows])

  // A ?provider= value seeded from the URL may name a provider with no models,
  // or a stale/misspelled one. No option would match, so the select would name
  // the bogus provider back at you and the table would show zero rows, a
  // confusing dead end. Once the catalog has loaded (options beyond "all"),
  // drop an unknown value back to "all".
  useEffect(() => {
    if (providerFilter === "all" || providerOptions.length <= 1) {
      return
    }
    if (!providerOptions.some((option) => option.value === providerFilter)) {
      setProviderFilter("all")
    }
  }, [providerOptions, providerFilter])

  const query = search.trim().toLowerCase()
  const minContextValue = Number(minContext) || 0
  const maxInputValue =
    maxInput === "" ? Number.POSITIVE_INFINITY : Number(maxInput)
  const comparisonContextTokens =
    comparisonContext === "" ? null : Number(comparisonContext)
  const releaseCutoff =
    releaseFilter === "all"
      ? null
      : Date.now() - Number(releaseFilter) * RELEASE_WINDOW_MS

  const filteredModels = useMemo(() => {
    const matches = (row: ModelRow): boolean => {
      if (
        query &&
        !row.key.toLowerCase().includes(query) &&
        !row.provider.toLowerCase().includes(query)
      ) {
        return false
      }
      if (providerFilter !== "all" && row.provider !== providerFilter) {
        return false
      }
      if (pricingFilter === "configured" && row.source !== "configured") {
        return false
      }
      if (pricingFilter === "default" && row.source !== "default") {
        return false
      }
      if (pricingFilter === "priced" && row.inputPrice == null) {
        return false
      }
      if (pricingFilter === "unpriced" && row.inputPrice != null) {
        return false
      }
      if (sourceFilter === "discovered" && !row.isDiscovered) {
        return false
      }
      if (sourceFilter === "custom" && row.isDiscovered) {
        return false
      }
      if (capabilityFilter !== "all") {
        const capability = MODEL_FILTER_CAPABILITIES.find(
          (entry) => entry.value === capabilityFilter,
        )
        const meta = metadataByKey[row.key]
        if (!capability || !meta || !capability.test(meta)) {
          return false
        }
      }
      if (
        minContextValue > 0 &&
        (row.contextWindow == null || row.contextWindow < minContextValue)
      ) {
        return false
      }
      if (
        maxInputValue !== Number.POSITIVE_INFINITY &&
        (row.inputPrice == null || row.inputPrice > maxInputValue)
      ) {
        return false
      }
      if (releaseCutoff != null) {
        const released = row.releaseDate
          ? Date.parse(row.releaseDate)
          : Number.NaN
        if (Number.isNaN(released) || released < releaseCutoff) {
          return false
        }
      }
      return true
    }

    const compare = (a: ModelRow, b: ModelRow): number => {
      const dir = sort.dir === "asc" ? 1 : -1
      if (sort.col === "model") {
        return a.model.localeCompare(b.model) * dir
      }
      if (sort.col === "released") {
        // models.dev dates are ISO (YYYY-MM-DD), so lexical order is chronological.
        const ad = a.releaseDate ?? null
        const bd = b.releaseDate ?? null
        if (!ad && !bd) {
          return a.model.localeCompare(b.model)
        }
        if (!ad) {
          return 1
        }
        if (!bd) {
          return -1
        }
        return (
          (ad < bd ? -1 : ad > bd ? 1 : 0) * dir ||
          a.model.localeCompare(b.model)
        )
      }
      const pick = (row: ModelRow) =>
        sort.col === "input" ? row.inputPrice : row.outputPrice
      const av = pick(a)
      const bv = pick(b)
      if (av == null && bv == null) {
        return a.model.localeCompare(b.model)
      }
      if (av == null) {
        return 1
      }
      if (bv == null) {
        return -1
      }
      return (av - bv) * dir || a.model.localeCompare(b.model)
    }

    return modelRows.filter(matches).sort(compare)
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
  ])

  const total = filteredModels.length
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const clampedPage = Math.min(page, pageCount - 1)
  const start = clampedPage * pageSize
  const pageModels = filteredModels.slice(start, start + pageSize)

  const sortDescriptor: SortDescriptor = {
    column: sort.col,
    direction: sort.dir === "asc" ? "ascending" : "descending",
  }
  const onSortChange = (descriptor: SortDescriptor) => {
    setSort({
      col: String(descriptor.column) as SortCol,
      dir: descriptor.direction === "ascending" ? "asc" : "desc",
    })
    setPage(0)
  }
  // Stable so ModelTable's memoized columns (which close over it) survive
  // parent re-renders; the setter-callback form needs no dependencies.
  const onEditPricing = useCallback(
    (key: string) => setPricingKey((current) => (current === key ? null : key)),
    [],
  )

  // Selection targets the visible page; "select all matching" expands to every
  // filtered model so a bulk price can be applied to the whole result set.
  const selectableKeys = pageModels.map((row) => row.key)
  const selectedModelKeys = resolveSelectedIds(
    selection.selectedKeys,
    selectableKeys,
  )
  const allPageSelected =
    selectableKeys.length > 0 &&
    selectedModelKeys.length === selectableKeys.length
  const canSelectAllMatching =
    allPageSelected && total > selectedModelKeys.length
  const bulkTargetKeys = selection.allMatching
    ? filteredModels.map((row) => row.key)
    : selectedModelKeys
  const bulkCount = selection.allMatching ? total : selectedModelKeys.length
  const pricingRow = pricingKey
    ? (filteredModels.find((row) => row.key === pricingKey) ??
      rowsByKey.get(pricingKey) ??
      null)
    : null

  const runBulkPricing = async (rates: ManualRates) => {
    setBulkPending(true)
    setBulkError(undefined)
    try {
      for (const key of bulkTargetKeys) {
        await setPricing.mutateAsync({
          model_key: key,
          input_price_per_million: rates.input_price_per_million,
          output_price_per_million: rates.output_price_per_million,
          cache_read_price_per_million:
            rates.cache_read_price_per_million ?? null,
          cache_write_price_per_million:
            rates.cache_write_price_per_million ?? null,
          cache_write_1h_price_per_million: null,
          pricing_tiers: [],
        })
      }
      selection.clear()
      setBulkPriceOpen(false)
    } catch (error) {
      setBulkError(error)
    } finally {
      setBulkPending(false)
    }
  }

  // A backend with no /v1/models endpoint serves models the catalog never
  // lists, so the only way to meter them is a key typed by hand. The stored key
  // is what the server normalized, so the new row is selected by that rather
  // than by the raw input (a legacy provider/model form collapses onto
  // provider:model). Cache 1h rates and tiers are left out of the request so
  // re-pricing an existing key inherits them instead of clearing them.
  const priceCustomModel = async (rates: ManualRates, modelKey: string) => {
    setCustomPending(true)
    setCustomError(undefined)
    try {
      const created = await setPricing.mutateAsync({
        model_key: modelKey,
        input_price_per_million: rates.input_price_per_million,
        output_price_per_million: rates.output_price_per_million,
        cache_read_price_per_million:
          rates.cache_read_price_per_million ?? null,
        cache_write_price_per_million:
          rates.cache_write_price_per_million ?? null,
      })
      setCustomPriceKey(null)
      setSelectedKey(created.model_key)
    } catch (error) {
      setCustomError(error)
    } finally {
      setCustomPending(false)
    }
  }

  const modelsLoading =
    models.isLoading || pricing.isLoading || discoverable.isLoading
  const hasActiveFilters =
    query !== "" ||
    providerFilter !== "all" ||
    pricingFilter !== "all" ||
    sourceFilter !== "all" ||
    capabilityFilter !== "all" ||
    minContext !== "0" ||
    maxInput !== "" ||
    releaseFilter !== "all"
  // A provider that serves no model listing answers requests for models the
  // catalog never shows, so searching for one you just called successfully is
  // the moment the gap is felt. Offer to price it right there, seeded with what
  // was typed (from `search`, not the lowercased `query`: model ids are
  // case-sensitive) when that looks like a selector rather than a word.
  const searchedSelector = isValidModelKey(search) ? search.trim() : null
  const emptyModels = (
    <div className="flex flex-col items-center gap-2 py-2">
      <span>
        {hasActiveFilters
          ? "No models match your filters."
          : "No models yet. Add a provider on the Providers page."}
      </span>
      <span className="text-xs text-muted">
        A provider that serves no model listing still answers requests, so a
        model you can call may not be listed here.
      </span>
      <Button
        size="sm"
        variant="outline"
        onPress={() => setCustomPriceKey(searchedSelector ?? "")}
      >
        {searchedSelector
          ? `Price ${searchedSelector}`
          : "Price a model by hand"}
      </Button>
    </div>
  )

  // The row selected in the table, resolved against the filtered list (falling
  // back to any known row) to fill the detail panel.
  const selectedRow = selectedKey
    ? (filteredModels.find((row) => row.key === selectedKey) ??
      rowsByKey.get(selectedKey) ??
      null)
    : null
  return (
    <div className="flex flex-col gap-6">
      <header className="pb-1">
        <h1 className="text-display">Models</h1>
        <p className="mt-1 max-w-[620px] text-sm text-muted">
          Every model your providers can serve. Set a price on any model so
          budgets and usage tracking work.
        </p>
      </header>

      <ErrorBanner
        error={
          models.error ?? pricing.error ?? discoverable.error ?? metadata.error
        }
      />

      <div
        className={`grid gap-4 lg:items-start ${
          selectedRow ? "lg:grid-cols-[minmax(0,1fr)_360px]" : "grid-cols-1"
        }`}
      >
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <SearchInput
              value={search}
              onChange={changeSearch}
              placeholder="Search models…"
            />
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
                ...MODEL_FILTER_CAPABILITIES.map((entry) => ({
                  value: entry.value,
                  label: entry.label,
                })),
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

          <DiscoveredErrors
            providers={discoveredErrors}
            onPriceModel={setCustomPriceKey}
          />

          {selectedModelKeys.length > 0 ? (
            <BulkActionBar
              selectedCount={bulkCount}
              allMatching={selection.allMatching}
              matchingTotal={total}
              canSelectAllMatching={canSelectAllMatching}
              onSelectAllMatching={selection.enableAllMatching}
              onClear={selection.clear}
            >
              <Button
                size="sm"
                variant="primary"
                onPress={() => setBulkPriceOpen(true)}
              >
                Set pricing
              </Button>
            </BulkActionBar>
          ) : null}

          <ModelTable
            rows={pageModels}
            isLoading={modelsLoading}
            empty={emptyModels}
            sortDescriptor={sortDescriptor}
            onSortChange={onSortChange}
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
            onEditPricing={onEditPricing}
            comparisonContextTokens={comparisonContextTokens}
            selectedKeys={selection.selectedKeys}
            onSelectionChange={selection.onSelectionChange}
          />

          {pricingRow ? (
            <Section
              className="border-y border-border"
              contentClassName="flex flex-col"
            >
              <div className="flex items-center justify-between border-b border-border py-2">
                <h2 className="text-title">Edit pricing</h2>
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() => setPricingKey(null)}
                >
                  Close
                </Button>
              </div>
              <InlinePriceForm
                row={pricingRow}
                onClose={() => setPricingKey(null)}
              />
            </Section>
          ) : null}

          <TablePagination
            page={clampedPage}
            pageSize={pageSize}
            total={total}
            rowsOnPage={pageModels.length}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size)
              setPage(0)
            }}
            pageSizeOptions={[15, 25, 50]}
          />
        </div>

        {selectedRow ? (
          <aside className="lg:sticky lg:top-4">
            <ModelDetailPanel
              row={selectedRow}
              metadata={metadataByKey[selectedRow.key]}
              metadataAvailable={metadataAvailable}
              onClose={() => setSelectedKey(null)}
            />
          </aside>
        ) : null}
      </div>

      <SetPriceDialog
        isOpen={bulkPriceOpen}
        onOpenChange={setBulkPriceOpen}
        targetCount={bulkCount}
        isPending={bulkPending}
        error={bulkError}
        onSubmit={runBulkPricing}
        title="Set pricing"
        description={(count) =>
          `Apply these per-1M rates to ${count.toLocaleString()} selected ${
            count === 1 ? "model" : "models"
          }. This replaces each model's price; pricing tiers and the 1h cache rate are cleared. Edit a single model for tiers.`
        }
      />

      <SetPriceDialog
        isOpen={customPriceKey !== null}
        onOpenChange={(open) =>
          setCustomPriceKey(open ? (customPriceKey ?? "") : null)
        }
        isPending={customPending}
        error={customError}
        onSubmit={priceCustomModel}
        collectModelKey
        initialModelKey={customPriceKey ?? ""}
        title="Price a model"
        description={() =>
          "Meter a model the catalog does not list, for a provider that serves no model listing. Type the selector you send as model and its rates; the model then appears here as custom, and its usage is costed and counted against budgets."
        }
      />
    </div>
  )
}
