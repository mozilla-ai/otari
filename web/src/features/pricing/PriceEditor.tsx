// biome-ignore-all lint/a11y/noLabelWithoutControl: every label here wraps MoneyInput, which renders the input and takes its own ariaLabel. The rule cannot see through a component, and it still works on this file's native inputs.
import { Button } from "@heroui/react"
import { useState } from "react"

import type { PricingResponse, PricingTier } from "@/client"
import { useDeletePricing, useSetPricing } from "@/shared/api/pricing"
import { ConfirmButton } from "@/shared/components/actions/ConfirmButton"
import { errorMessage } from "@/shared/components/feedback/errorMessage"
import { INPUT_CLASS } from "@/shared/components/forms/inputClass"
import { formatRate } from "@/shared/helpers/format"

// The one place a deployment rate is edited. It used to sit in the Models
// detail panel, which made the page that compares prices the page that changes
// them (otari-ai#2095); it lives on Model pricing now and the catalog links here
// with the selector in hand.

function isValidPrice(value: string): boolean {
  const n = Number(value)
  return value.trim() !== "" && Number.isFinite(n) && n >= 0
}

function isValidOptionalPrice(value: string): boolean {
  return value.trim() === "" || isValidPrice(value)
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
  for (const tier of tiers) {
    const threshold = Number(tier.minInputTokens)
    if (
      !Number.isInteger(threshold) ||
      threshold <= 0 ||
      thresholds.has(threshold)
    ) {
      return false
    }
    thresholds.add(threshold)
    const rates = [
      tier.input,
      tier.output,
      tier.cacheRead,
      tier.cacheWrite,
      tier.cacheWrite1h,
    ]
    if (!rates.every(isValidOptionalPrice)) return false
    if (rates.every((rate) => rate.trim() === "")) return false
  }
  return true
}

function pricingTiers(tiers: EditablePricingTier[]): PricingTier[] {
  return tiers.map((tier) => ({
    min_input_tokens: Number(tier.minInputTokens),
    input_price_per_million: optionalPrice(tier.input),
    output_price_per_million: optionalPrice(tier.output),
    cache_read_price_per_million: optionalPrice(tier.cacheRead),
    cache_write_price_per_million: optionalPrice(tier.cacheWrite),
    cache_write_1h_price_per_million: optionalPrice(tier.cacheWrite1h),
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
      className={`w-28 text-right tabular-nums ${INPUT_CLASS}`}
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
          <div className="text-emphasis">Long-context price tiers</div>
          <p className="text-caption">
            At a threshold, listed rates replace the base rate for the whole
            request.
          </p>
        </div>
        <Button size="sm" variant="ghost" onPress={add}>
          Add tier
        </Button>
      </div>
      {tiers.map((tier) => (
        <div
          key={tier.id}
          className="flex flex-wrap items-end gap-2 border-t border-border pt-2"
        >
          <label className="flex flex-col gap-1 text-caption">
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
              className={`w-28 text-right tabular-nums ${INPUT_CLASS}`}
            />
          </label>
          {(
            [
              ["input", "Input", "Tier input price"],
              ["output", "Output", "Tier output price"],
              ["cacheRead", "Cache read", "Tier cache read price"],
              ["cacheWrite", "Cache write", "Tier cache write price"],
              ["cacheWrite1h", "1h write", "Tier 1 hour cache write price"],
            ] as const
          ).map(([field, label, ariaLabel]) => (
            <label key={field} className="flex flex-col gap-1 text-caption">
              {label}
              <MoneyInput
                value={tier[field]}
                onChange={(value) => update(tier.id, field, value)}
                ariaLabel={ariaLabel}
              />
            </label>
          ))}
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

function Row({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-caption">{label}</span>
      {children}
    </div>
  )
}

function Current({ label, value }: { label: string; value: number | null }) {
  return (
    <Row label={label}>
      <span className="text-body tabular-nums">
        {value == null ? "—" : `${formatRate(value)} / 1M`}
      </span>
    </Row>
  )
}

/**
 * The deployment rate for one selector: what is stored, and set, edit or reset.
 *
 * `current` is the row in force today, or undefined when the selector has no
 * stored rate (it may still be metered at a genai-prices default; that is not
 * this editor's to show). Keyed by the caller on `modelKey`, so switching
 * selectors remounts it with clean fields.
 */
export function PriceEditor({
  modelKey,
  current,
  onDone,
}: {
  modelKey: string
  current: PricingResponse | undefined
  onDone?: () => void
}) {
  const setPricing = useSetPricing()
  const deletePricing = useDeletePricing()
  const [editing, setEditing] = useState(current === undefined)
  const [input, setInput] = useState(
    current ? String(current.input_price_per_million) : "",
  )
  const [output, setOutput] = useState(
    current ? String(current.output_price_per_million) : "",
  )
  const [cacheRead, setCacheRead] = useState(
    current?.cache_read_price_per_million == null
      ? ""
      : String(current.cache_read_price_per_million),
  )
  const [cacheWrite, setCacheWrite] = useState(
    current?.cache_write_price_per_million == null
      ? ""
      : String(current.cache_write_price_per_million),
  )
  const [cacheWrite1h, setCacheWrite1h] = useState(
    current?.cache_write_1h_price_per_million == null
      ? ""
      : String(current.cache_write_1h_price_per_million),
  )
  const [tiers, setTiers] = useState<EditablePricingTier[]>(
    editableTiers(current?.pricing_tiers ?? []),
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
        model_key: modelKey,
        input_price_per_million: Number(input),
        output_price_per_million: Number(output),
        cache_read_price_per_million: optionalPrice(cacheRead),
        cache_write_price_per_million: optionalPrice(cacheWrite),
        cache_write_1h_price_per_million: optionalPrice(cacheWrite1h),
        pricing_tiers: pricingTiers(tiers),
        unit:
          current?.unit === "requests" || current?.unit === "images"
            ? current.unit
            : "tokens",
      },
      {
        onSuccess: () => {
          setEditing(false)
          onDone?.()
        },
      },
    )
  }

  if (editing) {
    return (
      <div className="flex flex-col gap-2">
        <Row label="Input $ / 1M">
          <MoneyInput
            value={input}
            onChange={setInput}
            ariaLabel={`Input price for ${modelKey}`}
          />
        </Row>
        <Row label="Output $ / 1M">
          <MoneyInput
            value={output}
            onChange={setOutput}
            ariaLabel={`Output price for ${modelKey}`}
          />
        </Row>
        <Row label="Cache read $ / 1M">
          <MoneyInput
            value={cacheRead}
            onChange={setCacheRead}
            ariaLabel={`Cache read price for ${modelKey}`}
          />
        </Row>
        <Row label="Cache write $ / 1M">
          <MoneyInput
            value={cacheWrite}
            onChange={setCacheWrite}
            ariaLabel={`Cache write price for ${modelKey}`}
          />
        </Row>
        <Row label="1h cache write $ / 1M">
          <MoneyInput
            value={cacheWrite1h}
            onChange={setCacheWrite1h}
            ariaLabel={`1 hour cache write price for ${modelKey}`}
          />
        </Row>
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
            onPress={() => (current ? setEditing(false) : onDone?.())}
          >
            Cancel
          </Button>
        </div>
        {setPricing.error ? (
          <span className="text-caption text-danger">
            {errorMessage(setPricing.error)}
          </span>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <Current label="Input" value={current?.input_price_per_million ?? null} />
      <Current
        label="Output"
        value={current?.output_price_per_million ?? null}
      />
      <Current
        label="Cache read"
        value={current?.cache_read_price_per_million ?? null}
      />
      <Current
        label="Cache write"
        value={current?.cache_write_price_per_million ?? null}
      />
      <Current
        label="1h cache write"
        value={current?.cache_write_1h_price_per_million ?? null}
      />
      <Row label="Context tiers">
        <span className="text-body tabular-nums">
          {current?.pricing_tiers.length
            ? `${current.pricing_tiers.length} configured`
            : "—"}
        </span>
      </Row>
      <div className="flex items-center gap-2 pt-1">
        <Button size="sm" variant="ghost" onPress={() => setEditing(true)}>
          Edit price
        </Button>
        {current ? (
          <ConfirmButton
            confirmLabel="Reset to default"
            isPending={deletePricing.isPending}
            onConfirm={() =>
              deletePricing.mutate(modelKey, { onSuccess: () => onDone?.() })
            }
          >
            Reset price
          </ConfirmButton>
        ) : null}
        {deletePricing.error ? (
          <span className="text-caption text-danger">
            {errorMessage(deletePricing.error)}
          </span>
        ) : null}
      </div>
    </div>
  )
}
