import { Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

// Per-1M rates entered by an operator to reprice imported usage rows. Input and
// output are required; the cache rates are optional (blank folds those tokens
// into the fresh-input charge, matching how unset cache pricing behaves).
export interface ManualRates {
  input_price_per_million: number
  output_price_per_million: number
  cache_read_price_per_million?: number
  cache_write_price_per_million?: number
}

interface RateFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  isRequired?: boolean
  autoFocus?: boolean
}

function RateField({
  label,
  value,
  onChange,
  isRequired,
  autoFocus,
}: RateFieldProps) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isRequired={isRequired}
      className="flex flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <Input inputMode="decimal" placeholder="0.00" autoFocus={autoFocus} />
    </TextField>
  )
}

function parseRate(value: string): number | null {
  const trimmed = value.trim()
  if (trimmed === "") return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : Number.NaN
}

// A pricing row is only ever read back under a `prefix:model` selector, so a key
// with no provider or instance prefix would store a price nothing bills against
// (see normalize_pricing_key in services/provider_kwargs.py). Accept the legacy
// slash form too; the backend collapses it onto the colon form.
export function isValidModelKey(value: string): boolean {
  return /^[^\s:/]+[:/][^\s]+$/.test(value.trim())
}

export interface SetPriceDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** How many rows the price will be applied to, for the dialog copy. */
  targetCount?: number
  /**
   * Saves the rates. Awaited, and a rejection is reported inside this dialog:
   * the pending and error state live here, below the caller's key, so a refused
   * save cannot greet the next open (feedback.md). The callers each own a
   * different endpoint, which is why the mutation itself stays with them.
   */
  onSubmit: (rates: ManualRates, modelKey: string) => Promise<unknown>
  /** Dialog heading; defaults to "Set price". */
  title?: string
  /**
   * The submit's label, which is also its trigger's, word for word
   * (actions.md). Defaults to "Set price".
   */
  submitLabel?: string
  /** Body copy explaining what the rates apply to; a sensible usage default is used when omitted. */
  description?: (count: number) => string
  /**
   * Also collect the model key the rates apply to, for pricing a model that is
   * not in the catalog (a provider without model discovery). The trimmed key
   * is passed to `onSubmit`; without this the second argument is an empty string.
   */
  collectModelKey?: boolean
  /**
   * Seeds the model key each time the dialog opens (a selector taken from a
   * search box, a logged request, or a provider prefix). Only read with
   * `collectModelKey`.
   */
  initialModelKey?: string
}

const defaultDescription = (count: number): string =>
  `Recompute cost for ${count.toLocaleString()} imported ${
    count === 1 ? "row" : "rows"
  } from each row's own token counts at these per-1M rates. Enforced gateway rows are never affected.`

export function SetPriceDialog({
  isOpen,
  onOpenChange,
  targetCount = 0,
  onSubmit,
  submitLabel = "Set price",
  title = "Set price",
  description = defaultDescription,
  collectModelKey = false,
  initialModelKey = "",
}: SetPriceDialogProps) {
  // Seeded on mount only, because the caller remounts this on each open.
  // Reopening for a different selection must not inherit the last rates, which
  // is a real footgun when the values set money.
  const [modelKey, setModelKey] = useState(initialModelKey)
  const [input, setInput] = useState("")
  const [output, setOutput] = useState("")
  const [cacheRead, setCacheRead] = useState("")
  const [cacheWrite, setCacheWrite] = useState("")
  // One predicate naming every field, so what "unsaved" means cannot drift
  // from what the form holds.
  const { isDirty } = useDirtySnapshot({
    modelKey,
    input,
    output,
    cacheRead,
    cacheWrite,
  })

  const inputRate = parseRate(input)
  const outputRate = parseRate(output)
  const cacheReadRate = parseRate(cacheRead)
  const cacheWriteRate = parseRate(cacheWrite)

  const keyInvalid = collectModelKey && !isValidModelKey(modelKey)

  const invalid =
    keyInvalid ||
    inputRate === null ||
    Number.isNaN(inputRate) ||
    outputRate === null ||
    Number.isNaN(outputRate) ||
    Number.isNaN(cacheReadRate ?? 0) ||
    Number.isNaN(cacheWriteRate ?? 0)

  // Owned here rather than by the caller: below its key, so both reset with the
  // draft on the next open.
  const [isSaving, setIsSaving] = useState(false)
  const [failure, setFailure] = useState<unknown>(undefined)

  const submit = () => {
    if (invalid || inputRate === null || outputRate === null) return
    if (isSaving) return
    setFailure(undefined)
    setIsSaving(true)
    onSubmit(
      {
        input_price_per_million: inputRate,
        output_price_per_million: outputRate,
        ...(cacheReadRate !== null && !Number.isNaN(cacheReadRate)
          ? { cache_read_price_per_million: cacheReadRate }
          : {}),
        ...(cacheWriteRate !== null && !Number.isNaN(cacheWriteRate)
          ? { cache_write_price_per_million: cacheWriteRate }
          : {}),
      },
      modelKey.trim(),
    )
      .catch((error: unknown) => setFailure(error))
      .finally(() => setIsSaving(false))
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title={title}
      description={description(targetCount)}
      submitLabel={submitLabel}
      onSubmit={submit}
      isPending={isSaving}
      isSubmitDisabled={invalid}
      isDirty={isDirty}
      error={failure}
    >
      {collectModelKey ? (
        <Field
          label="Model key"
          value={modelKey}
          onChange={setModelKey}
          placeholder="provider:model"
          isRequired
          autoFocus
          description={
            modelKey.trim() !== "" && keyInvalid
              ? "Include the provider or instance prefix, as in ollama:llama3.2."
              : "The selector callers send as model, prefix included (for example vllm:mistral-small)."
          }
        />
      ) : null}
      <div className="grid gap-3 sm:grid-cols-2">
        <RateField
          label="Input $ / 1M"
          value={input}
          onChange={setInput}
          isRequired
          autoFocus={!collectModelKey}
        />
        <RateField
          label="Output $ / 1M"
          value={output}
          onChange={setOutput}
          isRequired
        />
        <RateField
          label="Cache read $ / 1M"
          value={cacheRead}
          onChange={setCacheRead}
        />
        <RateField
          label="Cache write $ / 1M"
          value={cacheWrite}
          onChange={setCacheWrite}
        />
      </div>
      <InfoBanner tone="info">
        Leave a cache rate blank to bill those tokens at the input rate.
      </InfoBanner>
    </FormDialog>
  )
}
