import { Input, Label, TextField } from "@heroui/react"
import { useState } from "react"

import type { OrganizationPricingOverride } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useCreateOrganizationPricing,
  useReplaceOrganizationPricing,
} from "@/shared/api/pricing"

import {
  findOverlapping,
  isValidModelKey,
  parseRate,
  periodBlockedReason,
} from "./pricingOverride"

// The form behind both Add and Edit. One component rather than two, because the
// only difference is whether the model key is editable: the endpoint replaces a
// row wholesale, so an edit sends every field exactly as an add does, and a
// shared form is what keeps that true.

export interface PricingOverrideDraft {
  model_key: string
  input_price_per_million: number
  output_price_per_million: number
  cache_read_price_per_million: number | null
  cache_write_price_per_million: number | null
  cache_write_1h_price_per_million: number | null
  effective_from: string | null
  effective_to: string | null
}

interface RateFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  isRequired?: boolean
  description?: string
  /** Takes focus on mount, for the instance that is the form's first field. */
  autoFocus?: boolean
}

function RateField({
  label,
  value,
  onChange,
  isRequired,
  description,
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
      {description ? <span className="text-caption">{description}</span> : null}
    </TextField>
  )
}

// A datetime-local value, which is what the two period inputs exchange. Rendered
// from an ISO instant in the browser's zone, and read back as one.
function toLocalInput(iso: string | null | undefined): string {
  if (!iso) return ""
  const parsed = new Date(iso)
  if (Number.isNaN(parsed.getTime())) return ""
  const offsetMs = parsed.getTimezoneOffset() * 60_000
  return new Date(parsed.getTime() - offsetMs).toISOString().slice(0, 16)
}

function fromLocalInput(value: string): string | null {
  const trimmed = value.trim()
  if (trimmed === "") return null
  const parsed = new Date(trimmed)
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString()
}

function rateToInput(value: number | null | undefined): string {
  return value === null || value === undefined ? "" : String(value)
}

export interface PricingOverrideDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The row being edited; absent means this is an add. */
  editing?: OrganizationPricingOverride
  /** Every stored override, so an overlapping period is refused before the request. */
  existing: readonly OrganizationPricingOverride[]
  /** Called once a save has landed, so the caller can close this. */
  onSaved: () => void
}

export function PricingOverrideDialog({
  isOpen,
  onOpenChange,
  editing,
  existing,
  onSaved,
}: PricingOverrideDialogProps) {
  // Below the caller's key with the draft, so a refused save cannot greet the
  // next open (feedback.md: the component that renders the FormDialog owns the
  // draft *and* its mutation).
  const create = useCreateOrganizationPricing()
  const replace = useReplaceOrganizationPricing()
  const save = (draft: PricingOverrideDraft) => {
    const onDone = { onSuccess: onSaved }
    if (editing) {
      // model_key is absent from the update body: the endpoint refuses to
      // repoint an override at another model.
      const { model_key: _unused, ...rest } = draft
      // The endpoint requires a start on a replacement, so that an omitted one
      // cannot silently move a stored period to the present. The form blocks a
      // blank start while editing; this narrows the type and is the belt to
      // that brace.
      if (rest.effective_from === null) return
      replace.mutate(
        {
          id: editing.id,
          body: { ...rest, effective_from: rest.effective_from },
        },
        onDone,
      )
      return
    }
    create.mutate(draft, onDone)
  }
  // Seeded on mount only, because the caller remounts this on each open. Not a
  // nicety: these values set money, and inheriting the last row's rates into a
  // different model is the expensive kind of mistake.
  const seed = {
    modelKey: editing?.model_key ?? "",
    input: rateToInput(editing?.input_price_per_million),
    output: rateToInput(editing?.output_price_per_million),
    cacheRead: rateToInput(editing?.cache_read_price_per_million),
    cacheWrite: rateToInput(editing?.cache_write_price_per_million),
    cacheWrite1h: rateToInput(editing?.cache_write_1h_price_per_million),
    from: toLocalInput(editing?.effective_from),
    to: toLocalInput(editing?.effective_to),
  }
  const [modelKey, setModelKey] = useState(seed.modelKey)
  const [input, setInput] = useState(seed.input)
  const [output, setOutput] = useState(seed.output)
  const [cacheRead, setCacheRead] = useState(seed.cacheRead)
  const [cacheWrite, setCacheWrite] = useState(seed.cacheWrite)
  const [cacheWrite1h, setCacheWrite1h] = useState(seed.cacheWrite1h)
  const [from, setFrom] = useState(seed.from)
  const [to, setTo] = useState(seed.to)
  // One predicate naming every field, so what "unsaved" means cannot drift
  // from what the form holds.
  const { isDirty } = useDirtySnapshot({
    modelKey,
    input,
    output,
    cacheRead,
    cacheWrite,
    cacheWrite1h,
    from,
    to,
  })

  const inputRate = parseRate(input)
  const outputRate = parseRate(output)
  const cacheReadRate = parseRate(cacheRead)
  const cacheWriteRate = parseRate(cacheWrite)
  const cacheWrite1hRate = parseRate(cacheWrite1h)

  const keyInvalid = !isValidModelKey(modelKey)
  const periodReason = periodBlockedReason(from, to)
  const fromMs = from.trim() === "" ? Date.now() : Date.parse(from)
  const toMs = to.trim() === "" ? undefined : Date.parse(to)
  const clash =
    keyInvalid || periodReason !== undefined
      ? undefined
      : findOverlapping(existing, {
          modelKey: modelKey.trim(),
          from: fromMs,
          to: toMs,
          excludeId: editing?.id,
        })

  const ratesInvalid =
    inputRate === undefined ||
    Number.isNaN(inputRate) ||
    outputRate === undefined ||
    Number.isNaN(outputRate) ||
    Number.isNaN(cacheReadRate ?? 0) ||
    Number.isNaN(cacheWriteRate ?? 0) ||
    Number.isNaN(cacheWrite1hRate ?? 0)

  // A replacement states the whole row, so the endpoint requires a start: an
  // omitted one would otherwise be defaulted to now and move a stored period.
  const startRequired = editing !== undefined && from.trim() === ""

  const blockedReason = keyInvalid
    ? modelKey.trim() === ""
      ? undefined
      : "A rate is stored under a 'provider:model' key, so it needs the provider prefix."
    : startRequired
      ? "An edit needs a start. Leaving it blank would move this override's period to now."
      : (periodReason ??
        (clash
          ? `This period overlaps an override already stored for ${clash.model_key}. Change the period, or edit that one instead.`
          : undefined))

  const invalid =
    keyInvalid || ratesInvalid || startRequired || blockedReason !== undefined

  const submit = () => {
    if (invalid || inputRate === undefined || outputRate === undefined) return
    save({
      model_key: modelKey.trim(),
      input_price_per_million: inputRate,
      output_price_per_million: outputRate,
      cache_read_price_per_million: Number.isNaN(cacheReadRate ?? 0)
        ? null
        : (cacheReadRate ?? null),
      cache_write_price_per_million: Number.isNaN(cacheWriteRate ?? 0)
        ? null
        : (cacheWriteRate ?? null),
      cache_write_1h_price_per_million: Number.isNaN(cacheWrite1hRate ?? 0)
        ? null
        : (cacheWrite1hRate ?? null),
      effective_from: fromLocalInput(from),
      effective_to: fromLocalInput(to),
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title={editing ? "Edit rate override" : "New rate override"}
      description="What this organization pays for a model, above the deployment's own price list. Requests in the period below are billed at these rates; a model with no override here keeps being priced by the deployment."
      submitLabel={editing ? "Save override" : "Add override"}
      onSubmit={submit}
      isPending={create.isPending || replace.isPending}
      isSubmitDisabled={invalid}
      isDirty={isDirty}
      error={editing ? replace.error : create.error}
    >
      {editing ? (
        <div className="flex flex-col gap-1">
          <span className="text-body">Model</span>
          <code className="font-mono text-caption">{editing.model_key}</code>
          <span className="text-caption">
            A model cannot be changed here. Delete this override and add one for
            the other model.
          </span>
        </div>
      ) : (
        <Field
          label="Model key"
          value={modelKey}
          onChange={setModelKey}
          placeholder="provider:model"
          isRequired
          autoFocus
          description="For example openai:gpt-4o. A provider instance name works too."
        />
      )}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <RateField
          label="Input, per 1M tokens"
          value={input}
          onChange={setInput}
          isRequired
          // The first field on the edit path: the model key's own `Field`, which
          // carries `autoFocus` on the add path, is replaced by a read-only
          // block there, and focus was landing on the frame's Close control.
          autoFocus={editing !== undefined}
        />
        <RateField
          label="Output, per 1M tokens"
          value={output}
          onChange={setOutput}
          isRequired
        />
        <RateField
          label="Cache read, per 1M tokens"
          value={cacheRead}
          onChange={setCacheRead}
          description="Leave blank to price cached reads as fresh input."
        />
        <RateField
          label="Cache write, per 1M tokens"
          value={cacheWrite}
          onChange={setCacheWrite}
          description="Leave blank to price cache writes as fresh input."
        />
        <RateField
          label="Cache write, 1 hour TTL"
          value={cacheWrite1h}
          onChange={setCacheWrite1h}
          description="Anthropic's longer cache TTL. Blank falls back to the ordinary cache-write rate."
        />
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <TextField
          value={from}
          onChange={setFrom}
          isRequired={editing !== undefined}
          className="flex flex-col gap-1"
        >
          <Label className="text-body">Applies from</Label>
          <Input type="datetime-local" />
          <span className="text-caption">
            {editing
              ? "Required when editing: a replacement states the whole period."
              : "Blank starts it now."}
          </span>
        </TextField>
        <TextField value={to} onChange={setTo} className="flex flex-col gap-1">
          <Label className="text-body">Applies until</Label>
          <Input type="datetime-local" />
          <span className="text-caption">
            Blank leaves it open ended. The end is exclusive, so the next period
            may start at the same moment.
          </span>
        </TextField>
      </div>
      {blockedReason ? (
        <p className="text-sm text-danger">{blockedReason}</p>
      ) : null}
    </FormDialog>
  )
}
