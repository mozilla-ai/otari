import { useState } from "react"

import type { HostedModel, HostedProvider } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  COLUMN_PRICING_UNIT_OPTIONS,
  hasOutputRate,
  PRICING_UNIT_DESCRIPTION,
  pricingUnitOf,
  unitOutputRateLabel,
  unitRateLabel,
} from "@/features/models/pricingUnit"
import {
  useAddHostedModel,
  useHostedAvailableModels,
  useUpdateHostedModel,
} from "@/shared/api/hostedProviders"
import { allowsFieldAutoFocus } from "@/shared/helpers/autoFocus"
import {
  canSubmitDraft,
  createBody,
  draftFrom,
  EMPTY_DRAFT,
  PRICE_ERROR,
  type PriceDraft,
  type PriceKey,
  readDraft,
  repriceBody,
} from "./modelPriceDraft"

// The form that offers a model on a hosted provider, and the one that reprices
// it: one component, seeded from the row when editing.
//
// Offering picks from what the provider says it serves, and accepts a typed id
// when the provider will not say: its silence must not block an operator who
// knows the model's name. The rate fields are laid out as the organization rate
// editor lays them out, Priced per included; the rules behind them live in
// `modelPriceDraft`.

// A popover cannot usefully render a provider's whole list; typing narrows it.
const MAX_OPTIONS = 50

export function HostedModelForm({
  provider,
  editing,
  isOpen,
  onClose,
  onSaved,
}: {
  provider: HostedProvider
  /** The model being repriced, or undefined when the form is offering one. */
  editing: HostedModel | undefined
  isOpen: boolean
  onClose: () => void
  onSaved: () => void
}) {
  const add = useAddHostedModel(provider.name)
  const reprice = useUpdateHostedModel(provider.name)
  const available = useHostedAvailableModels(
    provider.name,
    isOpen && editing === undefined,
  )
  const [draft, setDraft] = useState<PriceDraft>(() =>
    editing ? draftFrom(editing) : EMPTY_DRAFT,
  )
  const [query, setQuery] = useState("")
  // Repricing opens on the first rate, the field it is for. Not after a unit
  // change: that swaps the rate fields, and a remounted field must not pull
  // focus out of the radio group the operator is still in.
  const [unitChanged, setUnitChanged] = useState(false)
  const focusFirstRate = editing !== undefined && !unitChanged

  const { isDirty } = useDirtySnapshot(draft)
  const { parsed } = readDraft(draft)
  // This form sends rate columns, which the gateway stores per second on audio
  // input, so a model priced per second of video is not repriced here: the
  // save would move its price off the slot video billing reads.
  const pricedPerSecond =
    editing !== undefined && pricingUnitOf(editing.unit) === "second"
  const canSubmit =
    !pricedPerSecond && canSubmitDraft(draft, editing !== undefined)

  const saved = () => {
    onSaved()
    onClose()
  }

  const submit = () => {
    if (!canSubmit) return
    if (editing) {
      reprice.mutate(
        { modelId: editing.id, body: repriceBody(draft) },
        { onSuccess: saved },
      )
      return
    }
    add.mutate(createBody(draft), { onSuccess: saved })
  }

  const listed = available.data?.models ?? []
  const options = listed
    .filter((id) => !query || id.toLowerCase().includes(query.toLowerCase()))
    .slice(0, MAX_OPTIONS)
    .map((id) => ({ value: id, label: id }))
  const emptyMessage = available.isLoading
    ? "Asking the provider what it serves…"
    : available.data?.error
      ? `${available.data.error} Type the model id; it is sent exactly as typed either way.`
      : available.data?.discovery_unsupported
        ? "This provider publishes no model list. Type the model id."
        : available.isError
          ? "The model list could not be read. Type the model id."
          : "The provider listed no models."

  // Input and output are required together, so neither is marked required on
  // an offer, where both may be left blank to serve at the catalog's rates.
  const rateField = (label: string, key: PriceKey, description?: string) => (
    <Field
      label={label}
      value={draft[key]}
      onChange={(value) => setDraft({ ...draft, [key]: value })}
      inputMode="decimal"
      placeholder="0.00"
      autoFocus={allowsFieldAutoFocus() && focusFirstRate && key === "input"}
      isRequired={
        editing !== undefined &&
        (key === "input" || (key === "output" && draft.unit === "token"))
      }
      isInvalid={!parsed[key].ok}
      errorMessage={parsed[key].ok ? undefined : PRICE_ERROR}
      description={description}
    />
  )

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title={editing ? "Edit model price" : "Add model"}
      description={
        editing
          ? `What this deployment charges for the model on ${provider.name}. Requests are billed at this rate from the moment it is saved.`
          : `What ${provider.name} serves here, and at what price. Leave the rates blank to serve at the catalog's rates.`
      }
      submitLabel={editing ? "Save" : "Add model"}
      onSubmit={submit}
      isPending={add.isPending || reprice.isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={add.error ?? reprice.error}
    >
      {editing ? (
        <div className="flex flex-col gap-1">
          <span className="text-body">Model</span>
          <code className="font-mono text-caption">{editing.model}</code>
        </div>
      ) : (
        <ComboBoxField
          label="Model"
          value={draft.model}
          onChange={(model) => setDraft({ ...draft, model })}
          onQueryChange={setQuery}
          options={options}
          allowsCustomValue
          isRequired
          isSourceEmpty={listed.length === 0}
          emptyMessage={emptyMessage}
          noMatchesMessage="No model the provider lists matches what you typed. It is offered exactly as typed."
          description="The id as the provider spells it, with no provider prefix."
        />
      )}

      {pricedPerSecond ? (
        <InfoBanner tone="info">
          This model is priced per second of video, which this form cannot set
          yet. Send its price to <code>POST /api/v1/pricing</code> as a{" "}
          <code>rates</code> block instead.
        </InfoBanner>
      ) : (
        <>
          <RadioGroup
            label="Priced per"
            orientation="horizontal"
            value={draft.unit}
            onChange={(value) => {
              setUnitChanged(true)
              setDraft({ ...draft, unit: pricingUnitOf(value) })
            }}
            options={COLUMN_PRICING_UNIT_OPTIONS}
            description={PRICING_UNIT_DESCRIPTION}
          />
          {draft.unit === "token" ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {rateField(
                "Input, per 1M tokens",
                "input",
                editing ? undefined : "Set with output, or leave both blank.",
              )}
              {rateField("Output, per 1M tokens", "output")}
              {rateField(
                "Cache read, per 1M tokens",
                "cacheRead",
                "Leave blank to price cached reads as fresh input.",
              )}
              {rateField(
                "Cache write, per 1M tokens",
                "cacheWrite",
                "Leave blank to price cache writes as fresh input.",
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {rateField(
                unitRateLabel(draft.unit),
                "input",
                editing
                  ? undefined
                  : "Leave blank to serve at the catalog's rate.",
              )}
              {hasOutputRate(draft.unit)
                ? rateField(
                    unitOutputRateLabel(draft.unit),
                    "output",
                    "Leave blank where the model makes no audio.",
                  )
                : null}
            </div>
          )}
        </>
      )}
    </FormDialog>
  )
}
