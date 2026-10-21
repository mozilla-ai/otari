import { useState } from "react"

import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { useSetPricing } from "@/shared/api/pricing"

import { parsePerCallPrice, type ToolRateRow, toPerCall } from "./toolRates"

const PRICE_ERROR = "An amount in dollars, such as 0.01."

// Sets the rate a tool row is billed at, as the hosted-model form reprices a
// model: a new version effective now, so requests already billed keep theirs.
// There is no blank: `/pricing` writes a rate and cannot unprice a key, and 0
// is the price of a deliberately free tool.
export function ToolRateForm({
  row,
  current,
  onClose,
}: {
  row: ToolRateRow
  /** The stored per-million rate in force, or `undefined` while unpriced. */
  current: number | undefined
  onClose: () => void
}) {
  const setPricing = useSetPricing()
  const [price, setPrice] = useState(() =>
    current === undefined ? "" : String(toPerCall(current)),
  )
  const { isDirty } = useDirtySnapshot(price)
  const stored = parsePerCallPrice(price)
  const isInvalid = price.trim() !== "" && stored === undefined

  const submit = () => {
    if (stored === undefined) return
    setPricing.mutate(
      {
        kind: row.kind,
        model_key: row.modelKey,
        input_price_per_million: stored,
        // A call is one unit; there is no output side of it to price.
        output_price_per_million: 0,
        unit: "requests",
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="Edit tool price"
      description={`What this deployment charges for each successful call of ${row.label}. Calls are billed at this rate from the moment it is saved.`}
      submitLabel="Save"
      onSubmit={submit}
      isPending={setPricing.isPending}
      isSubmitDisabled={stored === undefined}
      isDirty={isDirty}
      error={setPricing.error}
    >
      <div className="flex flex-col gap-1">
        <span className="text-body">Priced under</span>
        <code className="font-mono text-caption">{row.modelKey}</code>
      </div>
      <Field
        label="Price per call, USD"
        value={price}
        onChange={(next) => {
          setPrice(next)
          // The failure was about the value it was sent with, not this one.
          if (setPricing.isError) setPricing.reset()
        }}
        inputMode="decimal"
        placeholder="0.00"
        autoFocus
        isRequired
        isInvalid={isInvalid}
        errorMessage={isInvalid ? PRICE_ERROR : undefined}
        description="0 prices the tool as free. A call its provider reports a cost for is billed at that cost instead."
      />
    </FormDialog>
  )
}
