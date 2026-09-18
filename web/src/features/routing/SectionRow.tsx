import type { ReactNode } from "react"

import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { useModelCatalog } from "@/features/models/ModelComboBox"

/**
 * One repeated row of a policy section, with the model catalog's hint under the
 * whole row rather than under the picker inside it.
 *
 * The hint is a sentence ("Could not list models for X. Check that provider's
 * credentials, ..."), and at this dialog's width it wraps. A wrapped message
 * makes its field taller than the siblings it shares an `items-end` row with,
 * which lifts that field's input line clear of theirs: measured at 40px on the
 * pool rows and 20px on the chain. `web/design/forms.md` ("Control rows") names
 * the break and this remedy. The picker keeps its empty caption line, so every
 * child of the row still reserves exactly one, and the hint is announced with
 * the input through `describedBy` because text outside a field never reaches
 * its description slot.
 */
export function SectionRow({
  id,
  modelValue,
  children,
}: {
  id: string
  modelValue: string
  children: ReactNode
}) {
  const { hint } = useModelCatalog(modelValue)
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-end gap-3">{children}</div>
      {hint ? (
        <FieldMessages shouldReserve={false}>
          <span id={id} className="text-muted">
            {hint}
          </span>
        </FieldMessages>
      ) : null}
    </div>
  )
}
