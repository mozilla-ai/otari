/**
 * Whether a key's requests skip its owner's budget. The deployment operator's
 * alone: a key's owner may not take its traffic out of their own budget.
 */

import { Checkbox } from "@/design-system/forms/Checkbox"

export function BudgetExemptToggle({
  checked,
  onChange,
}: {
  checked: boolean
  onChange: (value: boolean) => void
}) {
  // The consequence sits beside the control rather than inside its label: a
  // checkbox whose accessible name is a whole paragraph is read out in full on
  // every focus.
  return (
    <div className="flex flex-col gap-0.5 border border-control-border p-3">
      <Checkbox isSelected={checked} onChange={onChange}>
        <span className="font-medium text-foreground">Exempt from budget</span>
      </Checkbox>
      <p className="text-caption">
        Requests on this key are logged with their cost but never counted toward
        the owner’s budget or spend, and never blocked by it.
      </p>
    </div>
  )
}
