import { Select } from "@/design-system/forms/Select"
import {
  budgetLabeler,
  hasBudgetName,
  unnamedBudgetLabel,
} from "@/features/budgets/budgetLabel"
import type { CappingBudget } from "@/features/keys/KeyBudgetField"

/** The value for "no existing budget", which no budget id can be. */
export const OWN_LIMIT = ""

/**
 * Which existing budget a new key is capped by, or none. Offered to whoever
 * production lets cap a key (`useKeyBudgets().canCap`), from the budgets that
 * caller can already read.
 */
export function ExistingBudgetSelect({
  value,
  onChange,
  budgets,
  isLoading,
  ownLimitLabel,
}: {
  value: string
  onChange: (budgetId: string) => void
  budgets: readonly CappingBudget[]
  isLoading: boolean
  /** What the empty choice means here: a limit typed in below, or no budget. */
  ownLimitLabel: string
}) {
  const nameBudget = budgetLabeler(budgets)
  return (
    <Select
      label="Budget"
      value={value}
      onChange={onChange}
      isDisabled={isLoading}
      options={[
        { value: OWN_LIMIT, label: ownLimitLabel },
        ...budgets.map((budget) => ({
          value: budget.budget_id,
          label: hasBudgetName(budget)
            ? `${nameBudget(budget)} (${unnamedBudgetLabel(budget)})`
            : nameBudget(budget),
        })),
      ]}
      description="A budget that already exists caps this key’s own spend."
    />
  )
}
