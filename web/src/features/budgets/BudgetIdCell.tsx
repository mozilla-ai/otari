import { CopyableValue } from "@/design-system/actions/CopyField"
import { shortBudgetId } from "./budgetLabel"

/**
 * A budget's id as a table cell: the head of the id with a button that copies
 * the whole one.
 *
 * Shared by both budgets tables because the end-user API takes the full id, so
 * each has to hand it over the same way.
 */
export function BudgetIdCell({ budgetId }: { budgetId: string }) {
  return (
    // Only a prefix is rendered; the copy hands over the whole id.
    <CopyableValue value={budgetId} label="budget id">
      <code className="text-mono-micro" title={budgetId}>
        {shortBudgetId(budgetId)}
      </code>
    </CopyableValue>
  )
}
