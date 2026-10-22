import type { OrganizationBudget } from "@/client"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { useDeleteOrganizationBudget } from "@/shared/api/budgets"

// The one confirmation for deleting a budget, from the table and from the
// detail page. Deleting removes the ceilings applying it in the same write, so
// the question says what stops being capped rather than only what is deleted.
//
// The mutation lives here, under the caller's key, so a refusal (a workspace
// member default still naming the budget) does not greet the next open.

export function DeleteBudgetDialog({
  budget,
  budgetName,
  appliedTo,
  onOpenChange,
  onDeleted,
}: {
  /** The budget to delete; absent closes the dialog. */
  budget: OrganizationBudget | undefined
  budgetName: string
  /** What the budget applies to, as the "Applied to" cell reads it. */
  appliedTo: string
  onOpenChange: (open: boolean) => void
  onDeleted: () => void
}) {
  const remove = useDeleteOrganizationBudget()
  return (
    <ConfirmDialog
      isOpen={budget !== undefined}
      onOpenChange={onOpenChange}
      heading="Delete budget"
      body={
        budget
          ? budget.ceiling_count > budget.applied_to.length
            ? `${budgetName} also caps something outside this organization, which a deployment operator set, so this will be refused. Ask the operator to release it first.`
            : budget.applied_to.length > 0
              ? `${budgetName} is deleted, and ${appliedTo} ${budget.applied_to.length === 1 ? "stops" : "stop"} being capped by it from the next request. Spend already recorded stays in usage.`
              : `${budgetName} is deleted. It applies to nothing, so no cap changes.`
          : null
      }
      confirmLabel="Delete budget"
      isPending={remove.isPending}
      error={remove.error}
      onConfirm={() => {
        if (budget) remove.mutate(budget.budget_id, { onSuccess: onDeleted })
      }}
    />
  )
}
