import { useState } from "react"
import type { TeamDefault } from "@/features/rules/rules"
import { LimitDialog } from "@/features/spend/LimitDialog"
import { useUpdateOrganizationBudget } from "@/shared/api/budgets"

/**
 * A workspace's per-person default, by its amount: the budget it points at,
 * as `WorkspaceDefaultLimit` changes it. Owners and admins only.
 */
export function ChangeDefaultDialog({
  teamDefault,
  onClose,
}: {
  teamDefault: TeamDefault
  onClose: () => void
}) {
  const updateBudget = useUpdateOrganizationBudget()
  const [error, setError] = useState<unknown>()
  return (
    <LimitDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={`Spend limit for everyone in ${teamDefault.team}`}
      description="Each member gets the full amount, people who join later included."
      initial={teamDefault.max !== null ? String(teamDefault.max) : ""}
      period={teamDefault.period}
      isPending={updateBudget.isPending}
      error={error}
      onSave={(amount) => {
        setError(undefined)
        updateBudget
          .mutateAsync({
            id: teamDefault.budgetId,
            body: { max_budget: amount },
          })
          .then(onClose, setError)
      }}
    />
  )
}
