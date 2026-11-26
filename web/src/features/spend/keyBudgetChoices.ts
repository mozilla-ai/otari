import type { CappingBudget } from "@/features/keys/KeyBudgetField"
import {
  useAllWorkspaceBudgetDefaults,
  useWorkspaces,
} from "@/shared/api/workspaces"

const NONE: { budgets: readonly CappingBudget[]; isLoading: boolean } = {
  budgets: [],
  isLoading: false,
}

/**
 * The existing budgets a key may be capped by: those `useKeyBudgets` lists,
 * less any that backs a workspace's per-person default. A default is changed
 * by editing its budget in place, so a key capped by one would move with it.
 *
 * Loading until every workspace's defaults are read. If they cannot be read,
 * no existing budget is offered rather than one that may be a default.
 */
export function useKeyBudgetChoices(
  keyBudgets: { budgets: readonly CappingBudget[]; isLoading: boolean },
  isEnabled: boolean,
): { budgets: readonly CappingBudget[]; isLoading: boolean } {
  const workspaces = useWorkspaces(isEnabled)
  const defaults = useAllWorkspaceBudgetDefaults(
    isEnabled ? (workspaces.data ?? []).map((one) => one.id) : [],
  )
  if (!isEnabled || workspaces.isError || defaults.error) return NONE
  const backsDefault = new Set(
    defaults.data.map((row) => row.default.budget_id),
  )
  return {
    budgets: keyBudgets.budgets.filter(
      (one) => !backsDefault.has(one.budget_id),
    ),
    isLoading:
      keyBudgets.isLoading || workspaces.isPending || !defaults.isSuccess,
  }
}
