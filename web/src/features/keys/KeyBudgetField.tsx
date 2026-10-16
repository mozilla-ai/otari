import { useMutation } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"

import type {
  Budget,
  OrganizationBudget,
  OrganizationSpendCeiling,
  ScopedBudget,
} from "@/client"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import {
  budgetLabeler,
  hasBudgetName,
  shortBudgetId,
  unnamedBudgetLabel,
} from "@/features/budgets/budgetLabel"
import {
  hasNoLimit,
  limitLabel,
  PERIOD_OPTIONS,
} from "@/features/budgets/organizationBudget"
import { canManage } from "@/features/organization/roles"
import {
  useApiKeyCeilings,
  useBudgets,
  useCreateBudget,
  useCreateOrganizationBudget,
  useCreateOrganizationSpendCeiling,
  useCreateScopedBudget,
  useDeleteBudget,
  useDeleteOrganizationBudget,
  useDeleteOrganizationSpendCeiling,
  useDeleteScopedBudget,
  useOrganizationBudgets,
  useUpdateOrganizationSpendCeiling,
  useUpdateScopedBudget,
} from "@/shared/api/budgets"
import { useOrganizationContext } from "@/shared/api/organizations"
import { formatUsd } from "@/shared/helpers/format"

// Neither can be a budget id, which starts with a letter or a digit.
const NO_BUDGET = ":none"
const CUSTOM_BUDGET = ":custom"

/** What a key form holds for its cap: none, an existing budget, or a limit typed in. */
export interface KeyBudgetDraft {
  choice: string
  amount: string
  period: string
}

export const EMPTY_KEY_BUDGET: KeyBudgetDraft = {
  choice: NO_BUDGET,
  amount: "",
  period: "calendar_month",
}

/** The custom amount as dollars, or undefined when it is not a positive number. */
function customAmount(draft: KeyBudgetDraft): number | undefined {
  const parsed = Number(draft.amount.trim())
  return draft.amount.trim() !== "" && Number.isFinite(parsed) && parsed > 0
    ? parsed
    : undefined
}

/** Whether the draft asks for a cap the form cannot send yet. */
export function keyBudgetIncomplete(draft: KeyBudgetDraft): boolean {
  return draft.choice === CUSTOM_BUDGET && customAmount(draft) === undefined
}

/** A key's own ceiling, as either budget surface reports it. */
export type KeyCeiling = ScopedBudget | OrganizationSpendCeiling

/** A budget a key can be put under, from either surface. */
export type CappingBudget = Budget | OrganizationBudget

// Stable empties, so a list that has not loaded does not read as a new one
// on every render and bust the table's per-row cache.
const NO_BUDGETS: readonly CappingBudget[] = []
const NO_CEILINGS: readonly KeyCeiling[] = []

/**
 * How a ceiling's budget reads. One the list does not hold (a deployment
 * budget on an organization's key) is described by what it caps instead.
 */
export function ceilingLabel(
  ceiling: KeyCeiling,
  budgets: readonly CappingBudget[],
): string {
  const budget = budgets.find((each) => each.budget_id === ceiling.budget_id)
  return budget ? budgetLabeler(budgets)(budget) : unnamedBudgetLabel(ceiling)
}

/** What the key has spent against its cap this period, as one line. */
export function ceilingUsage(ceiling: KeyCeiling): string {
  if (ceiling.max_budget === null) return limitLabel(ceiling)
  // Counters roll on the next request, so a period that has ended still
  // carries the last one's spend until traffic comes back.
  const ended =
    ceiling.period_end !== null && Date.parse(ceiling.period_end) <= Date.now()
  const spent = ended ? 0 : ceiling.current_spend
  return `${formatUsd(spent)} of ${formatUsd(ceiling.max_budget)}`
}

const CALENDAR_RESETS: Record<string, string> = {
  calendar_day: "resets daily",
  calendar_week: "resets weekly",
  calendar_month: "resets monthly",
}

/** How often a budget resets, in words: "/ 30 days" did not read as a reset. */
function resetPhrase(budget: CappingBudget): string {
  if (budget.reset_alignment) {
    return CALENDAR_RESETS[budget.reset_alignment] ?? "resets"
  }
  const seconds = budget.budget_duration_sec
  if (!seconds) return "never resets"
  const [count, unit] =
    seconds % 86_400 === 0
      ? [seconds / 86_400, "day"]
      : seconds % 3_600 === 0
        ? [seconds / 3_600, "hour"]
        : [seconds, "second"]
  return count === 1 ? `resets every ${unit}` : `resets every ${count} ${unit}s`
}

/** A budget as the picker offers it: its name, its limit, and its reset spelled out. */
function optionLabel(
  budget: CappingBudget,
  nameBudget: (budget: CappingBudget) => string,
): string {
  const limit = hasNoLimit(budget)
    ? limitLabel(budget)
    : `${limitLabel(budget)}, ${resetPhrase(budget)}`
  if (hasBudgetName(budget)) return `${nameBudget(budget)} (${limit})`
  // The labeler appends an id only where two unnamed budgets read alike.
  return nameBudget(budget) === unnamedBudgetLabel(budget)
    ? limit
    : `${limit} (${shortBudgetId(budget.budget_id)})`
}

/** The draft an edit form opens on: whatever the key is capped by now. */
export function draftFor(ceiling: KeyCeiling | undefined): KeyBudgetDraft {
  return ceiling
    ? { ...EMPTY_KEY_BUDGET, choice: ceiling.budget_id }
    : EMPTY_KEY_BUDGET
}

/**
 * The budgets a key may be capped by, what each key is capped by now, and the
 * writes that change it.
 *
 * A key's own cap is a spend ceiling on the `api_token` scope, so it is written
 * after the key exists. An operator works with the deployment's budgets and
 * `/scoped-budgets`; an organization owner or admin with the organization's own
 * budgets and spend ceilings. Anyone else is offered nothing, because every one
 * of those writes would refuse them. A custom amount becomes a budget of the
 * key's own first, named after it, so it is listed and editable like any other.
 */
export function useKeyBudgets(isDeploymentWide: boolean, enabled = true) {
  const context = useOrganizationContext()
  const isOrganizationManager = !isDeploymentWide && canManage(context.data)
  const canCap = isDeploymentWide || isOrganizationManager
  const deployment = useBudgets(enabled && isDeploymentWide)
  const organization = useOrganizationBudgets(enabled && isOrganizationManager)
  const ceilings = useApiKeyCeilings(
    isDeploymentWide ? "deployment" : "organization",
    enabled && canCap,
  )
  const createDeploymentBudget = useCreateBudget()
  const createOrganizationBudget = useCreateOrganizationBudget()
  const deleteDeploymentBudget = useDeleteBudget()
  const deleteOrganizationBudget = useDeleteOrganizationBudget()
  const createScoped = useCreateScopedBudget()
  const updateScoped = useUpdateScopedBudget()
  const deleteScoped = useDeleteScopedBudget()
  const createCeiling = useCreateOrganizationSpendCeiling()
  const updateCeiling = useUpdateOrganizationSpendCeiling()
  const deleteCeiling = useDeleteOrganizationSpendCeiling()
  const list = isDeploymentWide ? deployment : organization
  const budgets = list.data ?? NO_BUDGETS
  const keyCeilings = ceilings.data ?? NO_CEILINGS
  const nameBudget = budgetLabeler(budgets)

  const cap = useMutation({
    /** Resolves to how the cap now reads, or undefined once there is none. */
    mutationFn: async ({
      keyId,
      keyLabel,
      draft,
      current,
    }: {
      keyId: string
      keyLabel: string
      draft: KeyBudgetDraft
      current?: KeyCeiling
    }): Promise<string | undefined> => {
      if (draft.choice === NO_BUDGET) {
        if (current) {
          if (isDeploymentWide) await deleteScoped.mutateAsync(current.id)
          else await deleteCeiling.mutateAsync(current.id)
        }
        return undefined
      }
      if (current?.budget_id === draft.choice) {
        return ceilingLabel(current, budgets)
      }
      let budget = budgets.find((each) => each.budget_id === draft.choice)
      let created: CappingBudget | undefined
      if (draft.choice === CUSTOM_BUDGET) {
        const body = {
          name: `${keyLabel} (API key)`.slice(0, 200),
          max_budget: customAmount(draft) ?? null,
          reset_alignment:
            PERIOD_OPTIONS.find((option) => option.value === draft.period)
              ?.alignment ?? null,
        }
        created = isDeploymentWide
          ? await createDeploymentBudget.mutateAsync(body)
          : await createOrganizationBudget.mutateAsync(body)
        budget = created
      }
      if (!budget) throw new Error("That budget is no longer listed.")
      const budgetId = budget.budget_id
      try {
        if (current) {
          const change = { id: current.id, body: { budget_id: budgetId } }
          if (isDeploymentWide) await updateScoped.mutateAsync(change)
          else await updateCeiling.mutateAsync(change)
        } else {
          const ceiling = {
            scope_type: "api_token" as const,
            scope_id: keyId,
            budget_id: budgetId,
          }
          if (isDeploymentWide) await createScoped.mutateAsync(ceiling)
          else await createCeiling.mutateAsync(ceiling)
        }
      } catch (error) {
        // A budget made for this key alone and left capping nothing would sit
        // in Spend & budgets, and a retry would make a second one.
        if (created) {
          const deleteBudget = isDeploymentWide
            ? deleteDeploymentBudget
            : deleteOrganizationBudget
          await deleteBudget.mutateAsync(budgetId).catch(() => undefined)
        }
        throw error
      }
      return draft.choice === CUSTOM_BUDGET
        ? unnamedBudgetLabel(budget)
        : nameBudget(budget)
    },
  })

  return {
    canCap,
    budgets,
    isLoading: list.isPending && !list.data,
    /** Each key's own ceiling, once read; a key with none has no row. */
    ceilings: keyCeilings,
    ceilingsReady: !canCap || ceilings.isSuccess,
    /** Cap a new key, unless the draft asks for nothing. */
    apply: (keyId: string, keyLabel: string, draft: KeyBudgetDraft) => {
      if (draft.choice !== NO_BUDGET) cap.mutate({ keyId, keyLabel, draft })
    },
    /** Move an existing key's cap to what the draft says, which may be none. */
    change: cap.mutateAsync,
    isApplying: cap.isPending,
    applied: cap.data,
    applyError: cap.error,
    reset: cap.reset,
  }
}

export function KeyBudgetField({
  value,
  onChange,
  budgets,
  isLoading,
  isExempt,
  current,
}: {
  value: KeyBudgetDraft
  onChange: (draft: KeyBudgetDraft) => void
  budgets: readonly CappingBudget[]
  isLoading: boolean
  /** A budget-exempt key is never gated, so a cap on it would never bind. */
  isExempt: boolean
  /**
   * What the key is capped by now, with how it reads. Carried as its own option
   * when the list does not hold its budget, so the field names it rather than
   * showing its id.
   */
  current?: { budgetId: string; label: string }
}) {
  const nameBudget = budgetLabeler(budgets)
  const unlisted =
    current && !budgets.some((budget) => budget.budget_id === current.budgetId)
      ? [{ value: current.budgetId, label: current.label }]
      : []
  const options = [
    { value: NO_BUDGET, label: "No budget" },
    ...unlisted,
    ...budgets.map((budget) => ({
      value: budget.budget_id,
      label: optionLabel(budget, nameBudget),
    })),
    { value: CUSTOM_BUDGET, label: "Custom amount…" },
  ]
  const isCustom = !isExempt && value.choice === CUSTOM_BUDGET
  const amountInvalid =
    value.amount.trim() !== "" && customAmount(value) === undefined
  return (
    <div className="flex flex-col gap-4">
      <Select
        label="Budget (optional)"
        value={value.choice}
        onChange={(choice) =>
          onChange({ ...value, choice: choice === "" ? NO_BUDGET : choice })
        }
        options={options}
        isDisabled={isExempt || isLoading}
        description={
          isExempt ? (
            "A budget-exempt key is never capped."
          ) : (
            <>
              Caps this key's own spend. Pick a budget from{" "}
              <Link
                to="/budgets"
                className="font-medium text-link hover:text-link-hover"
              >
                Spend &amp; budgets
              </Link>
              , or set an amount for this key alone.
            </>
          )
        }
      />
      {isCustom ? (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Limit (USD)"
            value={value.amount}
            onChange={(amount) => onChange({ ...value, amount })}
            placeholder="50"
            isRequired
            isInvalid={amountInvalid}
            errorMessage="Enter an amount above zero."
          />
          <Select
            label="Resets"
            value={value.period}
            onChange={(period) => onChange({ ...value, period })}
            options={PERIOD_OPTIONS.map((option) => ({
              value: option.value,
              label: option.label,
            }))}
          />
        </div>
      ) : null}
    </div>
  )
}
