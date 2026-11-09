import type { ApiKey } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import {
  budgetLabeler,
  type LabelableBudget,
} from "@/features/budgets/budgetLabel"
import { useBudgets, useOrganizationBudgets } from "@/shared/api/budgets"

/**
 * Whose budgets a service key's end users go on: an operator's key uses the
 * deployment's, and an organization's own key that organization's.
 */
export type ServiceKeyScope = "deployment" | "organization"

/** Mirrors the gateway's `DEFAULT_ORGANIZATION_MAX_END_USERS`. */
const ORGANIZATION_MAX_END_USERS = "10000"

/**
 * What the form holds about a service key: whether the key is one, the budgets
 * its requests may start an end user on, which of those applies when a
 * request names none (`""` for none, so an end user then starts uncapped), and
 * the most end users it may have as typed (`""` for no cap, which only an
 * operator's key may have).
 */
export type ServiceKeyChoice = {
  isServiceKey: boolean
  budgetIds: string[]
  defaultBudgetId: string
  maxEndUsers: string
}

export const NO_SERVICE_KEY: ServiceKeyChoice = {
  isServiceKey: false,
  budgetIds: [],
  defaultBudgetId: "",
  maxEndUsers: "",
}

export const serviceKeyChoice = (apiKey: ApiKey): ServiceKeyChoice => ({
  isServiceKey: apiKey.is_service_key,
  budgetIds: apiKey.end_user_budget_ids,
  defaultBudgetId: apiKey.end_user_budget_id ?? "",
  maxEndUsers:
    // A gateway older than the field omits it, which reads as no cap.
    apiKey.max_end_users == null ? "" : String(apiKey.max_end_users),
})

/**
 * Why the typed cap cannot be saved, or undefined when it can. An
 * organization's service key always has a cap, so only an operator's may be
 * left empty.
 */
export function maxEndUsersError(
  choice: ServiceKeyChoice,
  scope: ServiceKeyScope,
): string | undefined {
  if (!choice.isServiceKey) return undefined
  const typed = choice.maxEndUsers.trim()
  if (typed === "") {
    return scope === "organization" ? "Required" : undefined
  }
  if (!/^\d+$/.test(typed)) return "A whole number, 0 or more"
  // Past the safe range a number rounds, or becomes Infinity, which JSON sends
  // as null and so would clear the cap.
  return Number.isSafeInteger(Number(typed))
    ? undefined
    : `At most ${Number.MAX_SAFE_INTEGER.toLocaleString("en-US")}`
}

/**
 * The key fields a choice writes. A key that is not a service key sends only the
 * flag, so turning it off keeps the budgets it had for the day it is turned back on.
 */
export const serviceKeyBody = (choice: ServiceKeyChoice) =>
  choice.isServiceKey
    ? {
        is_service_key: true,
        end_user_budget_ids: choice.budgetIds,
        end_user_budget_id: choice.defaultBudgetId || null,
        max_end_users:
          choice.maxEndUsers.trim() === ""
            ? null
            : Number(choice.maxEndUsers.trim()),
      }
    : { is_service_key: false }

/**
 * Whether a form holding this choice may be saved: a service key's budgets have
 * to have loaded, or a failed load would read as a deployment with none, and
 * its cap has to be one the scope accepts.
 */
export function useServiceKeyReady(
  choice: ServiceKeyChoice,
  scope: ServiceKeyScope,
): boolean {
  const budgets = useEndUserBudgets(scope, choice.isServiceKey)
  return (
    !choice.isServiceKey ||
    (budgets.data !== undefined && !maxEndUsersError(choice, scope))
  )
}

const NO_BUDGETS: readonly LabelableBudget[] = []

// The "no default" choice. `Select` shows "" as nothing chosen, so this label is
// also the placeholder that names it while it is the value.
const NO_DEFAULT_BUDGET_LABEL = "None: uncapped unless a request names one"

/**
 * The budgets a service key in this scope may start end users on. The
 * deployment's list also carries tenants' budgets, which an operator's key
 * may not use, so those are left out.
 */
export function useEndUserBudgets(scope: ServiceKeyScope, enabled: boolean) {
  const deployment = useBudgets(enabled && scope === "deployment")
  const organization = useOrganizationBudgets(
    enabled && scope === "organization",
  )
  const query = scope === "deployment" ? deployment : organization
  const offered: readonly LabelableBudget[] =
    scope === "deployment"
      ? (deployment.data ?? []).filter(
          (budget) => budget.organization_id === null,
        )
      : (organization.data ?? NO_BUDGETS)
  return {
    offered,
    data: query.data,
    error: query.error,
    isError: query.isError,
    isPending: query.isPending,
    refetch: query.refetch,
  }
}

export function ServiceKeyFields({
  value,
  onChange,
  scope,
  isLocked = false,
}: {
  value: ServiceKeyChoice
  onChange: (value: ServiceKeyChoice) => void
  scope: ServiceKeyScope
  /** Whether the key is one is fixed: an organization decides it when the key is created. */
  isLocked?: boolean
}) {
  const budgets = useEndUserBudgets(scope, value.isServiceKey)
  const { offered } = budgets
  const label = budgetLabeler(offered)
  const labelOf = (budgetId: string) => {
    const budget = offered.find((candidate) => candidate.budget_id === budgetId)
    return budget ? label(budget) : budgetId
  }

  const toggle = (budgetId: string, isSelected: boolean) => {
    const budgetIds = isSelected
      ? [...value.budgetIds, budgetId]
      : value.budgetIds.filter((listed) => listed !== budgetId)
    // The default has to be on the list, so taking it off clears it.
    const defaultBudgetId = budgetIds.includes(value.defaultBudgetId)
      ? value.defaultBudgetId
      : ""
    onChange({ ...value, budgetIds, defaultBudgetId })
  }

  return (
    <div className="flex flex-col gap-3 border border-control-border p-3">
      <div className="flex flex-col gap-0.5">
        <Checkbox
          isSelected={value.isServiceKey}
          onChange={(isServiceKey) =>
            onChange({
              ...value,
              isServiceKey,
              // An organization's key starts at the gateway's default cap.
              maxEndUsers:
                isServiceKey &&
                scope === "organization" &&
                value.maxEndUsers === ""
                  ? ORGANIZATION_MAX_END_USERS
                  : value.maxEndUsers,
            })
          }
          isDisabled={isLocked}
          hasTouchTarget
        >
          <span className="font-medium text-foreground">Service key</span>
        </Checkbox>
        <p className="text-caption">
          A request on this key may name an end user in its <code>user</code>{" "}
          field. Each end user is created on first use and billed to a budget of
          its own.
        </p>
        {scope === "organization" ? (
          <p className="text-caption">
            It belongs to the organization rather than to you: every owner and
            admin manages it, and it cannot be turned into a plain key or back
            later.
          </p>
        ) : null}
      </div>
      {value.isServiceKey ? (
        <>
          <fieldset className="flex flex-col gap-2">
            <legend className="text-body">End-user budgets</legend>
            <p className="text-caption">
              The budgets a request may start a new end user on, named in the{" "}
              <code>Otari-End-User-Budget</code> header.
            </p>
            {budgets.isError && !budgets.data ? (
              <div className="flex flex-col items-start gap-2">
                <ErrorBanner error={budgets.error} />
                <Button onPress={() => budgets.refetch()}>Retry</Button>
              </div>
            ) : budgets.isPending && !budgets.data ? (
              <p className="text-caption">Loading budgets…</p>
            ) : offered.length === 0 ? (
              <p className="text-caption">
                {scope === "deployment"
                  ? "No deployment budgets yet."
                  : "No organization budgets yet."}{" "}
                Create one under Spend &amp; budgets.
              </p>
            ) : (
              <div className="flex max-h-48 flex-col gap-1 overflow-y-auto">
                {offered.map((budget) => (
                  <Checkbox
                    key={budget.budget_id}
                    isSelected={value.budgetIds.includes(budget.budget_id)}
                    onChange={(isSelected) =>
                      toggle(budget.budget_id, isSelected)
                    }
                    hasTouchTarget
                  >
                    <span className="text-foreground">{label(budget)}</span>
                    {/* The id is what a request names, so it is shown unless the label already is it. */}
                    {label(budget) === budget.budget_id ? null : (
                      <>
                        {" "}
                        <code className="text-caption">{budget.budget_id}</code>
                      </>
                    )}
                  </Checkbox>
                ))}
              </div>
            )}
          </fieldset>
          <Select
            label="Default end-user budget"
            value={value.defaultBudgetId}
            onChange={(defaultBudgetId) =>
              onChange({ ...value, defaultBudgetId })
            }
            placeholder={NO_DEFAULT_BUDGET_LABEL}
            // Nothing to say beside it and nothing that can be invalid, so no
            // message line to hold open under it.
            shouldReserveMessage={false}
            options={[
              { value: "", label: NO_DEFAULT_BUDGET_LABEL },
              ...value.budgetIds.map((budgetId) => ({
                value: budgetId,
                label: labelOf(budgetId),
              })),
            ]}
          />
          <Field
            label="Max end users"
            value={value.maxEndUsers}
            onChange={(maxEndUsers) => onChange({ ...value, maxEndUsers })}
            inputMode="numeric"
            isRequired={scope === "organization"}
            placeholder={scope === "deployment" ? "No cap" : undefined}
            isInvalid={maxEndUsersError(value, scope) !== undefined}
            errorMessage={maxEndUsersError(value, scope)}
          />
        </>
      ) : null}
    </div>
  )
}
