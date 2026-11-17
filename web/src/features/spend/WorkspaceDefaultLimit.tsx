import { useState } from "react"
import { FiDollarSign, FiTrash2 } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { LimitDialog } from "@/features/spend/LimitDialog"
import { perWords } from "@/features/spend/limits"
import {
  useCreateOrganizationBudget,
  useDeleteOrganizationBudget,
  useRefreshSpendCeilings,
  useUpdateOrganizationBudget,
} from "@/shared/api/budgets"
import { ApiError } from "@/shared/api/client"
import {
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspaceBudgetDefault,
  useWorkspaceBudgetDefaults,
} from "@/shared/api/workspaces"
import { formatUsd } from "@/shared/helpers/format"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * The limit everyone in the selected workspace gets, newcomers included, so a
 * team is capped once rather than person by person. A person's own limit on
 * the row below binds as well; the tighter of the two is the one that bites.
 */
export function WorkspaceDefaultLimit() {
  const { selected } = useSelectedWorkspace()
  const workspaceId = selected?.workspace_id ?? null
  const defaults = useWorkspaceBudgetDefaults(workspaceId)
  const createBudget = useCreateOrganizationBudget()
  const updateBudget = useUpdateOrganizationBudget()
  const deleteBudget = useDeleteOrganizationBudget()
  const createDefault = useCreateWorkspaceBudgetDefault()
  const deleteDefault = useDeleteWorkspaceBudgetDefault()
  const refreshCeilings = useRefreshSpendCeilings()
  const [isEditing, setEditing] = useState(false)
  const [error, setError] = useState<unknown>()

  if (!workspaceId) return null
  // An unread list is not an empty one: setting a limit over it would make a
  // second default beside the one it could not see.
  if (defaults.data === undefined && defaults.isError) {
    return <ErrorBanner error={defaults.error} />
  }
  const current = (defaults.data ?? []).find(
    (one) => one.provider_key_id === null,
  )
  // $0 is a limit: nothing may be spent.
  const max = current?.max_budget ?? null
  const workspace = selected?.name ?? "this workspace"

  const save = async (amount: number) => {
    setError(undefined)
    try {
      if (current) {
        await updateBudget.mutateAsync({
          id: current.budget_id,
          body: { max_budget: amount },
        })
      } else {
        const budget = await createBudget.mutateAsync({
          name: `Each member of ${workspace}, monthly`,
          max_budget: amount,
          reset_alignment: "calendar_month",
        })
        try {
          await createDefault.mutateAsync({
            workspaceId,
            body: { budget_id: budget.budget_id },
          })
        } catch (caught) {
          // A budget nothing points at would be left behind on every retry.
          await deleteBudget
            .mutateAsync(budget.budget_id)
            .catch(() => undefined)
          throw caught
        }
        // The default caps everyone already here as it is made, after the
        // budget's own write refreshed the ceilings.
        refreshCeilings()
      }
      setEditing(false)
    } catch (caught) {
      setError(caught)
    }
  }

  const remove = async () => {
    if (!current) return
    setError(undefined)
    try {
      await deleteDefault.mutateAsync({ workspaceId, defaultId: current.id })
      // The people the default already reached keep their limits, which hold
      // the budget, and the server refuses to delete a budget in use. Only an
      // unused one goes.
      await deleteBudget
        .mutateAsync(current.budget_id)
        .catch((caught: unknown) => {
          if (!(caught instanceof ApiError && caught.status === 409))
            throw caught
        })
    } catch (caught) {
      setError(caught)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-3 border-y border-border py-3">
        <span className="text-sm">
          <span className="text-foreground">Everyone in {workspace}</span>{" "}
          <span className="text-muted">
            {defaults.data === undefined
              ? "…"
              : current && max !== null
                ? `may spend up to ${formatUsd(max)} ${perWords(current) || "in total"} each.`
                : "has no default limit yet."}
          </span>
        </span>
        {current ? (
          <RowActionRow>
            <RowAction
              icon={FiDollarSign}
              label="Change spend limit…"
              onPress={() => setEditing(true)}
            />
            <ConfirmRowAction
              icon={FiTrash2}
              label={`Remove the spend limit for everyone in ${workspace}`}
              confirmLabel="Remove"
              isDestructive
              isPending={deleteDefault.isPending || deleteBudget.isPending}
              onConfirm={() => void remove()}
            />
          </RowActionRow>
        ) : defaults.data !== undefined ? (
          <Button size="sm" onPress={() => setEditing(true)}>
            Set spend limit for everyone
          </Button>
        ) : null}
      </div>
      <ErrorBanner error={error} />
      {isEditing ? (
        <LimitDialog
          isOpen
          onOpenChange={setEditing}
          title={`Spend limit for everyone in ${workspace}`}
          description="Each member gets the full amount, people who join later included."
          initial={max !== null ? String(max) : ""}
          period={current}
          isPending={
            createBudget.isPending ||
            updateBudget.isPending ||
            createDefault.isPending
          }
          error={error}
          onSave={(amount) => void save(amount)}
        />
      ) : null}
    </div>
  )
}
