import { Button } from "@heroui/react"
import { useState } from "react"

import type { Budget, WorkspaceBudgetDefault } from "@/client"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Select } from "@/design-system/forms/Select"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { budgetLabeler } from "@/features/budgets/budgetLabel"
import {
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspaceBudgetDefault,
  useUpdateWorkspaceBudgetDefault,
} from "@/shared/api/workspaces"

// The budget every member of a workspace is given. It is not a field on the
// workspace: it is a default row that materializes a real per-member ceiling
// when someone joins, so a person in two workspaces holds two ceilings, one from
// each.
//
// A workspace may also narrow a default to one provider, which the field below
// does not cover. Those are managed in the section under it rather than left
// unreachable: an upgraded deployment can already have them, they show on the
// budgets list under "Default for", and deleting the budget one names is refused
// with a message telling the operator to come and change it.
export const NO_DEFAULT = ""

function budgetChoices(budgets: Budget[]): { value: string; label: string }[] {
  const nameBudget = budgetLabeler(budgets)
  return budgets.map((budget) => ({
    value: budget.budget_id,
    label: nameBudget(budget),
  }))
}

export function DefaultBudgetPicker({
  budgets,
  value,
  onChange,
}: {
  budgets: Budget[]
  value: string
  onChange: (budgetId: string) => void
}) {
  return (
    <Select
      label="Default member budget"
      value={value}
      onChange={onChange}
      // No description and nothing that can be invalid, so there is no message
      // this could ever hold. `FieldMessages` reserves the line by default, and
      // `FilterSelect` reserved nothing, so the swap to `Select` brought ~23px
      // of empty back with it.
      shouldReserveMessage={false}
      options={[
        { value: NO_DEFAULT, label: "No default" },
        ...budgetChoices(budgets),
      ]}
    />
  )
}

/**
 * The defaults a workspace narrows to one provider.
 *
 * Separate from the field above because they are a different question: that one
 * is "what does everyone here get", these are "and what do they get on this
 * provider specifically". Writes go straight through rather than waiting for the
 * form's Save, since each is its own row and batching them would mean holding a
 * pending create, a pending retarget and a pending delete for an arbitrary number
 * of providers to no benefit.
 */
export function NarrowedDefaults({
  workspaceId,
  budgets,
  narrowed,
  providers,
  isReadOnly = false,
}: {
  workspaceId: string
  budgets: Budget[]
  narrowed: WorkspaceBudgetDefault[]
  providers: string[]
  /** Shown to a caller who may not change the workspace, every control held. */
  isReadOnly?: boolean
}) {
  const createDefault = useCreateWorkspaceBudgetDefault()
  const updateDefault = useUpdateWorkspaceBudgetDefault()
  const deleteDefault = useDeleteWorkspaceBudgetDefault()
  const [provider, setProvider] = useState("")
  const [budgetId, setBudgetId] = useState("")
  const [pendingDelete, setPendingDelete] = useState<WorkspaceBudgetDefault>()

  const taken = new Set(narrowed.map((row) => row.provider_key_id))
  const available = providers.filter((instance) => !taken.has(instance))
  const isPending =
    createDefault.isPending ||
    updateDefault.isPending ||
    deleteDefault.isPending
  const isHeld = isPending || isReadOnly

  return (
    <div className="flex flex-col gap-2">
      {/* The section head role, as the provider-keys group below it uses. */}
      <span className="text-title">Per-provider defaults</span>
      <ErrorBanner error={createDefault.error ?? updateDefault.error} />
      {narrowed.length === 0 ? (
        <span className="text-caption">
          None. The budget above applies on every provider.
        </span>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {narrowed.map((row) => (
            <li key={row.id} className="flex items-center gap-2">
              <span className="text-mono-caption text-foreground">
                {row.provider_key_id}
              </span>
              <FilterSelect
                ariaLabel={`Budget for ${row.provider_key_id}`}
                value={row.budget_id}
                onChange={(next) =>
                  updateDefault.mutate({
                    workspaceId,
                    defaultId: row.id,
                    body: { budget_id: next },
                  })
                }
                options={budgetChoices(budgets)}
                disabled={isHeld}
              />
              <Button
                size="sm"
                variant="ghost"
                // Named per row, as the picker beside it is: this is a list of
                // providers, not a table with a row header to lean on.
                aria-label={`Remove default for ${row.provider_key_id}`}
                isDisabled={isHeld}
                onPress={() => setPendingDelete(row)}
              >
                Remove
              </Button>
            </li>
          ))}
        </ul>
      )}
      {available.length > 0 && budgets.length > 0 ? (
        <div className="flex items-end gap-2">
          <FilterSelect
            label="Provider"
            value={provider}
            onChange={setProvider}
            options={[
              { value: "", label: "Select a provider…" },
              ...available.map((instance) => ({
                value: instance,
                label: instance,
              })),
            ]}
          />
          <FilterSelect
            label="Budget"
            value={budgetId}
            onChange={setBudgetId}
            options={[
              { value: "", label: "Select a budget…" },
              ...budgetChoices(budgets),
            ]}
          />
          <Button
            size="sm"
            variant="ghost"
            isDisabled={isHeld || provider === "" || budgetId === ""}
            onPress={() =>
              createDefault.mutate(
                {
                  workspaceId,
                  body: { budget_id: budgetId, provider_key_id: provider },
                },
                {
                  onSuccess: () => {
                    setProvider("")
                    setBudgetId("")
                  },
                },
              )
            }
          >
            Add
          </Button>
        </div>
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deleteDefault.reset()
        }}
        heading="Remove per-provider default"
        body={
          pendingDelete
            ? `Requests to ${pendingDelete.provider_key_id} fall back to the workspace default above. Nothing caps them separately on that provider any more.`
            : null
        }
        confirmLabel="Remove default"
        isPending={deleteDefault.isPending}
        error={deleteDefault.error}
        onConfirm={() => {
          if (!pendingDelete) return
          deleteDefault.mutate(
            { workspaceId, defaultId: pendingDelete.id },
            { onSuccess: () => setPendingDelete(undefined) },
          )
        }}
      />
    </div>
  )
}
