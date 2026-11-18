import { useState } from "react"

import type { Workspace } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  DefaultBudgetPicker,
  NarrowedDefaults,
  NO_DEFAULT,
} from "@/features/workspaces/WorkspaceBudgetDefaults"
import { WorkspaceTraceContent } from "@/features/workspaces/WorkspaceTraceContent"
import { useBudgets } from "@/shared/api/budgets"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useDeploymentProviderNames } from "@/shared/api/providers"
import {
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspaceBudgetDefault,
  useUpdateWorkspace,
  useUpdateWorkspaceBudgetDefault,
  useWorkspaceBudgetDefaults,
} from "@/shared/api/workspaces"

/**
 * A workspace's own settings: its name and description, the budget its members
 * are given, and whether it keeps what its sessions were about.
 *
 * The fields wait for Save. The per-provider defaults and the trace switch write
 * as they are changed, so nothing they hold is unsaved work.
 */
export function WorkspaceGeneralForm({
  workspace,
  canManage,
}: {
  workspace: Workspace
  canManage: boolean
}) {
  const update = useUpdateWorkspace()
  // Budgets and providers are deployment-wide reads that answer 403 to an
  // organization admin who does not operate the deployment (#821), so neither is
  // asked for unless the caller may read it, and the default-budget controls
  // they feed are withheld with them (the OrganizationMembersPage pattern,
  // otari#838): rendered anyway, the picker showed the current default as a raw
  // UUID and the per-provider section had no provider to offer. The defaults
  // read itself is workspace-scoped and would answer, but this form only reads
  // it into those controls, so it is declined together with them.
  const { isOperator } = useDeploymentOperator()
  const budgets = useBudgets(isOperator)
  const defaults = useWorkspaceBudgetDefaults(isOperator ? workspace.id : null)
  const createDefault = useCreateWorkspaceBudgetDefault()
  const updateDefault = useUpdateWorkspaceBudgetDefault()
  const deleteDefault = useDeleteWorkspaceBudgetDefault()
  // The aggregate default: the one narrowed to no provider. A workspace has at
  // most one, enforced by a partial unique index.
  const aggregate = (defaults.data ?? []).find(
    (row) => row.provider_key_id === null,
  )
  const narrowed = (defaults.data ?? []).filter(
    (row) => row.provider_key_id !== null,
  )
  const providers = useDeploymentProviderNames(isOperator)
  const [name, setName] = useState(workspace.name)
  const [description, setDescription] = useState(workspace.description ?? "")
  const [budgetId, setBudgetId] = useState<string>()
  // Unset until the operator touches the picker, so a default that arrives
  // after the form mounted is still what the picker shows.
  const selectedBudget = budgetId ?? aggregate?.budget_id ?? NO_DEFAULT
  const savingDefault =
    createDefault.isPending ||
    updateDefault.isPending ||
    deleteDefault.isPending
  // `budgetId` rather than `selectedBudget`: unset is "the picker was never
  // touched", so a default that resolves after mount is part of the seed rather
  // than a change the guard should arm on.
  const { isDirty, reset } = useDirtySnapshot({ name, description, budgetId })

  // Three outcomes rather than one call: the default is its own row, so moving
  // between "none" and a budget is a create or a delete, not a field write.
  const saveDefault = async (): Promise<void> => {
    // With the picker withheld, an untouched `selectedBudget` over an unfetched
    // defaults list would read as "none" and delete nothing, but say so rather
    // than lean on that coincidence.
    if (!isOperator) return
    if (selectedBudget === NO_DEFAULT) {
      if (aggregate) {
        await deleteDefault.mutateAsync({
          workspaceId: workspace.id,
          defaultId: aggregate.id,
        })
      }
      return
    }
    if (!aggregate) {
      await createDefault.mutateAsync({
        workspaceId: workspace.id,
        body: { budget_id: selectedBudget },
      })
      return
    }
    if (aggregate.budget_id !== selectedBudget) {
      await updateDefault.mutateAsync({
        workspaceId: workspace.id,
        defaultId: aggregate.id,
        body: { budget_id: selectedBudget },
      })
    }
  }

  const trimmed = name.trim()
  const save = () => {
    if (update.isPending || savingDefault || trimmed === "") return
    update.mutate(
      {
        id: workspace.id,
        body: { name: trimmed, description: description.trim() || null },
      },
      {
        onSuccess: async () => {
          await saveDefault()
          // What was saved is the new seed, and the picker goes back to
          // "untouched" so it reads the saved default again.
          const saved = { name: trimmed, description: description.trim() }
          setName(saved.name)
          setDescription(saved.description)
          setBudgetId(undefined)
          reset(saved)
        },
      },
    )
  }

  // One column at the fields' own width, so every control on the tab shares
  // its left and right edges, and Save closes the fields it saves.
  return (
    <div className="flex max-w-md flex-col gap-8">
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          save()
        }}
      >
        <h2 className="text-title">Details</h2>
        <ErrorBanner
          error={
            update.error ??
            createDefault.error ??
            updateDefault.error ??
            deleteDefault.error
          }
        />
        <Field
          label="Name"
          value={name}
          onChange={setName}
          isRequired
          shouldReserveMessage={false}
        />
        <Field
          label="Description"
          value={description}
          onChange={setDescription}
          shouldReserveMessage={false}
        />
        {/* Withheld from a caller who does not operate the deployment: the
          picker's options come from the operator-gated `/budgets` read, so
          offering it would be offering a control whose list is empty and whose
          save cannot succeed. */}
        {isOperator ? (
          <>
            <DefaultBudgetPicker
              budgets={budgets.data ?? []}
              value={selectedBudget}
              onChange={setBudgetId}
            />
            <span className="text-caption">
              Each member gets their own allowance of this budget. A change
              applies to members who join afterwards.
            </span>
          </>
        ) : null}
        <div>
          <Button
            type="submit"
            variant="primary"
            isPending={update.isPending || savingDefault}
            // Held while the defaults load: until they answer, the aggregate
            // reads as absent and a save would create a second one.
            isDisabled={
              !canManage ||
              !isDirty ||
              trimmed === "" ||
              (isOperator && defaults.isPending)
            }
          >
            Save
          </Button>
        </div>
      </form>
      {/* Both save as they change, so they sit after the form rather than
          inside what Save commits. */}
      {isOperator ? (
        <section className="flex flex-col gap-3 border-t border-border pt-6">
          <NarrowedDefaults
            workspaceId={workspace.id}
            budgets={budgets.data ?? []}
            narrowed={narrowed}
            providers={providers}
            isReadOnly={!canManage}
          />
        </section>
      ) : null}
      <section className="flex flex-col gap-3 border-t border-border pt-6">
        <WorkspaceTraceContent workspace={workspace} />
      </section>
    </div>
  )
}
