import { Button, Card, Chip } from "@heroui/react"
import { useMemo, useState } from "react"

import type { Budget, Workspace } from "@/client"
import { canManage } from "@/features/organization/roles"
import {
  useAllWorkspaceBudgetDefaults,
  useBudgets,
  useCreateWorkspace,
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspace,
  useDeleteWorkspaceBudgetDefault,
  useOrganizationContext,
  useUpdateWorkspace,
  useUpdateWorkspaceBudgetDefault,
  useWorkspaceBudgetDefaults,
  useWorkspaces,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  EmptyState,
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { formatDate } from "@/shared/helpers/format"

// Workspaces are the unit inside an organization that work is scoped to. This
// page lists the ones the caller can see, and opening a row reveals its roster:
// a workspace's members are only ever a subset of the organization's, which is
// why adding one is a picker over the roster rather than an invitation.

// The workspace vocabulary is the organization one: four fixed roles, the same
// spellings, published on both requests. `asMembershipRole` narrows a picker's
// string back to it.
const getWorkspaceRowKey = (workspace: Workspace): string => workspace.id

const LAST_WORKSPACE_REASON =
  "An organization keeps at least one workspace; create another first"

// The budget every member of a workspace is given. It is not a field on the
// workspace: it is a default row that materializes a real per-member ceiling
// when someone joins, so a person in two workspaces holds two ceilings, one from
// each. Only the aggregate default (the one narrowed to no provider) is offered
// here, which is the one the design draws; a provider-narrowed default still
// shows on the budgets list, under "Default for".
const NO_DEFAULT = ""

function DefaultBudgetPicker({
  budgets,
  value,
  onChange,
}: {
  budgets: Budget[]
  value: string
  onChange: (budgetId: string) => void
}) {
  return (
    <FilterSelect
      label="Default member budget"
      value={value}
      onChange={onChange}
      options={[
        { value: NO_DEFAULT, label: "No default" },
        ...budgets.map((budget) => ({
          value: budget.budget_id,
          label: budget.name ?? budget.budget_id.split("-")[0],
        })),
      ]}
    />
  )
}

/**
 * The one form that creates a workspace, wherever it is offered from.
 *
 * Exported because the scope switcher offers the same action (the navigation
 * design puts "Create workspace" at the foot of that menu), and two forms over
 * one endpoint drift: one of them grows the description field, or the ownership
 * note, and the other does not.
 */
export function CreateWorkspaceForm({ onClose }: { onClose: () => void }) {
  const create = useCreateWorkspace()
  const createDefault = useCreateWorkspaceBudgetDefault()
  const budgets = useBudgets()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [budgetId, setBudgetId] = useState(NO_DEFAULT)
  const trimmed = name.trim()
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Create workspace
        </div>
        <ErrorBanner error={create.error ?? createDefault.error} />
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Production"
          isRequired
          autoFocus
          description="Unique within this organization. You become its owner."
        />
        <Field
          label="Description (optional)"
          value={description}
          onChange={setDescription}
        />
        <DefaultBudgetPicker
          budgets={budgets.data ?? []}
          value={budgetId}
          onChange={setBudgetId}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={create.isPending || createDefault.isPending}
            onPress={() =>
              create.mutate(
                { name: trimmed, description: description.trim() || null },
                {
                  // The default is a second call: the workspace has to exist
                  // before anything can be defaulted onto its members. A failure
                  // here leaves the workspace created and undefaulted, which the
                  // banner reports and the edit form can finish.
                  onSuccess: (workspace) => {
                    if (budgetId === NO_DEFAULT) {
                      onClose()
                      return
                    }
                    createDefault.mutate(
                      {
                        workspaceId: workspace.id,
                        body: { budget_id: budgetId },
                      },
                      { onSuccess: onClose },
                    )
                  },
                },
              )
            }
          >
            Create workspace
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function EditWorkspaceForm({
  workspace,
  onClose,
}: {
  workspace: Workspace
  onClose: () => void
}) {
  const update = useUpdateWorkspace()
  const budgets = useBudgets()
  const defaults = useWorkspaceBudgetDefaults(workspace.id)
  const createDefault = useCreateWorkspaceBudgetDefault()
  const updateDefault = useUpdateWorkspaceBudgetDefault()
  const deleteDefault = useDeleteWorkspaceBudgetDefault()
  // The aggregate default: the one narrowed to no provider. A workspace has at
  // most one, enforced by a partial unique index.
  const aggregate = (defaults.data ?? []).find(
    (row) => row.provider_key_id === null,
  )
  const [name, setName] = useState(workspace.name)
  const [description, setDescription] = useState(workspace.description ?? "")
  const [budgetId, setBudgetId] = useState<string | null>(null)
  // Null until the operator touches the picker, so a default that arrives after
  // the form mounted is still what the picker shows.
  const selectedBudget = budgetId ?? aggregate?.budget_id ?? NO_DEFAULT
  const savingDefault =
    createDefault.isPending ||
    updateDefault.isPending ||
    deleteDefault.isPending

  // Three outcomes rather than one call: the default is its own row, so moving
  // between "none" and a budget is a create or a delete, not a field write.
  const saveDefault = async (): Promise<void> => {
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
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Edit <code>{workspace.name}</code>
        </div>
        <ErrorBanner
          error={
            update.error ??
            createDefault.error ??
            updateDefault.error ??
            deleteDefault.error
          }
        />
        <Field label="Name" value={name} onChange={setName} isRequired />
        <Field
          label="Description"
          value={description}
          onChange={setDescription}
        />
        <DefaultBudgetPicker
          budgets={budgets.data ?? []}
          value={selectedBudget}
          onChange={setBudgetId}
        />
        <span className="max-w-md text-xs text-muted">
          Every member of this workspace is held to this budget, each with their
          own allowance. Changing it applies to members who join afterwards;
          members already here keep what they were given.
        </span>
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={update.isPending || savingDefault}
            onPress={() =>
              update.mutate(
                {
                  id: workspace.id,
                  body: {
                    name: trimmed,
                    description: description.trim() || null,
                  },
                },
                {
                  onSuccess: async () => {
                    await saveDefault()
                    onClose()
                  },
                },
              )
            }
          >
            Save changes
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

export function WorkspacesPage() {
  const context = useOrganizationContext()
  const workspaces = useWorkspaces()
  const budgets = useBudgets()
  const remove = useDeleteWorkspace()

  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<Workspace | null>(null)

  const rows = workspaces.data ?? []
  const workspaceIds = useMemo(() => rows.map((row) => row.id), [rows])
  const workspaceDefaults = useAllWorkspaceBudgetDefaults(workspaceIds)
  // The budget each workspace hands to its members, by workspace. Only the
  // aggregate default (no provider narrowing) is named: that is the one the
  // edit form sets, and a narrowed one is the budget's business, where it shows
  // under "Default for".
  const defaultBudgetName = useMemo(() => {
    const names = new Map(
      (budgets.data ?? []).map((budget) => [
        budget.budget_id,
        budget.name ?? budget.budget_id.split("-")[0],
      ]),
    )
    const byWorkspace = new Map<string, string>()
    for (const { workspaceId, default: row } of workspaceDefaults.data) {
      if (row.provider_key_id === null) {
        byWorkspace.set(
          workspaceId,
          names.get(row.budget_id) ?? row.budget_id.split("-")[0],
        )
      }
    }
    return byWorkspace
  }, [budgets.data, workspaceDefaults.data])
  // Only once the list has actually answered: an empty list while loading is
  // not one workspace, and disabling on it would flicker.
  const isOnlyWorkspace = workspaces.isSuccess && rows.length === 1
  const manages = canManage(context.data)
  const editingWorkspace = rows.find((row) => row.id === editing) ?? null
  const showOnboarding = !workspaces.isLoading && rows.length === 0 && !creating

  const columns = useMemo<DataTableColumn<Workspace>[]>(
    () => [
      {
        id: "name",
        header: "Workspace",
        isRowHeader: true,
        cell: (workspace) => (
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-medium text-foreground">
              {workspace.name}
            </span>
            {workspace.description ? (
              <span className="text-xs text-muted">
                {workspace.description}
              </span>
            ) : null}
          </div>
        ),
      },
      {
        id: "created",
        header: "Created",
        cell: (workspace) => (
          <span className="text-muted">{formatDate(workspace.created_at)}</span>
        ),
      },
      {
        id: "default-budget",
        header: "Default member budget",
        cell: (workspace) => {
          const name = defaultBudgetName.get(workspace.id)
          return name ? (
            <Chip size="sm">{name}</Chip>
          ) : (
            <span className="text-xs text-muted">None</span>
          )
        },
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (workspace) => (
          <div className="flex items-center justify-end gap-1.5">
            <Button
              size="sm"
              variant="ghost"
              isDisabled={!manages}
              onPress={() => {
                setCreating(false)
                setEditing(workspace.id)
              }}
            >
              Edit
            </Button>
            {/* The server keeps every organization on at least one workspace
                (`LastWorkspaceError`), and first boot is the one-workspace
                state, so the ordinary case would be a button that always
                refuses. Say why instead of offering the refusal. The reason
                goes in the name, following the membership controls: a disabled
                control takes no focus, so a tooltip would reach a pointer and
                nothing else. */}
            <span title={isOnlyWorkspace ? LAST_WORKSPACE_REASON : undefined}>
              <Button
                size="sm"
                variant="danger-soft"
                aria-label={
                  isOnlyWorkspace
                    ? `Delete ${workspace.name} (${LAST_WORKSPACE_REASON})`
                    : undefined
                }
                isDisabled={!manages || isOnlyWorkspace}
                onPress={() => setDeleting(workspace)}
              >
                Delete
              </Button>
            </span>
          </div>
        ),
      },
    ],
    [manages, isOnlyWorkspace, defaultBudgetName],
  )

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Workspaces"
        description="The scopes inside this organization that work is grouped into. Each one carries its own members and roles."
        action={
          creating || !manages ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null)
                setCreating(true)
              }}
            >
              Create workspace
            </Button>
          )
        }
      />

      {/* `remove.error` is deliberately absent: the confirm dialog renders that
          mutation's error itself, and listing it here too paints the same
          message twice, once behind the open dialog. */}
      <ErrorBanner error={context.error ?? workspaces.error} />

      {/* Withheld until the context answers, so an owner is not told for one
          paint that they may not manage their own workspaces. */}
      {context.isLoading || manages ? null : (
        <InfoBanner>
          Only organization owners and admins can create, edit, or delete
          workspaces.
        </InfoBanner>
      )}

      {creating ? (
        <CreateWorkspaceForm onClose={() => setCreating(false)} />
      ) : null}

      {/* Keyed on the workspace so switching which one is edited remounts the
          form: its fields seed from `workspace` on mount only. */}
      {editingWorkspace ? (
        <EditWorkspaceForm
          key={editingWorkspace.id}
          workspace={editingWorkspace}
          onClose={() => setEditing(null)}
        />
      ) : null}

      {showOnboarding ? (
        <EmptyState
          title="No workspaces yet"
          description="A workspace groups the work inside this organization and carries its own members and roles. Every organization is created with one, so an empty list usually means the default was deleted."
          actionLabel={manages ? "Create a workspace" : undefined}
          onAction={manages ? () => setCreating(true) : undefined}
        />
      ) : (
        <DataTable
          ariaLabel="Workspaces"
          columns={columns}
          rows={rows}
          getRowKey={getWorkspaceRowKey}
          isLoading={workspaces.isLoading}
          emptyContent="No workspaces yet."
        />
      )}

      <ConfirmDialog
        isOpen={deleting !== null}
        onOpenChange={(open) => {
          if (!open) setDeleting(null)
        }}
        heading="Delete workspace"
        body={
          <>
            Delete <strong>{deleting?.name}</strong> and its memberships? Usage
            the gateway already recorded is not deleted with it.
          </>
        }
        confirmLabel="Delete workspace"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (deleting) {
            remove.mutate(deleting.id, {
              onSuccess: () => {
                if (editing === deleting.id) setEditing(null)
                setDeleting(null)
              },
            })
          }
        }}
      />
    </div>
  )
}
