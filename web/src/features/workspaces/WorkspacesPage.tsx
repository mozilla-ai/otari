import { Button, Card, Chip, Spinner } from "@heroui/react"
import { useEffect, useMemo, useRef, useState } from "react"

import type { Budget, Workspace, WorkspaceBudgetDefault } from "@/client"
import { canManage } from "@/features/organization/roles"
import { ApiError } from "@/shared/api/client"
import {
  useAllWorkspaceBudgetDefaults,
  useBudgets,
  useCreateWorkspace,
  useCreateWorkspaceBudgetDefault,
  useDeleteWorkspace,
  useDeleteWorkspaceBudgetDefault,
  useOrganizationContext,
  useProviders,
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
  errorMessage,
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
// each.
//
// A workspace may also narrow a default to one provider, which the field below
// does not cover. Those are managed in the section under it rather than left
// unreachable: an upgraded deployment can already have them, they show on the
// budgets list under "Default for", and deleting the budget one names is refused
// with a message telling the operator to come and change it.
const NO_DEFAULT = ""

function budgetLabel(budget: Budget): string {
  return budget.name ?? budget.budget_id.split("-")[0]
}

function budgetChoices(budgets: Budget[]): { value: string; label: string }[] {
  return budgets.map((budget) => ({
    value: budget.budget_id,
    label: budgetLabel(budget),
  }))
}

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
function NarrowedDefaults({
  workspaceId,
  budgets,
  narrowed,
  providers,
}: {
  workspaceId: string
  budgets: Budget[]
  narrowed: WorkspaceBudgetDefault[]
  providers: string[]
}) {
  const createDefault = useCreateWorkspaceBudgetDefault()
  const updateDefault = useUpdateWorkspaceBudgetDefault()
  const deleteDefault = useDeleteWorkspaceBudgetDefault()
  const [provider, setProvider] = useState("")
  const [budgetId, setBudgetId] = useState("")

  const taken = new Set(narrowed.map((row) => row.provider_key_id))
  const available = providers.filter((instance) => !taken.has(instance))
  const pending =
    createDefault.isPending ||
    updateDefault.isPending ||
    deleteDefault.isPending

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-foreground">
        Per-provider defaults
      </span>
      <ErrorBanner
        error={
          createDefault.error ?? updateDefault.error ?? deleteDefault.error
        }
      />
      {narrowed.length === 0 ? (
        <span className="text-xs text-muted">
          None. The budget above applies on every provider.
        </span>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {narrowed.map((row) => (
            <li key={row.id} className="flex items-center gap-2">
              <Chip size="sm">{row.provider_key_id}</Chip>
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
                disabled={pending}
              />
              <Button
                size="sm"
                variant="danger-soft"
                isDisabled={pending}
                onPress={() =>
                  deleteDefault.mutate({ workspaceId, defaultId: row.id })
                }
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
            isDisabled={pending || provider === "" || budgetId === ""}
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
    </div>
  )
}

// How long the submit holds before a create that navigates hands the page over.
// Long enough for the spinner to register as this press having done something,
// short enough not to be a wait: the operator should read it as the button
// acknowledging them, not as the gateway being slow.
const ENTER_HOLD_MS = 800

/**
 * The one form that creates a workspace, wherever it is offered from.
 *
 * Exported because the scope switcher offers the same action (the navigation
 * design puts "Create workspace" at the foot of that menu), and two forms over
 * one endpoint drift: one of them grows the description field, or the ownership
 * note, and the other does not.
 */
export function CreateWorkspaceForm({
  onClose,
  onCreated,
}: {
  onClose: () => void
  // Fired once the workspace exists (and its default budget, when one was
  // picked), so the caller that opened the form can follow the new workspace.
  // The list page does not need it: it is already looking at the row that
  // appeared. The scope switcher does, because creating from there is a request
  // to work in the new workspace, not just to add it.
  onCreated?: (workspace: Workspace) => void
}) {
  const create = useCreateWorkspace()
  const createDefault = useCreateWorkspaceBudgetDefault()
  const budgets = useBudgets()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [budgetId, setBudgetId] = useState(NO_DEFAULT)
  // The hold is running, on a form whose submit navigates. Kept separately from
  // the mutations' own pending flags, because it outlives them: the request can
  // answer in 90ms and the button still has a beat left to serve.
  const [holding, setHolding] = useState(false)
  // Whether this form is still the one on screen. The hold outlives the render
  // that started it, so a dismissal during it would otherwise still hand the
  // page over: the operator cancels and is taken somewhere anyway. Disabling
  // Cancel is not enough on its own, because Escape and a click outside dismiss
  // the modal too, and neither goes through a button.
  const active = useRef(true)
  useEffect(
    () => () => {
      active.current = false
    },
    [],
  )
  const trimmed = name.trim()
  // The submit promises the navigation only where it performs one, so the label
  // and the hold below are read off the same prop that does it. A form whose
  // button said "and open" while nothing opened would be the worse bug of the
  // two this fixes.
  const entersWorkspace = onCreated !== undefined
  const pending = create.isPending || createDefault.isPending || holding
  // Only a refusal *about the name* belongs on the name. These three are the
  // ones this endpoint answers with when the input is the problem: taken (409),
  // malformed (400), or rejected by the schema (422). A 403, a 500 or a dropped
  // connection is not something the operator can fix by retyping, and reddening
  // the field for it would tell them, and assistive tech, that it was.
  const nameRefusal =
    create.error instanceof ApiError &&
    [400, 409, 422].includes(create.error.status)
      ? create.error
      : null
  // Everything else the form can fail on, in the one place that is not a field:
  // a create that failed for another reason, and the default-budget call, which
  // fails after the workspace already exists.
  const bannerError = createDefault.error ?? (nameRefusal ? null : create.error)
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-foreground">
          Create workspace
        </div>
        {/* A refusal about the name is carried by the name, not by a block above
            the form that resizes whatever frames it. What is left here is what
            no field state can honestly say: a failure the operator cannot
            retype their way out of, and the default-budget call, which is a
            separate call about a different control and fails after the
            workspace already exists. */}
        <ErrorBanner error={bannerError} />
        <Field
          label="Name"
          value={name}
          onChange={(next) => {
            setName(next)
            // The refusal was about the name that produced it, so editing the
            // name retires it. Without this the field stays red while the
            // operator types the correction, and the button they press next
            // looks like it is retrying a rejected name. Reset on any create
            // error, not just a name one: the banner's copy is about the
            // attempt, and the attempt is what editing the name replaces.
            if (create.error) create.reset()
          }}
          placeholder="Production"
          isRequired
          autoFocus
          isInvalid={nameRefusal !== null}
          errorMessage={nameRefusal ? errorMessage(nameRefusal) : undefined}
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
            isPending={pending}
            onPress={() => {
              // Started before the request, not after it answers, so the two run
              // together: the operator waits a beat, not a beat plus a round
              // trip. A create slower than the hold keeps the spinner until it
              // answers, which is the honest reading of the same indicator.
              const held = entersWorkspace
                ? new Promise<void>((resolve) => {
                    setTimeout(resolve, ENTER_HOLD_MS)
                  })
                : Promise.resolve()
              setHolding(entersWorkspace)
              create.mutate(
                { name: trimmed, description: description.trim() || null },
                {
                  // The default is a second call: the workspace has to exist
                  // before anything can be defaulted onto its members. A failure
                  // here leaves the workspace created and undefaulted, which the
                  // banner reports and the edit form can finish.
                  onSuccess: (workspace) => {
                    const finish = async () => {
                      await held
                      // Dismissed while it was held: the workspace exists, and
                      // the list and the switcher will both show it, but the
                      // operator said not to go there.
                      if (!active.current) return
                      // No setHolding here: the form unmounts on close, and the
                      // failure paths below are what release the button.
                      onClose()
                      onCreated?.(workspace)
                    }
                    if (budgetId === NO_DEFAULT) {
                      void finish()
                      return
                    }
                    createDefault.mutate(
                      {
                        workspaceId: workspace.id,
                        body: { budget_id: budgetId },
                      },
                      {
                        onSuccess: () => {
                          void finish()
                        },
                        onError: () => setHolding(false),
                      },
                    )
                  },
                  onError: () => setHolding(false),
                },
              )
            }}
          >
            {/* The spinner takes the label's place rather than sitting beside
                it, so pressing moves nothing: the label holds the button's width
                while faded, and the spinner is centered over it by the `relative`
                that `.button` already sets.

                `opacity-0`, not `invisible`: visibility removes the label from
                the accessibility tree, which would leave the button unnamed for
                the whole wait. Faded, it still names the control while the
                spinner reports the state. That is also why the spinner is
                `aria-hidden` (and it is its own live region as of HeroUI 3.2.4,
                which would announce a bare "Loading" over the name).

                `color="current"` because the default is `accent`, which on this
                accent-filled variant paints the spinner in the fill's own
                color; `current` inherits the label's. */}
            <span className={pending ? "opacity-0" : undefined}>
              {entersWorkspace ? "Create and open" : "Create workspace"}
            </span>
            {pending ? (
              <Spinner
                size="sm"
                color="current"
                aria-hidden="true"
                className="absolute inset-0 m-auto"
              />
            ) : null}
          </Button>
          {/* Disabled while the create is in flight, as `ConfirmDialog` does
              with its own: the workspace is already being made, so offering to
              abandon it mid-flight only invites the operator to expect that it
              was not. */}
          <Button variant="ghost" onPress={onClose} isDisabled={pending}>
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
  const narrowed = (defaults.data ?? []).filter(
    (row) => row.provider_key_id !== null,
  )
  const providers = useProviders()
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
          members already here keep the budget they were given, though editing
          that budget still moves them.
        </span>
        <NarrowedDefaults
          workspaceId={workspace.id}
          budgets={budgets.data ?? []}
          narrowed={narrowed}
          providers={(providers.data?.providers ?? []).map(
            (provider) => provider.instance,
          )}
        />
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
