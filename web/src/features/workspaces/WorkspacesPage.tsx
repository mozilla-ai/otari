import { Button, Spinner } from "@heroui/react"
import { useEffect, useMemo, useRef, useState } from "react"

import type { Budget, Workspace, WorkspaceBudgetDefault } from "@/client"
import { canManage, isDeploymentOperator } from "@/features/organization/roles"
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
  PageIntro,
  RowAction,
  RowActionRow,
  Section,
  TableScrollFrame,
} from "@/shared/components/surface"
import {
  EmptyState,
  ErrorBanner,
  errorMessage,
  FilterSelect,
  InfoBanner,
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
      <span className="text-body">Per-provider defaults</span>
      <ErrorBanner
        error={
          createDefault.error ?? updateDefault.error ?? deleteDefault.error
        }
      />
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
// acknowledging them, not as the gateway being slow. How long the shipped beat
// lasts, and nothing a test has to sit through: `hold` below is the seam for
// that, and it is a gate rather than a duration so a test can own when the beat
// ends instead of guessing at a margin over this number.
const ENTER_HOLD_MS = 800

// The shipped beat, as a gate. A caller that wants to own when it opens (the
// tests do, so the dismissal window is entered and left on purpose rather than
// slept through) passes its own.
const defaultHold = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ENTER_HOLD_MS)
  })

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
  hold = defaultHold,
}: {
  onClose: () => void
  /**
   * The acknowledged-press beat, as a gate rather than a duration. Defaults to
   * `ENTER_HOLD_MS` of wall clock; a test supplies one it opens itself, so the
   * dismissal window is entered and left deterministically instead of by a
   * sleep long enough to have outlasted it.
   */
  hold?: () => Promise<void>
  // Fired once the workspace exists (and its default budget, when one was
  // picked), so the caller that opened the form can follow the new workspace.
  // The list page does not need it: it is already looking at the row that
  // appeared. The scope switcher does, because creating from there is a request
  // to work in the new workspace, not just to add it.
  onCreated?: (workspace: Workspace) => void
}) {
  const create = useCreateWorkspace()
  const createDefault = useCreateWorkspaceBudgetDefault()
  // The budgets list is a deployment-wide read that answers 403 to an
  // organization admin who does not operate the deployment (#821), so it is not
  // asked for unless the caller may read it, and the default-budget picker it
  // feeds is withheld with it (the OrganizationMembersPage pattern, otari#838):
  // offered anyway, the picker could only say "No default", which misreads as
  // the deployment having no budgets. Resolved here rather than passed in
  // because the workspace switcher offers this same form.
  const context = useOrganizationContext()
  const operates = isDeploymentOperator(context.data)
  const budgets = useBudgets(operates)
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
  // Set on the way in as well as cleared on the way out. StrictMode remounts
  // every component once in development (`main.tsx` wraps the app in it), which
  // runs mount, cleanup, mount: a cleanup-only effect would leave this `false`
  // for the rest of the session, and every create would then return early and
  // leave the modal open with the button spinning. `VerifyEmailPage` carries the
  // same hazard for the same reason.
  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])
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
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 className="text-title">Create workspace</h2>
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
        // Dropped while the refusal is up, rather than shown above it: `Field`
        // renders the two as separate rows, and a modal that grows a line when
        // it reports a problem moves the fields under the pointer that caused
        // it. The refusal is the more useful of the two at that moment.
        description={
          nameRefusal
            ? undefined
            : "Unique within this organization. You become its owner."
        }
      />
      <Field
        label="Description (optional)"
        value={description}
        onChange={setDescription}
      />
      {/* Withheld from a caller who does not operate the deployment: the
          picker's options come from the operator-gated `/v1/budgets` read, so
          offering it would be offering a control whose list is empty and whose
          save cannot succeed. */}
      {operates ? (
        <DefaultBudgetPicker
          budgets={budgets.data ?? []}
          value={budgetId}
          onChange={setBudgetId}
        />
      ) : null}
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
            const held = entersWorkspace ? hold() : Promise.resolve()
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
    </Section>
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
  // Budgets and providers are deployment-wide reads that answer 403 to an
  // organization admin who does not operate the deployment (#821), so neither is
  // asked for unless the caller may read it, and the default-budget controls
  // they feed are withheld with them (the OrganizationMembersPage pattern,
  // otari#838): rendered anyway, the picker showed the current default as a raw
  // UUID and the per-provider section had no provider to offer. The defaults
  // read itself is workspace-scoped and would answer, but this form only reads
  // it into those controls, so it is declined together with them.
  const context = useOrganizationContext()
  const operates = isDeploymentOperator(context.data)
  const budgets = useBudgets(operates)
  const defaults = useWorkspaceBudgetDefaults(operates ? workspace.id : null)
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
  const providers = useProviders(operates)
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
    // With the picker withheld, an untouched `selectedBudget` over an unfetched
    // defaults list would read as "none" and delete nothing, but say so rather
    // than lean on that coincidence.
    if (!operates) return
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
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <h2 className="text-title">
        Edit <code>{workspace.name}</code>
      </h2>
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
      {/* Withheld from a caller who does not operate the deployment: the
          picker's options come from the operator-gated `/v1/budgets` read, so
          offering it would be offering a control whose list is empty and whose
          save cannot succeed. */}
      {operates ? (
        <>
          <DefaultBudgetPicker
            budgets={budgets.data ?? []}
            value={selectedBudget}
            onChange={setBudgetId}
          />
          <span className="max-w-md text-xs text-muted">
            Every member of this workspace is held to this budget, each with
            their own allowance. Changing it applies to members who join
            afterwards; members already here keep the budget they were given,
            though editing that budget still moves them.
          </span>
          <NarrowedDefaults
            workspaceId={workspace.id}
            budgets={budgets.data ?? []}
            narrowed={narrowed}
            providers={(providers.data?.providers ?? []).map(
              (provider) => provider.instance,
            )}
          />
        </>
      ) : null}
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
    </Section>
  )
}

export function WorkspacesPage() {
  const context = useOrganizationContext()
  const workspaces = useWorkspaces()
  // The budgets read is deployment-wide and answers 403 to an organization
  // admin who does not operate the deployment (#821), so it is not asked for
  // unless the caller may read it, and the column it names is withheld with it
  // (the OrganizationMembersPage pattern, otari#838): without the names, every
  // cell could only echo a UUID fragment of the default's id.
  const operates = isDeploymentOperator(context.data)
  const budgets = useBudgets(operates)
  const remove = useDeleteWorkspace()

  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<Workspace | null>(null)

  const rows = workspaces.data ?? []
  const workspaceIds = useMemo(() => rows.map((row) => row.id), [rows])
  // Emptied rather than fetched for a non-operator: the fan-out itself is
  // workspace-scoped and would answer, but this page only reads it into the
  // withheld column below.
  const workspaceDefaults = useAllWorkspaceBudgetDefaults(
    operates ? workspaceIds : [],
  )
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

  // The default-budget column is dropped, not emptied, for a caller who cannot
  // read the budget names it shows; see the note on `operates` above.
  const columns = useMemo<DataTableColumn<Workspace>[]>(() => {
    const all: DataTableColumn<Workspace>[] = [
      {
        id: "name",
        header: "Workspace",
        isRowHeader: true,
        cell: (workspace) => (
          <div className="flex flex-col gap-0.5">
            <span className="text-sm font-medium text-foreground">
              {workspace.name}
            </span>
            {/* One line, truncated. Capping this lane so the columns after it
                stay adjacent is what made the description wrap and took two
                rows off the 58px pitch, which is the same trade the members
                table already resolved this way: the row keeps its rhythm and
                the full text stays in the title. */}
            {workspace.description ? (
              <span
                className="truncate text-xs text-muted"
                title={workspace.description}
              >
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
          // A name, not a chip. It is the budget's own name and nothing
          // else in the row is boxed, so the box was the only thing making it
          // look like a different kind of value from its neighbors.
          return name ? (
            <span className="text-sm text-foreground">{name}</span>
          ) : (
            <span className="text-xs text-subtle">None</span>
          )
        },
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (workspace) => (
          <RowActionRow>
            <RowAction
              isDisabled={!manages}
              onPress={() => {
                setCreating(false)
                setEditing(workspace.id)
              }}
            >
              Edit
            </RowAction>
            {/* The server keeps every organization on at least one workspace
                (`LastWorkspaceError`), and first boot is the one-workspace
                state, so the ordinary case would be a button that always
                refuses. Say why instead of offering the refusal. The reason
                goes in the name, following the membership controls: a disabled
                control takes no focus, so a tooltip would reach a pointer and
                nothing else. */}
            <span title={isOnlyWorkspace ? LAST_WORKSPACE_REASON : undefined}>
              <RowAction
                ariaLabel={
                  isOnlyWorkspace
                    ? `Delete ${workspace.name} (${LAST_WORKSPACE_REASON})`
                    : undefined
                }
                isDisabled={!manages || isOnlyWorkspace}
                onPress={() => setDeleting(workspace)}
              >
                Delete
              </RowAction>
            </span>
          </RowActionRow>
        ),
      },
    ]
    return all.filter((column) => operates || column.id !== "default-budget")
  }, [manages, isOnlyWorkspace, defaultBudgetName, operates])

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Workspaces"
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
      >
        The scopes inside this organization that work is grouped into. Each one
        carries its own members and roles.
      </PageIntro>

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
        <TableScrollFrame className="otari-workspaces-table">
          <DataTable
            ariaLabel="Workspaces"
            columns={columns}
            rows={rows}
            getRowKey={getWorkspaceRowKey}
            isLoading={workspaces.isLoading}
            emptyContent="No workspaces yet."
          />
        </TableScrollFrame>
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
