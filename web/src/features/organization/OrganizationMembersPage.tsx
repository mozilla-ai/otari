import { Button, Card, Chip } from "@heroui/react"
import { useMemo, useState } from "react"

import type {
  User as ApiUser,
  Budget,
  CreateOrganizationMemberRequest,
  InviteOrganizationMemberRequest,
  InviteOrganizationMemberResult,
  MembershipRole,
  OrganizationContext,
  OrganizationMember,
  ScopedBudget,
  Workspace,
  WorkspaceAssignment,
  WorkspaceBudgetDefault,
  WorkspaceMemberRole,
} from "@/client"
import {
  accessLabel,
  ModelScopeControl,
} from "@/features/models/ModelScopeControl"
import {
  useAddOrganizationMember,
  useAddWorkspaceMember,
  useAllWorkspaceBudgetDefaults,
  useAllWorkspaceMembers,
  useBudgets,
  useCreateScopedBudget,
  useDeleteScopedBudget,
  useInviteOrganizationMember,
  useOrganizationContext,
  useOrganizationMembers,
  useRemoveOrganizationMember,
  useRemoveWorkspaceMember,
  useRevokeOrganizationMemberInvitation,
  useScopedBudgets,
  useUpdateOrganizationMember,
  useUpdateScopedBudget,
  useUpdateUser,
  useUpdateWorkspaceMemberRole,
  useUsers,
  useWorkspaces,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  Checkbox,
  CopyableValue,
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"

import {
  asMembershipRole,
  canManage,
  MEMBERSHIP_ROLES,
  memberLabel,
  membershipChangeBlockedReason,
  membershipLabel,
} from "./roles"

// The roster of the caller's active organization: who is in it, what role they
// hold, and whether that membership is live. Roles are fixed (owner, admin,
// member, viewer) and the server enforces the same two rules this page disables
// controls for, so a refusal is explained here rather than only reported.

// What a member spends and what their keys may call live on the gateway's own
// `users` row, not on the membership: `organization_member` has no such columns.
// `attribution_user_id` is the join, and it is nullable: null when no usable
// gateway row exists, whether because none was ever minted for this member or
// because it was soft-deleted afterwards. Those cells stay empty rather than
// reading as zero, which would claim the person is on the gateway and has spent
// nothing. otari-ai#1727 decides how the two tables converge.
const usd = new Intl.NumberFormat(undefined, {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
})

/** One workspace a person is in, with the ceiling they hold there. */
interface WorkspacePlacement {
  workspaceId: string
  workspaceName: string
  // The `workspace_member` row's id. Carried explicitly rather than read back
  // off the ceiling: a ceiling names a membership, so deriving the membership
  // from the ceiling is null exactly when there is no ceiling yet, which is the
  // case where one is about to be created.
  membershipId: string
  role: string
  ceiling: ScopedBudget | null
}

const ROLE_OPTIONS = MEMBERSHIP_ROLES.map((role) => ({
  value: role,
  label: membershipLabel(role),
}))

// A membership row is keyed by its own id, and a pending invitation has none
// yet, so those fall back to the identity or the address they name.
function memberRowKey(member: OrganizationMember): string {
  return (
    member.organization_member_id ??
    member.invitation_id ??
    member.user_id ??
    member.email ??
    "unknown"
  )
}

function StatusChip({ status }: { status: string }) {
  // Not a membership status: it is the gateway refusing this person's keys, and
  // it is shown here because the membership is active while every request fails.
  if (status === "blocked") {
    return (
      <Chip size="sm" color="danger">
        Blocked
      </Chip>
    )
  }
  if (status === "active") {
    return (
      <Chip size="sm" color="accent">
        Active
      </Chip>
    )
  }
  return (
    <Chip size="sm" color={status === "suspended" ? "warning" : "default"}>
      {membershipLabel(status)}
    </Chip>
  )
}

// Adding someone is an address plus a role, and optionally the workspaces to
// drop them into in the same request. A local identity is created for an address
// nothing else knows yet, which is the handle a future sign-in flow claims it
// by; until then the row is a place to hang a role, which is the point.
function AddMemberForm({ onClose }: { onClose: () => void }) {
  const add = useAddOrganizationMember()
  const workspaces = useWorkspaces()
  const { selected } = useSelectedWorkspace()
  const [email, setEmail] = useState("")
  const [role, setRole] = useState<MembershipRole>("member")
  const [workspaceIds, setWorkspaceIds] = useState<string[]>([])
  const trimmed = email.trim()

  // Seeded once the workspace list answers, and only then: the default is a
  // starting point the operator can clear, not a value re-imposed on every
  // render. Nothing was checked before, so an organization member could be
  // created belonging to no workspace at all, which reads as a working account
  // and behaves like one with nothing in it.
  const rows = workspaces.data
  const [seeded, setSeeded] = useState(false)
  if (!seeded && rows && rows.length > 0) {
    setSeeded(true)
    // The workspace the shell is on, when it is one of this organization's.
    // Otherwise the first, which is the default workspace on a deployment that
    // has not made others.
    const preferred = rows.find(
      (workspace) => workspace.id === selected?.workspace_id,
    )
    setWorkspaceIds([(preferred ?? rows[0]).id])
  }

  const toggleWorkspace = (id: string, checked: boolean) =>
    setWorkspaceIds((current) =>
      checked ? [...current, id] : current.filter((one) => one !== id),
    )

  const submit = () => {
    const body: CreateOrganizationMemberRequest = {
      email: trimmed,
      role,
      // Omitted rather than sent empty: no assignment is not the same request
      // as an empty list of them. The role is stated rather than left to the
      // server's default, so what this form grants is visible in the request
      // it sends and in the copy above it.
      workspace_assignments:
        workspaceIds.length > 0
          ? workspaceIds.map(
              (workspace_id): WorkspaceAssignment => ({
                workspace_id,
                role: "member",
              }),
            )
          : null,
    }
    add.mutate(body, { onSuccess: onClose })
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-title">Add member</div>
        <ErrorBanner error={add.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Email address"
            value={email}
            onChange={setEmail}
            placeholder="alice@example.com"
            isRequired
            autoFocus
            description="The handle this identity is claimed by. Nothing is emailed here; the membership is active straight away. Use Invite member instead to email an accept link."
          />
          <FilterSelect
            label="Role"
            value={role}
            onChange={(value) => setRole(asMembershipRole(value) ?? "member")}
            options={ROLE_OPTIONS}
          />
        </div>
        {workspaces.data && workspaces.data.length > 0 ? (
          <fieldset className="flex flex-col gap-2">
            <legend className="text-sm font-medium text-foreground">
              Workspaces (optional)
            </legend>
            <span className="text-xs text-muted">
              Joined as a member of each, in the same request, so someone never
              exists without the access they were added for. Workspace roles are
              changed afterwards on the Workspaces page.
            </span>
            {workspaceIds.length === 0 ? (
              <span className="text-xs text-warning">
                With none selected they join the organization but no workspace,
                and will see nothing until someone assigns them one.
              </span>
            ) : null}
            {workspaces.data.map((workspace) => (
              <Checkbox
                key={workspace.id}
                isSelected={workspaceIds.includes(workspace.id)}
                onChange={(isSelected) =>
                  toggleWorkspace(workspace.id, isSelected)
                }
              >
                {workspace.name}
              </Checkbox>
            ))}
          </fieldset>
        ) : null}
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={add.isPending}
            onPress={submit}
          >
            Add member
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

// Invites rather than adds: the membership lands `invited`, not `active`, and
// an email with an accept link goes out if mail is configured. Kept separate
// from AddMemberForm rather than a toggle on it: the two produce different
// results (`mail_sent`, `accept_link`) and this one has something to show
// after it succeeds, which AddMemberForm's immediate close does not.
function InviteMemberForm({ onClose }: { onClose: () => void }) {
  const invite = useInviteOrganizationMember()
  const workspaces = useWorkspaces()
  const { mail_ready } = useDeployment()
  const { selected } = useSelectedWorkspace()
  const [email, setEmail] = useState("")
  const [role, setRole] = useState<MembershipRole>("member")
  const [workspaceIds, setWorkspaceIds] = useState<string[]>([])
  const [result, setResult] = useState<InviteOrganizationMemberResult | null>(
    null,
  )
  const trimmed = email.trim()

  const rows = workspaces.data
  const [seeded, setSeeded] = useState(false)
  if (!seeded && rows && rows.length > 0) {
    setSeeded(true)
    const preferred = rows.find(
      (workspace) => workspace.id === selected?.workspace_id,
    )
    setWorkspaceIds([(preferred ?? rows[0]).id])
  }

  const toggleWorkspace = (id: string, checked: boolean) =>
    setWorkspaceIds((current) =>
      checked ? [...current, id] : current.filter((one) => one !== id),
    )

  const submit = () => {
    const body: InviteOrganizationMemberRequest = {
      email: trimmed,
      role,
      workspace_assignments:
        workspaceIds.length > 0
          ? workspaceIds.map(
              (workspace_id): WorkspaceAssignment => ({
                workspace_id,
                role: "member",
              }),
            )
          : null,
    }
    invite.mutate(body, { onSuccess: setResult })
  }

  // After a successful invite: whether it was actually emailed, and the link
  // to share by hand when it was not (or when mail is unconfigured entirely).
  if (result) {
    return (
      <Card>
        <Card.Content className="flex flex-col gap-4 p-5">
          <div className="text-title">Invitation sent</div>
          {result.mail_sent ? (
            <InfoBanner>
              An email with an accept link was sent to{" "}
              <strong>{result.email}</strong>.
            </InfoBanner>
          ) : (
            <InfoBanner>
              {/* Not "mail isn't configured": mail_sent is also false when a
                  configured transport's send failed, and that copy would send
                  an operator to debug a configuration that may be fine. */}
              Otari did not send the email. Share this link with{" "}
              <strong>{result.email}</strong> yourself; it works the same either
              way.
              <div className="mt-2">
                <CopyableValue value={result.accept_link} label="Accept link">
                  <span className="break-all text-xs">
                    {result.accept_link}
                  </span>
                </CopyableValue>
              </div>
            </InfoBanner>
          )}
          <div className="flex gap-2">
            <Button variant="primary" onPress={onClose}>
              Done
            </Button>
          </div>
        </Card.Content>
      </Card>
    )
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-title">Invite member</div>
        <ErrorBanner error={invite.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Email address"
            value={email}
            onChange={setEmail}
            placeholder="alice@example.com"
            isRequired
            autoFocus
            description={
              mail_ready
                ? "An email with an accept link is sent here; the membership becomes active once they follow it."
                : "Invitation email is unavailable, so you will get a link to share with them yourself."
            }
          />
          <FilterSelect
            label="Role"
            value={role}
            onChange={(value) => setRole(asMembershipRole(value) ?? "member")}
            options={ROLE_OPTIONS}
          />
        </div>
        {workspaces.data && workspaces.data.length > 0 ? (
          <fieldset className="flex flex-col gap-2">
            <legend className="text-sm font-medium text-foreground">
              Workspaces (optional)
            </legend>
            <span className="text-xs text-muted">
              Granted once the invitation is accepted, not before.
            </span>
            {workspaces.data.map((workspace) => (
              <Checkbox
                key={workspace.id}
                isSelected={workspaceIds.includes(workspace.id)}
                onChange={(isSelected) =>
                  toggleWorkspace(workspace.id, isSelected)
                }
              >
                {workspace.name}
              </Checkbox>
            ))}
          </fieldset>
        ) : null}
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={invite.isPending}
            onPress={submit}
          >
            Send invitation
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

/**
 * Everything about one person that is not their organization role.
 *
 * One editor rather than a control per column: what their keys may call, which
 * workspaces they are in, and what they may spend in each are the same
 * question asked three ways, and editing them separately meant three round
 * trips through the same row.
 *
 * The three live in different tables, so a save is several writes rather than
 * one. They are ordered: memberships first, then ceilings, because a ceiling is
 * keyed on the *membership* and a workspace someone has just been added to has
 * no membership id until the server answers. The scoped budgets are refetched
 * between the two passes for the same reason, since joining a workspace with a
 * default budget materializes a ceiling server-side that this form then has to
 * edit rather than duplicate.
 */
function MemberEditor({
  member,
  spendRow,
  workspaces,
  budgets,
  defaultByWorkspace,
  placements,
  onClose,
}: {
  member: OrganizationMember
  spendRow: ApiUser | undefined
  workspaces: Workspace[]
  budgets: Budget[]
  // What each workspace hands a new member, used both to say what someone would
  // get and to give a ceiling created here the same cadence.
  defaultByWorkspace: ReadonlyMap<string, WorkspaceBudgetDefault>
  placements: WorkspacePlacement[]
  onClose: () => void
}) {
  const updateUser = useUpdateUser()
  const addMember = useAddWorkspaceMember()
  const removeMember = useRemoveWorkspaceMember()
  const updateRole = useUpdateWorkspaceMemberRole()
  const scopedBudgets = useScopedBudgets()
  const createCeiling = useCreateScopedBudget()
  const updateCeiling = useUpdateScopedBudget()
  const deleteCeiling = useDeleteScopedBudget()

  const initial = useMemo(() => {
    const byWorkspace = new Map(placements.map((p) => [p.workspaceId, p]))
    return new Map(
      workspaces.map((workspace) => {
        const placement = byWorkspace.get(workspace.id)
        return [
          workspace.id,
          {
            member: placement !== undefined,
            role: placement?.role ?? "member",
            // A budget, not a figure. Nothing outside the budgets page maps a
            // cap to an amount, so the period comes with it and there is no
            // cadence to reconcile here.
            budgetId: placement?.ceiling?.budget_id ?? "",
          },
        ]
      }),
    )
  }, [workspaces, placements])

  const [rows, setRows] = useState(initial)
  const [allowedModels, setAllowedModels] = useState<string[] | null>(
    spendRow?.allowed_models ?? null,
  )
  const [scopeValid, setScopeValid] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<unknown>(undefined)

  // "No ceiling" first, then every budget. A workspace's own default is labelled
  // so an operator can tell the inherited one from the rest without leaving the
  // form to look it up.
  const budgetOptions = (fallback: WorkspaceBudgetDefault | undefined) => [
    { value: "", label: "No ceiling" },
    ...budgets.map((budget) => ({
      value: budget.budget_id,
      label:
        budget.budget_id === fallback?.budget_id
          ? `${budget.name ?? budget.budget_id.split("-")[0]} (workspace default)`
          : (budget.name ?? budget.budget_id.split("-")[0]),
    })),
  ]

  const setRow = (
    id: string,
    patch: Partial<{ member: boolean; role: string; budgetId: string }>,
  ) =>
    setRows((current) => {
      const next = new Map(current)
      const existing = next.get(id)
      if (existing) next.set(id, { ...existing, ...patch })
      return next
    })

  const canSave = !saving && scopeValid

  const save = async () => {
    if (!canSave || !member.user_id) return
    setSaving(true)
    setError(undefined)
    try {
      if (spendRow) {
        await updateUser.mutateAsync({
          id: spendRow.user_id,
          body: { allowed_models: allowedModels },
        })
      }

      // Pass one: memberships. The id of anything created here is kept, since
      // a ceiling names the membership and nothing else can resolve it yet.
      // From the membership row, not from the ceiling: reading it off the
      // ceiling was null for anyone already in a workspace who held no ceiling
      // yet, so the create branch below never ran and the form closed reporting
      // success. That is the ordinary path through this page: the member is
      // already in the workspace and is being given a budget for the first time.
      const membershipIds = new Map<string, string | null>(
        placements.map((p) => [p.workspaceId, p.membershipId]),
      )
      const wasMember = new Set(placements.map((p) => p.workspaceId))
      const roleWas = new Map(placements.map((p) => [p.workspaceId, p.role]))
      for (const [workspaceId, row] of rows) {
        if (row.member && !wasMember.has(workspaceId)) {
          const created = await addMember.mutateAsync({
            workspaceId,
            userId: member.user_id,
            role: row.role as WorkspaceMemberRole,
          })
          membershipIds.set(workspaceId, created.id)
        } else if (!row.member && wasMember.has(workspaceId)) {
          await removeMember.mutateAsync({
            workspaceId,
            userId: member.user_id,
          })
        } else if (row.member && roleWas.get(workspaceId) !== row.role) {
          await updateRole.mutateAsync({
            workspaceId,
            userId: member.user_id,
            role: row.role as WorkspaceMemberRole,
          })
        }
      }

      // Pass two: ceilings, against a roster that now includes the joins above
      // and the ceilings their workspaces' defaults just materialized.
      const fresh = await scopedBudgets.refetch()
      const ceilings = new Map(
        (fresh.data ?? [])
          .filter((budget) => budget.scope_type === "workspace_member")
          .map((budget) => [budget.scope_id, budget]),
      )
      for (const [workspaceId, row] of rows) {
        if (!row.member) continue
        const membershipId = membershipIds.get(workspaceId)
        if (!membershipId) continue
        const existing = ceilings.get(membershipId)
        const wanted = row.budgetId === "" ? null : row.budgetId
        if (existing && wanted === null) {
          await deleteCeiling.mutateAsync(existing.id)
        } else if (existing && existing.budget_id !== wanted) {
          await updateCeiling.mutateAsync({
            id: existing.id,
            body: { budget_id: wanted },
          })
        } else if (!existing && wanted !== null) {
          await createCeiling.mutateAsync({
            scope_type: "workspace_member",
            scope_id: membershipId,
            budget_id: wanted,
          })
        }
      }
      onClose()
    } catch (caught) {
      setError(caught)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-5 p-5">
        <div className="text-title">Edit {memberLabel(member)}</div>
        <ErrorBanner error={error} />

        {spendRow ? (
          <ModelScopeControl
            title="Model access (default for this member's keys)"
            description="The models this member's keys may list and call by default. A key can narrow this, but never exceed it."
            initial={spendRow.allowed_models}
            onChange={(value, isValid) => {
              setAllowedModels(value)
              setScopeValid(isValid)
            }}
          />
        ) : (
          <span className="text-xs text-muted">
            No spend row yet, so there is no model access to set. One is minted
            when a key is issued to this member.
          </span>
        )}

        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium text-foreground">
            Workspace access
          </span>
          <div className="max-w-3xl overflow-x-auto">
            <table className="w-full min-w-lg text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="py-1 font-medium">Workspace</th>
                  <th className="py-1 font-medium">Role</th>
                  <th className="py-1 font-medium">Budget</th>
                </tr>
              </thead>
              <tbody>
                {workspaces.map((workspace) => {
                  const row = rows.get(workspace.id)
                  if (!row) return null
                  return (
                    <tr key={workspace.id} className="border-t border-border">
                      <td className="py-1.5">
                        <label className="flex items-center gap-2 text-foreground">
                          <input
                            type="checkbox"
                            checked={row.member}
                            onChange={(event) =>
                              setRow(workspace.id, {
                                member: event.target.checked,
                              })
                            }
                          />
                          {workspace.name}
                        </label>
                      </td>
                      <td className="py-1.5">
                        <FilterSelect
                          ariaLabel={`Role in ${workspace.name}`}
                          value={row.role}
                          onChange={(next) =>
                            setRow(workspace.id, { role: next })
                          }
                          options={ROLE_OPTIONS}
                          disabled={!row.member}
                        />
                      </td>
                      <td className="py-1.5">
                        <FilterSelect
                          ariaLabel={`Budget in ${workspace.name}`}
                          value={row.budgetId}
                          onChange={(next) =>
                            setRow(workspace.id, { budgetId: next })
                          }
                          options={budgetOptions(
                            defaultByWorkspace.get(workspace.id),
                          )}
                          disabled={!row.member}
                        />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <span className="max-w-2xl text-xs text-muted">
            Each workspace holds its own allowance, so someone in two workspaces
            has two. The amount and the reset period belong to the budget, so
            editing one moves everyone held to it; pick a different budget here
            to change only this person. Adding them to a workspace that has a
            default member budget gives them that budget unless another is
            chosen.
          </span>
        </div>

        <div className="flex gap-2">
          <Button variant="primary" isDisabled={!canSave} onPress={save}>
            {saving ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" isDisabled={saving} onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

export function OrganizationMembersPage() {
  const context = useOrganizationContext()
  const members = useOrganizationMembers()
  const update = useUpdateOrganizationMember()
  const remove = useRemoveOrganizationMember()
  const revoke = useRevokeOrganizationMemberInvitation()

  const users = useUsers()
  const updateUser = useUpdateUser()
  const workspaces = useWorkspaces()
  const workspaceIds = useMemo(
    () => (workspaces.data ?? []).map((w) => w.id),
    [workspaces.data],
  )
  const workspaceMembers = useAllWorkspaceMembers(workspaceIds)
  const workspaceDefaults = useAllWorkspaceBudgetDefaults(workspaceIds)
  const budgets = useBudgets()
  const scopedBudgets = useScopedBudgets()

  const [editingMember, setEditingMember] = useState<string | null>(null)
  const [removing, setRemoving] = useState<OrganizationMember | null>(null)
  const [revoking, setRevoking] = useState<OrganizationMember | null>(null)
  const [adding, setAdding] = useState(false)
  const [inviting, setInviting] = useState(false)

  const rows = useMemo(() => members.data ?? [], [members.data])
  const userByAttribution = useMemo(
    () => new Map((users.data ?? []).map((user) => [user.user_id, user])),
    [users.data],
  )

  // Where a person is, and what they may spend there. A workspace ceiling is a
  // `scoped_budgets` row keyed on the *membership* id, not on the person, which
  // is why the roster has to be resolved first: a member in two workspaces holds
  // two memberships and therefore two ceilings, one per workspace.
  const ceilingByMembership = useMemo(
    () =>
      new Map(
        (scopedBudgets.data ?? [])
          .filter((budget) => budget.scope_type === "workspace_member")
          .map((budget) => [budget.scope_id, budget]),
      ),
    [scopedBudgets.data],
  )
  const placementsByUser = useMemo(() => {
    const names = new Map((workspaces.data ?? []).map((w) => [w.id, w.name]))
    const byUser = new Map<string, WorkspacePlacement[]>()
    for (const { workspaceId, member } of workspaceMembers.data) {
      const placement: WorkspacePlacement = {
        workspaceId,
        workspaceName: names.get(workspaceId) ?? workspaceId.slice(0, 8),
        membershipId: member.id,
        role: member.role,
        ceiling: ceilingByMembership.get(member.id) ?? null,
      }
      byUser.set(member.user_id, [
        ...(byUser.get(member.user_id) ?? []),
        placement,
      ])
    }
    return byUser
  }, [workspaces.data, workspaceMembers.data, ceilingByMembership])
  // What each workspace hands a new member: the aggregate default (the one
  // narrowed to no provider). The editor needs it for two reasons: to show what
  // someone would get, and to give a ceiling it creates the same cadence, rather
  // than one that silently never resets.
  const defaultByWorkspace = useMemo(
    () =>
      new Map(
        workspaceDefaults.data
          .filter(({ default: row }) => row.provider_key_id === null)
          .map(({ workspaceId, default: row }) => [workspaceId, row]),
      ),
    [workspaceDefaults.data],
  )
  const activeContext: OrganizationContext | undefined = context.data
  const manages = canManage(activeContext)
  const editingRow =
    rows.find((row) => memberRowKey(row) === editingMember) ?? null

  const columns = useMemo<DataTableColumn<OrganizationMember>[]>(
    () => [
      {
        id: "member",
        header: "Member",
        isRowHeader: true,
        cell: (member) => (
          <div className="flex flex-col gap-0.5">
            <span className="text-sm text-foreground">
              {memberLabel(member)}
            </span>
            {member.email && member.full_name ? (
              <span className="text-xs text-muted">{member.email}</span>
            ) : null}
          </div>
        ),
      },
      {
        id: "role",
        header: "Role",
        cell: (member) => {
          const blocked = membershipChangeBlockedReason({
            member,
            context: activeContext,
            members: rows,
          })
          return (
            // `title` reaches a mouse; the reason is folded into the control's
            // own name so it reaches everyone else too. A disabled control is
            // not focusable, so an `aria-describedby` on it would never be
            // announced either.
            <span title={blocked}>
              <FilterSelect
                ariaLabel={
                  blocked
                    ? `Role for ${memberLabel(member)} (${blocked})`
                    : `Role for ${memberLabel(member)}`
                }
                value={member.role}
                disabled={blocked !== undefined || update.isPending}
                options={ROLE_OPTIONS}
                onChange={(value) => {
                  const role = asMembershipRole(value)
                  if (member.organization_member_id && role) {
                    update.mutate({
                      id: member.organization_member_id,
                      body: { role },
                    })
                  }
                }}
              />
            </span>
          )
        },
      },
      {
        id: "status",
        header: "Status",
        // Shown, not set. The gateway accepts two settable statuses, active and
        // suspended, and suspending is exactly what Remove does one column
        // over, with a confirmation in front of it; a dropdown offering the
        // same thing would be an unconfirmed removal. The other direction has
        // no subject either: a suspended membership leaves the roster
        // (LISTABLE_STATUSES), so there is no row here to reactivate, and
        // re-adding the address revives the membership instead. "invited" has
        // its own control in the Actions column (Revoke) rather than a status
        // a picker could set, for the same reason.
        cell: (member) => {
          const spendRow = member.attribution_user_id
            ? userByAttribution.get(member.attribution_user_id)
            : undefined
          // Blocked outranks the membership status here: the membership is
          // active, and every request the person makes is still refused, which
          // is what someone reading this column wants to know.
          return spendRow?.blocked ? (
            <StatusChip status="blocked" />
          ) : (
            <StatusChip status={member.status} />
          )
        },
      },
      {
        id: "access",
        header: "Model access",
        // The default every key issued to this person inherits. A key may narrow
        // it and never widen it, so this is the ceiling rather than the grant.
        cell: (member) => {
          const spendRow = member.attribution_user_id
            ? userByAttribution.get(member.attribution_user_id)
            : undefined
          if (!spendRow) {
            return <span className="text-xs text-muted">&mdash;</span>
          }
          const { text, tone } = accessLabel(spendRow.allowed_models)
          return (
            <span
              className={
                tone === "danger"
                  ? "text-xs text-danger"
                  : tone === "muted"
                    ? "text-xs text-muted"
                    : "text-xs text-foreground"
              }
            >
              {text}
            </span>
          )
        },
      },
      {
        id: "workspaces",
        header: "Workspaces",
        // A toggle rather than chips: the budget someone holds in a workspace is
        // the other half of the answer, and neither fits in a cell beside the
        // other. The detail panel below carries both.
        cell: (member) => {
          const placements = member.user_id
            ? (placementsByUser.get(member.user_id) ?? [])
            : []
          if (placements.length === 0) {
            return <span className="text-xs text-muted">None</span>
          }
          return (
            <div className="flex flex-wrap gap-1">
              {placements.map((placement) => (
                <Chip key={placement.workspaceId} size="sm">
                  {placement.workspaceName}
                  {placement.ceiling?.max_budget != null
                    ? ` · ${usd.format(placement.ceiling.max_budget)}`
                    : ""}
                </Chip>
              ))}
            </div>
          )
        },
      },
      {
        id: "spend",
        header: "Spend",
        align: "end",
        cell: (member) => {
          const spendRow = member.attribution_user_id
            ? userByAttribution.get(member.attribution_user_id)
            : undefined
          if (!spendRow) {
            return <span className="text-xs text-muted">&mdash;</span>
          }
          return (
            <div className="flex flex-col items-end gap-0.5">
              <span className="text-sm text-foreground">
                {usd.format(spendRow.spend)}
              </span>
              {spendRow.reserved > 0 ? (
                <span className="text-xs text-muted">
                  {usd.format(spendRow.reserved)} in flight
                </span>
              ) : null}
            </div>
          )
        },
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (member) => {
          const blocked = membershipChangeBlockedReason({
            member,
            context: activeContext,
            members: rows,
          })
          // An invited row's only action is Revoke: it has nothing to demote
          // or reassign yet, and Remove's own guard (membershipChangeBlockedReason)
          // already refuses a row with no organization_member_id, which every
          // invited row here has, so Remove would otherwise render enabled and
          // do the wrong thing on a pending invitation.
          if (member.status === "invited" && member.invitation_id) {
            return (
              <Button
                size="sm"
                variant="danger-soft"
                isDisabled={!manages}
                onPress={() => setRevoking(member)}
              >
                Revoke
              </Button>
            )
          }
          // Blocking stops this person's keys from making requests without
          // touching their membership or their spend history, which is a
          // different act from removing them from the organization. It writes
          // the gateway's `users` row, so a member with no attribution row has
          // nothing to block and the control is absent rather than disabled.
          const spendRow = member.attribution_user_id
            ? userByAttribution.get(member.attribution_user_id)
            : undefined
          return (
            <div className="flex items-center justify-end gap-1.5">
              {manages ? (
                <Button
                  size="sm"
                  variant="ghost"
                  onPress={() => setEditingMember(memberRowKey(member))}
                >
                  Edit
                </Button>
              ) : null}
              {manages && spendRow ? (
                <Button
                  size="sm"
                  variant={spendRow.blocked ? "ghost" : "danger-soft"}
                  isDisabled={updateUser.isPending}
                  onPress={() =>
                    updateUser.mutate({
                      id: spendRow.user_id,
                      body: { blocked: !spendRow.blocked },
                    })
                  }
                >
                  {spendRow.blocked ? "Unblock" : "Block"}
                </Button>
              ) : null}
              <span title={blocked}>
                <Button
                  size="sm"
                  variant="danger-soft"
                  // See the Role cell: the reason has to be in the name, not only
                  // in the tooltip, to reach anything but a pointer.
                  aria-label={
                    blocked
                      ? `Remove ${memberLabel(member)} (${blocked})`
                      : undefined
                  }
                  isDisabled={blocked !== undefined}
                  onPress={() => setRemoving(member)}
                >
                  Remove
                </Button>
              </span>
            </div>
          )
        },
      },
    ],
    [
      activeContext,
      rows,
      update.isPending,
      update.mutate,
      manages,
      userByAttribution,
      updateUser.isPending,
      updateUser.mutate,
      placementsByUser,
    ],
  )

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Members"
        description="Who belongs to this organization and what each of them may do. Roles are fixed: owners and admins manage the organization, members use it, viewers only read. Budgets and API keys do not attach to this list; they attach to the gateway identity a member is linked to, which is what lets a key be issued to them by name. A member with no such link yet shows no access or spend, and cannot own a key until one exists."
        action={
          manages && !adding && !inviting ? (
            <div className="flex gap-2">
              <Button variant="ghost" onPress={() => setAdding(true)}>
                Add member
              </Button>
              <Button variant="primary" onPress={() => setInviting(true)}>
                Invite member
              </Button>
            </div>
          ) : null
        }
      />

      {/* `remove.error`/`revoke.error` are deliberately absent: their confirm
          dialogs render each mutation's error themselves, and listing it here
          too paints the same message twice, once behind the open dialog. */}
      <ErrorBanner
        error={
          context.error ??
          members.error ??
          update.error ??
          // The reads and the write the row's own controls use. Without these a
          // failed roster renders the access, workspace and spend cells empty as
          // though the member simply had none, and a refused Block says nothing.
          users.error ??
          updateUser.error ??
          workspaces.error ??
          scopedBudgets.error
        }
      />

      {/* Withheld until the context answers. Rendering the refusal first shows
          an owner "you cannot change memberships" for one paint and then takes
          it back, which reads as a permissions bug rather than a load. */}
      {context.isLoading || manages ? null : (
        <InfoBanner>
          Only organization owners and admins can change memberships.
        </InfoBanner>
      )}

      {adding ? <AddMemberForm onClose={() => setAdding(false)} /> : null}
      {inviting ? (
        <InviteMemberForm onClose={() => setInviting(false)} />
      ) : null}

      {/* Keyed on the row so switching which member is edited remounts the
          form: its fields seed from the member on mount only. */}
      {editingRow ? (
        <MemberEditor
          key={memberRowKey(editingRow)}
          member={editingRow}
          spendRow={
            editingRow.attribution_user_id
              ? userByAttribution.get(editingRow.attribution_user_id)
              : undefined
          }
          workspaces={workspaces.data ?? []}
          budgets={budgets.data ?? []}
          defaultByWorkspace={defaultByWorkspace}
          placements={
            editingRow.user_id
              ? (placementsByUser.get(editingRow.user_id) ?? [])
              : []
          }
          onClose={() => setEditingMember(null)}
        />
      ) : null}

      <DataTable
        ariaLabel="Organization members"
        columns={columns}
        rows={rows}
        getRowKey={memberRowKey}
        isLoading={members.isLoading}
        emptyContent="No members yet."
      />

      <ConfirmDialog
        isOpen={removing !== null}
        onOpenChange={(open) => {
          if (!open) setRemoving(null)
        }}
        heading="Remove member"
        body={
          <>
            Remove <strong>{removing ? memberLabel(removing) : ""}</strong> from
            this organization? The membership is suspended rather than deleted,
            so anything already attributed to them still resolves, and the row
            leaves this roster. Adding the same address again revives that
            membership rather than starting a second one.
          </>
        }
        confirmLabel="Remove member"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (removing?.organization_member_id) {
            remove.mutate(removing.organization_member_id, {
              onSuccess: () => setRemoving(null),
            })
          }
        }}
      />

      <ConfirmDialog
        isOpen={revoking !== null}
        onOpenChange={(open) => {
          if (!open) setRevoking(null)
        }}
        heading="Revoke invitation"
        body={
          <>
            Revoke the invitation to{" "}
            <strong>{revoking ? memberLabel(revoking) : ""}</strong>? Their
            accept link stops working, and the membership is suspended rather
            than deleted. Inviting the same address again revives it.
          </>
        }
        confirmLabel="Revoke invitation"
        isPending={revoke.isPending}
        error={revoke.error}
        onConfirm={() => {
          if (revoking?.invitation_id) {
            revoke.mutate(revoking.invitation_id, {
              onSuccess: () => setRevoking(null),
            })
          }
        }}
      />
    </div>
  )
}
