import { Button, Card, Chip } from "@heroui/react"
import { useMemo, useState } from "react"

import type {
  CreateOrganizationMemberRequest,
  MembershipRole,
  OrganizationContext,
  OrganizationMember,
  WorkspaceAssignment,
} from "@/client"
import {
  useAddOrganizationMember,
  useOrganizationContext,
  useOrganizationMembers,
  useRemoveOrganizationMember,
  useUpdateOrganizationMember,
  useWorkspaces,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import {
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

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
  if (status === "active") {
    return (
      <Chip size="sm" color="accent">
        Active
      </Chip>
    )
  }
  return (
    <Chip size="sm" color={status === "suspended" ? "warning" : "default"}>
      {status}
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
        <div className="text-sm font-semibold text-foreground">Add member</div>
        <ErrorBanner error={add.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Email address"
            value={email}
            onChange={setEmail}
            placeholder="alice@example.com"
            isRequired
            autoFocus
            description="The handle this identity is claimed by. Nothing is emailed: a standalone gateway has no invitation to send, so the membership is active straight away."
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
              <label
                key={workspace.id}
                className="flex w-fit items-center gap-2 text-sm text-foreground"
              >
                <input
                  type="checkbox"
                  checked={workspaceIds.includes(workspace.id)}
                  onChange={(event) =>
                    toggleWorkspace(workspace.id, event.target.checked)
                  }
                />
                {workspace.name}
              </label>
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

export function OrganizationMembersPage() {
  const context = useOrganizationContext()
  const members = useOrganizationMembers()
  const update = useUpdateOrganizationMember()
  const remove = useRemoveOrganizationMember()

  const [removing, setRemoving] = useState<OrganizationMember | null>(null)
  const [adding, setAdding] = useState(false)

  const rows = useMemo(() => members.data ?? [], [members.data])
  const activeContext: OrganizationContext | undefined = context.data
  const manages = canManage(activeContext)

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
            <span title={blocked}>
              <FilterSelect
                ariaLabel={`Role for ${memberLabel(member)}`}
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
        // re-adding the address revives the membership instead. "invited"
        // arrives with the invitation flow and gets its own control then.
        cell: (member) => <StatusChip status={member.status} />,
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
          return (
            <span title={blocked}>
              <Button
                size="sm"
                variant="danger-soft"
                isDisabled={blocked !== undefined}
                onPress={() => setRemoving(member)}
              >
                Remove
              </Button>
            </span>
          )
        },
      },
    ],
    [activeContext, rows, update.isPending, update.mutate],
  )

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Members"
        description="Who belongs to this organization and what each of them may do. Roles are fixed: owners and admins manage the organization, members use it, viewers only read."
        action={
          manages && !adding ? (
            <Button variant="primary" onPress={() => setAdding(true)}>
              Add member
            </Button>
          ) : null
        }
      />

      <ErrorBanner
        error={context.error ?? members.error ?? update.error ?? remove.error}
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
    </div>
  )
}
