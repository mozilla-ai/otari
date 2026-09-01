import { Button, Chip } from "@heroui/react"
import { useMemo, useState } from "react"

import type {
  OrganizationMember,
  WorkspaceMember,
  WorkspaceMemberRole,
} from "@/client"
import {
  asMembershipRole,
  MEMBERSHIP_ROLES,
  memberLabel,
  membershipLabel,
} from "@/features/organization/roles"
import {
  useAddWorkspaceMember,
  useRemoveWorkspaceMember,
  useUpdateWorkspaceMemberRole,
  useWorkspaceMembers,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { ErrorBanner, FilterSelect, InfoBanner } from "@/shared/components/ui"

// A workspace's roster, shared by the two places one is shown: expanded inside
// a row on the Workspaces page, and as the whole of the Members page in the
// workspace context. Extracted rather than duplicated because the rules it
// encodes (a workspace's members are a subset of the organization's, and the
// roles are the organization's four) belong to the roster, not to either page.

// The workspace vocabulary is the organization one: four fixed roles, the same
// spellings, published on both requests. `asMembershipRole` narrows a picker's
// string back to it.
const ROLE_OPTIONS = MEMBERSHIP_ROLES.map((role) => ({
  value: role,
  label: membershipLabel(role),
}))

function AddWorkspaceMember({
  workspaceId,
  candidates,
  rosterResolved,
}: {
  workspaceId: string
  candidates: OrganizationMember[]
  /**
   * Whether the organization roster actually answered. An empty candidate list
   * means "everyone is already here" only once it has: while it is loading, or
   * after it failed, the list is empty for a reason the operator should not be
   * told is a full workspace.
   */
  rosterResolved: boolean
}) {
  const add = useAddWorkspaceMember()
  const [userId, setUserId] = useState("")
  const [role, setRole] = useState<WorkspaceMemberRole>("member")

  if (!rosterResolved) {
    return null
  }

  if (candidates.length === 0) {
    return (
      <InfoBanner>
        Every active member of this organization is already in this workspace. A
        workspace's members are always a subset of the organization's, so add
        someone there first, on the Members page.
      </InfoBanner>
    )
  }

  return (
    <div className="flex flex-wrap items-end gap-2">
      <FilterSelect
        label="Organization member"
        value={userId}
        onChange={setUserId}
        options={[
          { value: "", label: "Select a member…" },
          ...candidates.map((member) => ({
            value: member.user_id ?? "",
            label: memberLabel(member),
          })),
        ]}
      />
      <FilterSelect
        label="Role"
        value={role}
        onChange={(value) => setRole(asMembershipRole(value) ?? "member")}
        options={ROLE_OPTIONS}
      />
      <Button
        variant="primary"
        isDisabled={userId === ""}
        isPending={add.isPending}
        onPress={() =>
          add.mutate(
            { workspaceId, userId, role },
            { onSuccess: () => setUserId("") },
          )
        }
      >
        Add member
      </Button>
      <ErrorBanner error={add.error} />
    </div>
  )
}

export function WorkspaceMembersPanel({
  workspaceId,
  workspaceName,
  orgMembers,
  rosterResolved,
  canManageWorkspace,
}: {
  // Id and name rather than a Workspace: the Members page reaches this holding
  // only the caller's membership, which carries both and nothing else.
  workspaceId: string
  workspaceName: string
  orgMembers: OrganizationMember[]
  rosterResolved: boolean
  canManageWorkspace: boolean
}) {
  const members = useWorkspaceMembers(workspaceId)
  const updateRole = useUpdateWorkspaceMemberRole()
  const removeMember = useRemoveWorkspaceMember()
  const [removing, setRemoving] = useState<WorkspaceMember | null>(null)

  const rows = members.data ?? []
  const nameByUserId = useMemo(
    () =>
      new Map(
        orgMembers
          .filter((member) => member.user_id)
          .map((member) => [member.user_id as string, memberLabel(member)]),
      ),
    [orgMembers],
  )
  const present = new Set(rows.map((member) => member.user_id))
  const candidates = orgMembers.filter(
    (member) =>
      member.user_id &&
      member.status === "active" &&
      !present.has(member.user_id),
  )

  return (
    <div className="flex flex-col gap-4 p-4">
      <h2 className="text-title">Members of {workspaceName}</h2>
      <ErrorBanner
        error={members.error ?? updateRole.error ?? removeMember.error}
      />

      {members.isLoading ? (
        <p className="text-sm text-muted">Loading members…</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-muted">This workspace has no members yet.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((member) => (
            <li
              key={member.id}
              className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface-alt px-3 py-2"
            >
              <span className="text-sm text-foreground">
                {nameByUserId.get(member.user_id) ??
                  `Identity ${member.user_id.slice(0, 8)}`}
              </span>
              <Chip
                size="sm"
                color={member.status === "active" ? "accent" : "default"}
              >
                {member.status}
              </Chip>
              <span className="ml-auto flex items-center gap-2">
                <FilterSelect
                  ariaLabel={`Role for ${nameByUserId.get(member.user_id) ?? member.user_id} in ${workspaceName}`}
                  value={member.role}
                  disabled={!canManageWorkspace || updateRole.isPending}
                  options={ROLE_OPTIONS}
                  onChange={(value) => {
                    const role = asMembershipRole(value)
                    if (role) {
                      updateRole.mutate({
                        workspaceId: workspaceId,
                        userId: member.user_id,
                        role,
                      })
                    }
                  }}
                />
                <Button
                  size="sm"
                  variant="danger-soft"
                  isDisabled={!canManageWorkspace}
                  onPress={() => setRemoving(member)}
                >
                  Remove
                </Button>
              </span>
            </li>
          ))}
        </ul>
      )}

      {canManageWorkspace ? (
        <AddWorkspaceMember
          workspaceId={workspaceId}
          candidates={candidates}
          rosterResolved={rosterResolved}
        />
      ) : null}

      <ConfirmDialog
        isOpen={removing !== null}
        onOpenChange={(open) => {
          if (!open) setRemoving(null)
        }}
        heading="Remove workspace member"
        body={
          <>
            Remove{" "}
            <strong>
              {removing
                ? (nameByUserId.get(removing.user_id) ?? removing.user_id)
                : ""}
            </strong>{" "}
            from {workspaceName}? They keep their organization membership and
            can be added back.
          </>
        }
        confirmLabel="Remove member"
        isPending={removeMember.isPending}
        error={removeMember.error}
        onConfirm={() => {
          if (removing) {
            removeMember.mutate(
              { workspaceId: workspaceId, userId: removing.user_id },
              { onSuccess: () => setRemoving(null) },
            )
          }
        }}
      />
    </div>
  )
}
