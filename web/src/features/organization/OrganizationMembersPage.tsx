import { Button, Chip } from "@heroui/react"
import { useMemo, useState } from "react"

import type { OrganizationContext, OrganizationMember } from "@/client"
import {
  useOrganizationContext,
  useOrganizationMembers,
  useRemoveOrganizationMember,
  useUpdateOrganizationMember,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"

import {
  canManage,
  MEMBERSHIP_ROLES,
  MEMBERSHIP_STATUSES,
  memberLabel,
  membershipChangeBlockedReason,
} from "./roles"

// The roster of the caller's active organization: who is in it, what role they
// hold, and whether that membership is live. Roles are fixed (owner, admin,
// member, viewer) and the server enforces the same two rules this page disables
// controls for, so a refusal is explained here rather than only reported.

const ROLE_OPTIONS = MEMBERSHIP_ROLES.map((role) => ({
  value: role,
  label: role,
}))
const STATUS_OPTIONS = MEMBERSHIP_STATUSES.map((status) => ({
  value: status,
  label: status,
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

export function OrganizationMembersPage() {
  const context = useOrganizationContext()
  const members = useOrganizationMembers()
  const update = useUpdateOrganizationMember()
  const remove = useRemoveOrganizationMember()

  const [removing, setRemoving] = useState<OrganizationMember | null>(null)

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
                onChange={(role) => {
                  if (member.organization_member_id) {
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
        cell: (member) => {
          const blocked = membershipChangeBlockedReason({
            member,
            context: activeContext,
            members: rows,
          })
          // Read-only for anyone who cannot manage the organization: a chip
          // states the standing without offering a control the server refuses.
          if (blocked !== undefined) {
            return (
              <span title={blocked}>
                <StatusChip status={member.status} />
              </span>
            )
          }
          return (
            <FilterSelect
              ariaLabel={`Status for ${memberLabel(member)}`}
              value={member.status}
              disabled={update.isPending}
              options={STATUS_OPTIONS}
              onChange={(status) => {
                if (member.organization_member_id) {
                  update.mutate({
                    id: member.organization_member_id,
                    body: { status },
                  })
                }
              }}
            />
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
      />

      <ErrorBanner
        error={context.error ?? members.error ?? update.error ?? remove.error}
      />

      {manages ? (
        <InfoBanner>
          A standalone gateway authenticates one operator identity, so this
          roster stays at one row until per-user sign-in and invitations land.
          Roles and statuses here are still what the server authorizes every
          request against.
        </InfoBanner>
      ) : (
        <InfoBanner>
          Only organization owners and admins can change memberships.
        </InfoBanner>
      )}

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
            this organization? Their membership is suspended rather than
            deleted, so past usage stays attributable and they can be
            reactivated here.
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
