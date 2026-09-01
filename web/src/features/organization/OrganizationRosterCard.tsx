import { Card } from "@heroui/react"

import type { OrganizationMember } from "@/client"
import { MembershipStatusChip } from "@/features/organization/MembershipStatusChip"
import {
  memberLabel,
  memberRowKey,
  membershipLabel,
} from "@/features/organization/roles"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"

// Who belongs to the caller's organization, read-only, for a caller who is not
// offered Organization > Members & roles: the roles matrix has the members list
// at View for a member (otari-ai#1960), and that page sits on the organization
// rail, which the shell opens only to someone who manages the organization.
// Rendered on the workspace Members page, beside the roster of the workspace
// itself, which is the subset.
//
// Three columns of the management page's seven. The rest are either a control a
// member cannot use or a deployment-wide read the server refuses them.
export function OrganizationRosterCard({
  members,
  isLoading,
  organizationName,
}: {
  members: OrganizationMember[]
  isLoading: boolean
  organizationName: string | undefined
}) {
  const columns: DataTableColumn<OrganizationMember>[] = [
    {
      id: "member",
      header: "Member",
      isRowHeader: true,
      cell: (member) => (
        <div className="flex flex-col gap-0.5">
          <span className="text-body">{memberLabel(member)}</span>
          {member.email && member.full_name ? (
            <span className="text-caption">{member.email}</span>
          ) : null}
        </div>
      ),
    },
    {
      id: "role",
      header: "Role",
      cell: (member) => (
        <span className="text-body">{membershipLabel(member.role)}</span>
      ),
    },
    {
      id: "status",
      header: "Status",
      cell: (member) => <MembershipStatusChip status={member.status} />,
    },
  ]

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-title">Organization members</h2>
      <p className="text-sm text-muted">
        Everyone in {organizationName ?? "this organization"} and the role they
        hold in it. Roles are set by an owner or admin, and a workspace's
        members are always a subset of this list.
      </p>
      <Card>
        <Card.Content className="p-0">
          <DataTable
            ariaLabel="Organization members"
            columns={columns}
            rows={members}
            getRowKey={memberRowKey}
            isLoading={isLoading}
            // Says nothing about the organization: an empty table is also what a
            // failed read leaves behind, and the page's banner is the only thing
            // that knows which happened.
            emptyContent="No members to show."
          />
        </Card.Content>
      </Card>
    </section>
  )
}
