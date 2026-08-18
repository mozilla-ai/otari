import { canManage } from "@/features/organization/roles"
import { WorkspaceMembersPanel } from "@/features/workspaces/WorkspaceMembersPanel"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/hooks"
import { EmptyState, PageHeader } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The roster of the workspace the switcher has selected, which is the one page
// in the workspace context that is genuinely scoped to it today: the workspace
// member routes take a workspace id, so switching here changes what is shown.
// The rest of the context is still deployment-wide until the request plane
// carries a workspace.
//
// Distinct from Organization > Members & roles, which is who belongs to the
// tenant. A workspace's members are always a subset of that, so someone is
// added to the organization first and assigned here second.
export function WorkspaceMembersPage() {
  const { selected, isLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  const orgMembers = useOrganizationMembers()

  // Workspace membership is not what grants management here: the organization's
  // owners and admins manage every workspace in it, which is the rule the
  // workspace service enforces server-side.
  const manages = canManage(context.data)

  if (!selected) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader
          title="Members"
          description="People assigned to this workspace and their role in it."
        />
        <EmptyState
          title={isLoading ? "Loading…" : "No workspace selected"}
          description={
            isLoading
              ? "Reading the workspaces you belong to."
              : "You do not belong to a workspace in this organization yet. An owner or admin can add you to one on the Workspaces page."
          }
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Members"
        description={`People assigned to ${selected.name} and their role in it. A workspace's members are a subset of the organization's, so someone joins the organization first.`}
      />
      <WorkspaceMembersPanel
        workspaceId={selected.workspace_id}
        workspaceName={selected.name}
        orgMembers={orgMembers.data ?? []}
        rosterResolved={orgMembers.isSuccess}
        canManageWorkspace={manages}
      />
    </div>
  )
}
