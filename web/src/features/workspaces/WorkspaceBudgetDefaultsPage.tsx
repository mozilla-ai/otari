import { canManageWorkspace } from "@/features/organization/roles"
import { WorkspaceBudgetDefaultsPanel } from "@/features/workspaces/WorkspaceBudgetDefaultsPanel"
import { useOrganizationContext } from "@/shared/api/hooks"
import { EmptyState, PageHeader } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The budget-default templates of the workspace the switcher has selected,
// same scoping as `WorkspaceMembersPage`. A default only ever governs members
// of the one workspace it belongs to, so there is no organization-wide view.
export function WorkspaceBudgetDefaultsPage() {
  const { selected, isLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()

  // An organization's owners and admins manage every workspace's defaults,
  // and so does an owner/admin of this workspace specifically, which is the
  // rule `WorkspaceBudgetDefaultService` enforces server-side.
  const manages = canManageWorkspace(context.data, selected?.role)

  if (!selected) {
    return (
      <div className="flex flex-col gap-6">
        <PageHeader
          title="Budget defaults"
          description="Per-member spend templates for this workspace."
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
        title="Budget defaults"
        description={`Per-member spend templates for ${selected.name}. Creating one gives every current member a matching budget immediately; a member who joins later gets one too.`}
      />
      <WorkspaceBudgetDefaultsPanel
        workspaceId={selected.workspace_id}
        workspaceName={selected.name}
        canManageWorkspace={manages}
      />
    </div>
  )
}
