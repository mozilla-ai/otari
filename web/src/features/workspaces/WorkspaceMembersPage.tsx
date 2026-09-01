import { Link } from "@tanstack/react-router"

import { OrganizationRosterCard } from "@/features/organization/OrganizationRosterCard"
import { canManage, canManageWorkspace } from "@/features/organization/roles"
import { WorkspaceMembersPanel } from "@/features/workspaces/WorkspaceMembersPanel"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/hooks"
import { EmptyState, ErrorBanner, PageHeader } from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The roster of the workspace the switcher has selected, which is the one page
// in the workspace context that is genuinely scoped to it today: the workspace
// member routes take a workspace id, so switching here changes what is shown.
// The rest of the context is still deployment-wide until the request plane
// carries a workspace.
//
// Also where a member reads the *organization's* roster (otari-ai#1960). The
// tenant-wide list has a page of its own, Organization > Members & roles, and it
// is on the organization rail, which the shell opens only to a caller who
// manages the organization: the design hides that row from a member outright
// rather than degrading it. So the two callers are answered differently here. A
// manager is pointed at that page, where the roster can also be changed, and
// everyone else reads it below, which is the View the roles matrix asks for.
export function WorkspaceMembersPage() {
  const { selected, isLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  const orgMembers = useOrganizationMembers()

  // An organization's owners and admins manage every workspace in it, and so
  // does an owner/admin of this workspace specifically, which is the rule the
  // workspace service enforces server-side.
  const manages = canManageWorkspace(context.data, selected?.role)
  // Withheld until the context answers, rather than treating a pending read as
  // "does not manage": an owner would otherwise be shown a roster they have a
  // better page for and have it swapped for the pointer to it one paint later.
  // A context that *failed* does read as "does not manage", which offers the
  // roster rather than withholding it, and the server authorizes that read
  // either way.
  const organizationAnswered = !context.isLoading
  const managesOrganization = canManage(context.data)

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Members"
        description={
          selected
            ? `People assigned to ${selected.name} and their role in it. A workspace's members are a subset of the organization's, so someone joins the organization first.`
            : "People assigned to this workspace and their role in it."
        }
      />
      {/* The organization roster is what the panel picks candidates from, and
          what the card below lists, so a failure there is reported rather than
          left to read as "everyone is already in this workspace", which is what
          an empty candidate list otherwise looks like. */}
      <ErrorBanner error={orgMembers.error} />
      {selected ? (
        <WorkspaceMembersPanel
          workspaceId={selected.workspace_id}
          workspaceName={selected.name}
          orgMembers={orgMembers.data ?? []}
          rosterResolved={orgMembers.isSuccess}
          canManageWorkspace={manages}
        />
      ) : (
        // An empty roster would read as "this workspace has nobody in it",
        // which is a different fact from "you are in no workspace". The
        // organization roster below still renders: who else is in the tenant is
        // the answer someone in no workspace yet most needs.
        <EmptyState
          title={isLoading ? "Loading…" : "No workspace selected"}
          description={
            isLoading
              ? "Reading the workspaces you belong to."
              : "You do not belong to a workspace in this organization yet. An owner or admin can add you to one on the Workspaces page."
          }
        />
      )}
      {organizationAnswered && !managesOrganization ? (
        <OrganizationRosterCard
          members={orgMembers.data ?? []}
          isLoading={orgMembers.isPending && !orgMembers.data}
          organizationName={context.data?.organization?.name}
        />
      ) : null}
      {organizationAnswered && managesOrganization ? (
        <p className="text-sm text-muted">
          Everyone in the organization, and the role each of them holds, is on{" "}
          <Link
            to="/organization/members"
            className="font-medium text-link hover:text-link-hover"
          >
            Organization → Members &amp; roles
          </Link>
          .
        </p>
      ) : null}
    </div>
  )
}
