import { Link } from "@tanstack/react-router"
import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Toolbar } from "@/design-system/layout/Toolbar"
import { OrganizationRosterCard } from "@/features/organization/OrganizationRosterCard"
import {
  canManage,
  canManageWorkspace,
  memberLabel,
} from "@/features/organization/roles"
import {
  AddWorkspaceMemberDialog,
  memberNames,
  WorkspaceMembersPanel,
  workspaceMemberLabel,
} from "@/features/workspaces/WorkspaceMembersPanel"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { useWorkspaceMembers } from "@/shared/api/workspaces"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { matchesSearch, useListSearch } from "@/shared/hooks/useListSearch"

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
  // The same query the panel below reads, so this costs no second request. It
  // is here because who is *already* in the workspace is what makes the rest of
  // the organization a candidate list, and the control that uses that list now
  // sits in the heading row.
  const members = useWorkspaceMembers(selected?.workspace_id ?? null)
  const search = useListSearch()
  const [isAdding, setIsAdding] = useState(false)
  const [openCount, setOpenCount] = useState(0)

  // An organization's owners and admins manage every workspace in it, and so
  // does an owner/admin of this workspace specifically, which is the rule the
  // workspace service enforces server-side.
  const canManageSelectedWorkspace = canManageWorkspace(
    context.data,
    selected?.role,
  )
  // Withheld until the context answers, rather than treating a pending read as
  // "does not manage": an owner would otherwise be shown a roster they have a
  // better page for and have it swapped for the pointer to it one paint later.
  // A context that *failed* does read as "does not manage", which offers the
  // roster rather than withholding it, and the server authorizes that read
  // either way.
  const hasOrganizationAnswered = !context.isLoading
  const canManageOrganization = canManage(context.data)

  // Both reads, not just the organization's. The candidate list is the
  // organization minus whoever is already in the workspace, so answering it
  // from one read offers exactly the people the other one was about to
  // exclude. Empty until both have landed, which is also what makes
  // "everyone is already here" safe to say.
  const bothRostersAnswered = orgMembers.isSuccess && members.isSuccess
  const present = new Set((members.data ?? []).map((member) => member.user_id))
  const candidates = bothRostersAnswered
    ? (orgMembers.data ?? []).filter(
        (member) =>
          member.user_id &&
          member.status === "active" &&
          !present.has(member.user_id),
      )
    : []

  // People, not rows: someone in the workspace is also on the organization
  // roster below it, and is one match however many lists show them.
  const names = memberNames(orgMembers.data ?? [])
  const showsRoster = hasOrganizationAnswered && !canManageOrganization
  const matched = new Set([
    ...(selected ? (members.data ?? []) : [])
      .filter((member) =>
        matchesSearch(search.term, workspaceMemberLabel(member, names)),
      )
      .map((member) => member.user_id),
    ...(showsRoster ? (orgMembers.data ?? []) : [])
      .filter((member) =>
        matchesSearch(search.term, memberLabel(member), member.email ?? ""),
      )
      .map((member) => member.user_id ?? member.email ?? ""),
  ])
  // Withheld until every roster it counts has answered, and when one failed:
  // a roster still loading, or lost, would count as nobody.
  const rostersAnswered =
    orgMembers.isSuccess && (!selected || members.isSuccess)

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Members"
        action={
          selected && canManageSelectedWorkspace ? (
            <Button
              variant="primary"
              onPress={() => {
                setOpenCount((count) => count + 1)
                setIsAdding(true)
              }}
            >
              Add member
            </Button>
          ) : null
        }
      >
        {selected
          ? `People assigned to ${selected.name} and their role in it. A workspace's members are a subset of the organization's, so someone joins the organization first.`
          : "People assigned to this workspace and their role in it."}
      </PageIntro>
      {/* Keyed on the open count, so each open remounts a blank form. Clearing
          the draft on close instead would blank the fields while the dialog is
          still animating away. */}
      {selected ? (
        <AddWorkspaceMemberDialog
          key={openCount}
          isOpen={isAdding}
          onClose={() => setIsAdding(false)}
          workspaceId={selected.workspace_id}
          candidates={candidates}
          rosterResolved={bothRostersAnswered}
          canManageOrganization={canManageOrganization}
          // The dialog closes with no other word that the member was added, so
          // a search they do not match would hide them: adding ends the search.
          onAdded={() => search.onChange("")}
        />
      ) : null}
      {/* The organization roster is what the panel picks candidates from, and
          what the card below lists, so a failure there is reported rather than
          left to read as "everyone is already in this workspace", which is what
          an empty candidate list otherwise looks like. */}
      <ErrorBanner error={orgMembers.error} />
      {/* One search for both lists: who is in this workspace, and the
          organization roster beside it, are the same people at two scopes. */}
      <Toolbar>
        <SearchField
          label="Search members"
          placeholder="Search by name"
          value={search.value}
          onChange={search.onChange}
          resultCount={rostersAnswered ? matched.size : undefined}
          className="w-full md:w-96"
        />
      </Toolbar>
      {selected ? (
        <WorkspaceMembersPanel
          workspaceId={selected.workspace_id}
          workspaceName={selected.name}
          orgMembers={orgMembers.data ?? []}
          canManageWorkspace={canManageSelectedWorkspace}
          searchTerm={search.term}
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
      {hasOrganizationAnswered && !canManageOrganization ? (
        <OrganizationRosterCard
          members={orgMembers.data ?? []}
          isLoading={orgMembers.isPending && !orgMembers.data}
          organizationName={context.data?.organization?.name}
          searchTerm={search.term}
        />
      ) : null}
      {hasOrganizationAnswered && canManageOrganization ? (
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
