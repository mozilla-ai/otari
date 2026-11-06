import { Link } from "@tanstack/react-router"

import type { CallerWorkspaceMembership } from "@/client"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import {
  canManage as canManageOrganization,
  canManageWorkspace,
} from "@/features/organization/roles"
import {
  ACCESS_TABS,
  WorkspaceAccessSection,
} from "@/features/workspaces/access/WorkspaceAccessSection"
import { WorkspaceGeneralForm } from "@/features/workspaces/WorkspaceGeneralForm"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useWorkspaces } from "@/shared/api/workspaces"
import { useUrlState } from "@/shared/helpers/urlState"

// Each capability is a tab of its own rather than a section of one long tab,
// so everything a workspace can be given is named in the tab row.
const TABS = [{ id: "general", label: "General" }, ...ACCESS_TABS]

const URL_DEFAULTS = { tab: "general" }

function asTab(value: string): string {
  return TABS.find((tab) => tab.id === value)?.id ?? "general"
}

/**
 * One workspace's own page: its settings, and what it may use.
 *
 * Opened from the Workspaces list. The tab is held in the URL, so a link can
 * land on one capability directly. Which workspace is a prop rather than read
 * from the router, so a test can hand it one without a route tree.
 */
export function WorkspaceDetailPage({ workspaceId }: { workspaceId: string }) {
  const workspaces = useWorkspaces()
  const context = useOrganizationContext()
  const url = useUrlState(URL_DEFAULTS)
  const tab = asTab(url.get("tab"))
  const workspace = (workspaces.data ?? []).find(
    (row) => row.id === workspaceId,
  )

  const back = (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link to="/workspaces" className="text-link hover:text-link-hover">
        ← Workspaces
      </Link>
    </nav>
  )

  if (workspace === undefined) {
    if (workspaces.isPending) return <PageLoading label="Loading workspace…" />
    return (
      <div className="flex flex-col gap-6">
        {back}
        <ErrorBanner error={workspaces.error} />
        {workspaces.error ? null : (
          <EmptyState
            title="Workspace not found"
            description="It may have been deleted, or it belongs to an organization you are not acting in."
          />
        )}
      </div>
    )
  }

  // The caller's own role here, which is what lets a workspace admin who does
  // not manage the organization change this workspace.
  const role = context.data?.workspace_memberships?.find(
    (membership) => membership.workspace_id === workspace.id,
  )?.role
  const canManage = canManageWorkspace(context.data, role)
  const membership: CallerWorkspaceMembership = {
    workspace_id: workspace.id,
    name: workspace.name,
    role: role ?? "",
  }

  return (
    <div className="flex flex-col gap-6">
      {back}
      <PageIntro title={workspace.name}>
        {workspace.description ||
          "Settings, and the providers and models this workspace can use."}
      </PageIntro>
      <TabRow>
        {TABS.map((entry) => (
          <Tab
            key={entry.id}
            isActive={tab === entry.id}
            onPress={() => url.patch({ tab: entry.id })}
          >
            {entry.label}
          </Tab>
        ))}
      </TabRow>
      {tab === "general" ? (
        <WorkspaceGeneralForm
          key={workspace.id}
          workspace={workspace}
          canManage={canManage}
        />
      ) : (
        <WorkspaceAccessSection
          sectionId={tab}
          workspace={membership}
          canManage={canManage}
          canManageOrganization={canManageOrganization(context.data)}
        />
      )}
    </div>
  )
}
