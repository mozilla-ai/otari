import type { ReactNode } from "react"

import type { CallerWorkspaceMembership } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { InferenceAccessSection } from "@/features/workspaces/access/InferenceAccessSection"

/** What every section is handed: the workspace, and whether the caller may change it. */
export interface AccessSectionContext {
  workspace: CallerWorkspaceMembership
  canManage: boolean
  /** Whether the caller manages the organization, which some reads behind a section need. */
  canManageOrganization: boolean
}

/**
 * One capability a workspace can be given or kept from, shown as a tab of its
 * own on the workspace's page.
 *
 * A new capability is one entry here and the drawer its rows open. Every
 * section answers in the same terms: a workspace inherits everything until an
 * admin narrows it, it can switch a source off or narrow it, and it never
 * grants what the organization or the deployment above it withholds.
 */
interface AccessSection {
  id: string
  /** The tab's label, which names the section, so the section draws no heading. */
  title: string
  description: string
  render: (context: AccessSectionContext) => ReactNode
}

const ACCESS_SECTIONS: readonly AccessSection[] = [
  {
    id: "providers",
    title: "Providers",
    description: "Which providers and models this workspace can call.",
    render: ({ workspace, canManage, canManageOrganization }) => (
      <InferenceAccessSection
        workspaceId={workspace.workspace_id}
        canManage={canManage}
        canManageOrganization={canManageOrganization}
      />
    ),
  },
]

/** The sections as the page's tabs list them. */
export const ACCESS_TABS = ACCESS_SECTIONS.map(({ id, title }) => ({
  id,
  label: title,
}))

/** One capability's section, by the id its tab carries. */
export function WorkspaceAccessSection({
  sectionId,
  ...context
}: AccessSectionContext & { sectionId: string }) {
  const section = ACCESS_SECTIONS.find((entry) => entry.id === sectionId)
  if (section === undefined) return null
  return (
    <section aria-label={section.title} className="flex flex-col gap-4">
      {context.canManage ? null : (
        <InfoBanner>
          Organization owners and admins, and this workspace's own owners and
          admins, can change what it may use.
        </InfoBanner>
      )}
      <p className="max-w-2xl text-caption">{section.description}</p>
      {section.render(context)}
    </section>
  )
}
