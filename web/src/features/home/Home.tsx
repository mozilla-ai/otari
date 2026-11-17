import { PageError } from "@/design-system/feedback/PageError"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { FirstRunStrip } from "@/features/home/FirstRunStrip"
import { MemberHome } from "@/features/home/MemberHome"
import { TeamHome } from "@/features/home/TeamHome"
import {
  useIsOrganizationManager,
  useIsRoleSettled,
} from "@/features/organization/access"
import {
  isNoOrganizationError,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * The page the product opens on, as one of two things depending on who opened
 * it:
 *
 * - an owner or admin: what needs them, what the window cost against the
 *   limit, and where it went;
 * - anyone else: the owner's Home for one person (`MemberHome.tsx`).
 *
 * A workspace that has never served a request gets the first-request guide
 * over it, mounted by the layout (`features/overview/OverviewPage.tsx`).
 */
export function Home() {
  const { selected, isSettled } = useSelectedWorkspace()
  const isManager = useIsOrganizationManager()
  const isRoleSettled = useIsRoleSettled()
  const context = useOrganizationContext()
  // The role never arrives from a failed read, so waiting on it would never
  // end. No organization at all is the shell's to answer, not this page's.
  if (
    context.data === undefined &&
    context.isError &&
    !isNoOrganizationError(context.error)
  ) {
    return <PageError error={context.error} />
  }
  // Which Home this is depends on the caller's role, so it waits for it.
  if (!isSettled || !isRoleSettled) return <PageLoading />
  // The team's Home spans the organization and picks its own workspace, so an
  // owner or admin has it whether or not they belong to a workspace.
  if (isManager) {
    return <TeamHome notice={<FirstRunStrip />} />
  }
  if (!selected) {
    return (
      <PageIntro
        title="Home"
        detail="You are not in a workspace yet. An owner or admin of the organization can add you to one."
      />
    )
  }
  return (
    <MemberHome
      workspaceId={selected.workspace_id}
      workspace={selected.name}
      notice={<FirstRunStrip />}
    />
  )
}
