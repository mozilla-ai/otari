import { Navigate } from "@tanstack/react-router"
import type { OrganizationContext } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import {
  isNoOrganizationError,
  useOrganizationContext,
} from "@/shared/api/organizations"

import { OrganizationBudgetsCard } from "./OrganizationBudgetsCard"
import { SpendCeilingsCard } from "./SpendCeilingsCard"

// Spend & budgets, for an organization owner or admin.
//
// The organization's own budgets, whoever opens them, an operator included
// (otari#2099); the deployment's are `DeploymentBudgetsPage`. The roles matrix
// (otari-ai#1943) puts this row at Edit for an admin and Hidden for a member.
// It reads only the `/organizations/me/` routes: `/budgets` and
// `/scoped-budgets` are `require_deployment_operator` throughout.
//
// Two sections, because the schema has two objects and they answer different
// questions: a budget is *what a cap is*, and a ceiling is *who is capped*. They
// stay separate rather than merging into one table of "caps", because one budget
// is deliberately shared by several ceilings, and a merged row would have to
// either duplicate the figure or hide the sharing. The second is what makes
// raising a limit one edit rather than one per place it applies.

export function OrganizationBudgetsPage({
  organization,
}: {
  organization: OrganizationContext
}) {
  return (
    <div className="flex flex-col gap-6">
      {/* No action here, unlike every other page's intro: this page holds two
          collections, and each one's Add sits on its own section heading rather
          than competing for one slot that could only name one of them. */}
      <PageIntro title="Spend & budgets" />
      <OrganizationBudgetsCard />
      <SpendCeilingsCard
        organizationId={organization.organization.id}
        organizationName={organization.organization.name}
      />
    </div>
  )
}

/**
 * The `/budgets` route: Spend & budgets for the organization the caller is in.
 *
 * Deployment authority is not asked. The route is the organization rail's, so
 * it opens to an owner or admin (`useRouteAccess`), and an operator wanting the
 * deployment's budgets has `/admin/budgets`.
 */
export function SpendAndBudgetsPage() {
  const organization = useOrganizationContext()

  // No organization means no budgets of its own; the index is where such a
  // caller is told how to get in.
  if (isNoOrganizationError(organization.error)) {
    return <Navigate to="/" replace />
  }
  if (organization.data) {
    return <OrganizationBudgetsPage organization={organization.data} />
  }
  if (organization.error) {
    return <ErrorBanner error={organization.error} />
  }
  return <PageLoading label="Loading spend and budgets…" />
}
