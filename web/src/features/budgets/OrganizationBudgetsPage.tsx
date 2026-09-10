import type { OrganizationContext } from "@/client"
import { PageHeader } from "@/shared/components/deprecated/PageHeader"

import { OrganizationBudgetsCard } from "./OrganizationBudgetsCard"
import { SpendCeilingsCard } from "./SpendCeilingsCard"

// Spend & budgets, for an organization owner or admin.
//
// The tenant-scoped half of `BudgetsPage`, which keeps the deployment-wide page
// an operator sees. The roles matrix (otari-ai#1943) puts this row at Edit for an
// admin and Hidden for a member, and before this it was operator-only end to
// end: `/budgets` gates every handler on `require_deployment_operator` and
// `/scoped-budgets` gates its whole router, so a hosted organization could
// manage neither its own caps nor where they applied.
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
      <PageHeader
        title="Spend & budgets"
        description="What this organization may spend, and where each limit applies. A request is refused when any ceiling covering it is out of headroom."
      />
      <OrganizationBudgetsCard />
      <SpendCeilingsCard organizationName={organization.organization.name} />
    </div>
  )
}
