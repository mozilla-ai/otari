import { Link, Navigate, useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import type {
  OrganizationBudget,
  OrganizationContext,
  OrganizationSpendCeiling,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { isDeploymentOperator } from "@/features/organization/roles"
import {
  useOrganizationBudgets,
  useOrganizationSpendCeilingsAll,
} from "@/shared/api/budgets"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  formatDate,
  formatNumber,
  formatTokens,
  formatUsd,
} from "@/shared/helpers/format"

import { ceilingKey, entityNamer } from "./appliedEntities"
import { appliedToLabel } from "./appliedTo"
import { budgetLabeler } from "./budgetLabel"
import { DeleteBudgetDialog } from "./DeleteBudgetDialog"
import { OrganizationBudgetDialog } from "./OrganizationBudgetDialog"
import { limitLabel } from "./organizationBudget"
import { cycleLabel } from "./resetCycle"
import { useEntitySources } from "./useEntitySources"

// One budget, and what each entity it applies to has spent this period.
//
// The spend is per entity rather than summed, because each entity draws on its
// own allowance of the limit: a total would read as one pool the budget does not
// have. Tokens and requests get a column only when the budget caps them.

/** The route's page: an organization budget for an admin, the list for an operator. */
export function BudgetDetailPage({ budgetId }: { budgetId: string }) {
  const organization = useOrganizationContext()
  if (organization.isPending && !organization.data) {
    return <PageLoading label="Loading the budget…" />
  }
  // An operator's budgets are the deployment's, on a page with no detail route.
  if (!organization.data || isDeploymentOperator(organization.data)) {
    return <Navigate to="/budgets" />
  }
  return (
    <OrganizationBudgetDetail
      organization={organization.data}
      budgetId={budgetId}
    />
  )
}

function spentLabel(ceiling: OrganizationSpendCeiling): string {
  const spent = formatUsd(ceiling.current_spend)
  const held =
    ceiling.reserved_spend > 0
      ? ` (+${formatUsd(ceiling.reserved_spend)} held)`
      : ""
  return ceiling.max_budget === null
    ? `${spent}${held}`
    : `${spent}${held} of ${formatUsd(ceiling.max_budget)}`
}

export function OrganizationBudgetDetail({
  organization,
  budgetId,
}: {
  organization: OrganizationContext
  budgetId: string
}) {
  const navigate = useNavigate()
  const budgets = useOrganizationBudgets()
  const ceilings = useOrganizationSpendCeilingsAll()
  const organizationId = organization.organization.id
  const organizationName = organization.organization.name
  const { sources } = useEntitySources(organizationId)

  const [isEditing, setEditing] = useState(false)
  const [openCount, setOpenCount] = useState(0)
  const [pendingDelete, setPendingDelete] = useState<OrganizationBudget>()

  const budget = budgets.data?.find((row) => row.budget_id === budgetId)
  const held = (ceilings.data ?? []).filter(
    (ceiling) => ceiling.budget_id === budgetId,
  )

  const back = (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link to="/budgets" className="text-link hover:text-link-hover">
        ← All budgets
      </Link>
    </nav>
  )

  if (budgets.isPending && !budgets.data) {
    return <PageLoading label="Loading the budget…" />
  }
  if (!budget) {
    return (
      <div className="flex flex-col gap-6">
        {back}
        <ErrorBanner error={budgets.error} />
        {budgets.error ? null : (
          <EmptyState
            title="Budget not found"
            description="It may have been deleted, or it belongs to another organization."
            actionLabel="All budgets"
            onAction={() => void navigate({ to: "/budgets" })}
          />
        )}
      </div>
    )
  }

  const name = budgetLabeler(budgets.data ?? [])(budget)
  const nameEntity = entityNamer(sources, organizationName)
  // The window is the budget's, so every entity shares one next reset. One
  // already passed is a period still waiting on a request to roll it, and the
  // spend reads zero for it, so it is not a reset still to come.
  const nextReset = held
    .map((ceiling) => ceiling.period_end)
    .find((end) => end && new Date(end).getTime() > Date.now())

  const columns: DataTableColumn<OrganizationSpendCeiling>[] = [
    {
      id: "entity",
      header: "Entity",
      isRowHeader: true,
      cell: (row) => {
        const entity = nameEntity(ceilingKey(row))
        return (
          <div className="flex flex-col gap-0.5">
            <span className="text-body">{entity.name}</span>
            <span className="text-caption">{entity.kind}</span>
          </div>
        )
      },
    },
    {
      id: "spent",
      header: "Spent this period",
      cell: (row) => (
        <div className="flex min-w-48 flex-col gap-1">
          <span>{spentLabel(row)}</span>
          {row.max_budget === null ? null : (
            <SpendMeter
              spent={row.current_spend + row.reserved_spend}
              allocated={row.max_budget}
              ariaLabel={`${nameEntity(ceilingKey(row)).name}: ${spentLabel(row)}`}
            />
          )}
        </div>
      ),
    },
  ]
  if (budget.token_limit !== null) {
    columns.push({
      id: "tokens",
      header: "Tokens",
      align: "end",
      cell: (row) =>
        `${formatTokens(row.current_tokens)} of ${formatTokens(budget.token_limit ?? 0)}`,
    })
  }
  if (budget.request_limit !== null) {
    columns.push({
      id: "requests",
      header: "Requests",
      align: "end",
      cell: (row) =>
        `${formatNumber(row.current_requests)} of ${formatNumber(budget.request_limit)}`,
    })
  }

  const openEdit = () => {
    setOpenCount((count) => count + 1)
    setEditing(true)
  }

  return (
    <div className="flex flex-col gap-6">
      {back}
      <PageIntro
        title={name}
        action={
          <div className="flex gap-2">
            <Button variant="ghost" onPress={() => setPendingDelete(budget)}>
              Delete
            </Button>
            <Button variant="primary" onPress={openEdit}>
              Edit budget
            </Button>
          </div>
        }
      >
        {limitLabel(budget)}. {cycleLabel(budget)}
        {nextReset ? `. Next reset ${formatDate(nextReset)}.` : "."}
      </PageIntro>

      <ErrorBanner error={ceilings.error} />

      {ceilings.data !== undefined && held.length === 0 ? (
        <EmptyState
          title="Not applied yet"
          description="This budget caps nothing until it applies to an entity."
          actionLabel="Edit budget"
          onAction={openEdit}
        />
      ) : (
        <TableScrollFrame className="otari-budget-detail-table">
          <DataTable
            ariaLabel="Spend by entity"
            columns={columns}
            rows={held}
            getRowKey={(row) => row.id}
            isLoading={ceilings.isPending && !ceilings.data}
            emptyContent="Not applied yet."
          />
        </TableScrollFrame>
      )}

      <OrganizationBudgetDialog
        key={`${organizationId}:${openCount}`}
        isOpen={isEditing}
        onOpenChange={setEditing}
        editing={budget}
        organizationId={organizationId}
        organizationName={organizationName}
        onSaved={() => setEditing(false)}
      />

      <DeleteBudgetDialog
        key={pendingDelete?.budget_id}
        budget={pendingDelete}
        budgetName={name}
        appliedTo={appliedToLabel(held, {
          organizationName,
          workspaces: sources.workspaces,
        })}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        onDeleted={() => void navigate({ to: "/budgets" })}
      />
    </div>
  )
}
