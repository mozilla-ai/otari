import { Link } from "@tanstack/react-router"
import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { OrganizationBudget } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { HeadroomRing } from "@/design-system/metrics/HeadroomRing"

import { formatAppliedTo } from "./appliedTo"
import { hasNoLimit, limitLabel, tightestUsage } from "./organizationBudget"
import { cycleLabel } from "./resetCycle"

// One row per budget. Usage is the tightest entity, not a sum: each entity draws
// on its own allowance of the limit, so a summed cell would read as a single pool
// and hide the one entity already being refused behind the others' headroom. The
// cell names that entity when there is more than one; every entity's own bar is
// on the detail page.

export interface OrganizationBudgetsTableProps {
  budgets: OrganizationBudget[]
  isLoading: boolean
  nameBudget: (budget: OrganizationBudget) => string
  onEdit: (budget: OrganizationBudget) => void
  onDelete: (budget: OrganizationBudget) => void
}

export function OrganizationBudgetsTable({
  budgets,
  isLoading,
  nameBudget,
  onEdit,
  onDelete,
}: OrganizationBudgetsTableProps) {
  const columns: DataTableColumn<OrganizationBudget>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        <Link
          to="/budgets/$budgetId"
          params={{ budgetId: row.budget_id }}
          className="text-link hover:text-link-hover"
        >
          {nameBudget(row)}
        </Link>
      ),
    },
    {
      id: "limit",
      header: "Limit",
      align: "end",
      cell: (row) => limitLabel(row),
    },
    {
      id: "applied",
      header: "Applied to",
      cell: (row) => formatAppliedTo(row.applied_to),
    },
    {
      id: "usage",
      header: "Usage",
      cell: (row) => {
        if (hasNoLimit(row))
          return <span className="text-subtle">Uncapped</span>
        const tightest = tightestUsage(row, row.applied_to)
        if (!tightest) return <span className="text-subtle">Not applied</span>
        return (
          <div className="flex flex-col gap-0.5">
            <HeadroomRing used={tightest.used} />
            {/* Named only when it stands out: at nothing used, every entity ties. */}
            {row.applied_to.length > 1 && tightest.used > 0 ? (
              <span className="text-caption">
                {formatAppliedTo([tightest.ceiling])}
              </span>
            ) : null}
          </div>
        )
      },
    },
    {
      id: "resets",
      header: "Reset cycle",
      cell: (row) => cycleLabel(row),
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiEdit2}
            label="Edit"
            ariaLabel={`Edit ${nameBudget(row)}`}
            onPress={() => onEdit(row)}
          />
          {/* Carries no danger color: this opens the confirmation rather than
              deleting, and the dialog's own confirm is what wears it. */}
          <RowAction
            icon={FiTrash2}
            label="Delete"
            ariaLabel={`Delete ${nameBudget(row)}`}
            onPress={() => onDelete(row)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <TableScrollFrame className="otari-org-budgets-table">
      <DataTable
        ariaLabel="Budgets"
        columns={columns}
        rows={budgets}
        getRowKey={(row) => row.budget_id}
        isLoading={isLoading}
        emptyContent="No budgets to show."
      />
    </TableScrollFrame>
  )
}
