import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { OrganizationBudget } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"

import { formatAppliedTo } from "./appliedTo"
import { limitLabel } from "./organizationBudget"
import { cycleLabel } from "./resetCycle"

// One row per budget. What each budget has spent is per applied entity, because
// each entity draws on its own allowance of the limit, so there is no spend
// column: summed into one cell it would read as a single pool.

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
      cell: (row) => <span className="text-body">{nameBudget(row)}</span>,
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
