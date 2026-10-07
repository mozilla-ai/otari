import { Link } from "@tanstack/react-router"
import { useState } from "react"
import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { OrganizationBudget, OrganizationContext } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useOrganizationBudgets,
  useOrganizationSpendCeilingsAll,
} from "@/shared/api/budgets"
import { useWorkspaces } from "@/shared/api/workspaces"

import {
  APPLIED_TO_NOTHING,
  appliedToLabel,
  ceilingsByBudget,
} from "./appliedTo"
import { budgetLabeler } from "./budgetLabel"
import { DeleteBudgetDialog } from "./DeleteBudgetDialog"
import { OrganizationBudgetDialog } from "./OrganizationBudgetDialog"
import { limitLabel } from "./organizationBudget"
import { cycleLabel } from "./resetCycle"

// Budgets, for an organization owner or admin.
//
// One object, one table. A budget is a limit, a reset cycle, and the entities it
// applies to, so the page that used to carry a budgets card above a ceilings
// card now carries the budgets and names what each one applies to in a cell. The
// split was deliberate and is deliberately undone: it existed because one budget
// is shared by several ceilings and a merged *ceiling* row would have had to
// duplicate the figure or hide the sharing. A budget-centric row does neither.
// The figure is written once, where it is defined, and the sharing is the cell.
//
// Deliberately no spend column. A budget's figure is one number; what has been
// spent against it is per applied entity, because each entity draws on its own
// allowance of the limit. That belongs on the detail view, one bar per entity,
// not summed into a cell that would read as a single pool.

export function OrganizationBudgetsPage({
  organization,
}: {
  organization: OrganizationContext
}) {
  const budgets = useOrganizationBudgets()
  const ceilings = useOrganizationSpendCeilingsAll()
  const workspaces = useWorkspaces()

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in. Clearing it on close would blank the fields while the dialog
  // is still animating away.
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<OrganizationBudget>()
  const [pendingDelete, setPendingDelete] = useState<OrganizationBudget>()

  // Switching organization invalidates every query rather than remounting this
  // page, so an open form or confirmation would outlive the budget it was opened
  // on. Dropped during render rather than in an effect, so nothing renders
  // against the new organization holding the old one's row.
  const organizationId = organization.organization.id
  const [shownFor, setShownFor] = useState(organizationId)
  if (shownFor !== organizationId) {
    setShownFor(organizationId)
    setDialogOpen(false)
    setEditing(undefined)
    setPendingDelete(undefined)
  }

  const rows = budgets.data ?? []
  const applied = ceilingsByBudget(ceilings.data ?? [])
  const context = {
    organizationName: organization.organization.name,
    workspaces: workspaces.data ?? [],
  }

  const openAdd = () => {
    setOpenCount((count) => count + 1)
    setEditing(undefined)
    setDialogOpen(true)
  }

  const openEdit = (budget: OrganizationBudget) => {
    setOpenCount((count) => count + 1)
    setEditing(budget)
    setDialogOpen(true)
  }

  const appliedLabel = (row: OrganizationBudget) => {
    if (row.ceiling_count === 0) return APPLIED_TO_NOTHING
    const held = applied.get(row.budget_id) ?? []
    return held.length === 0
      ? `${row.ceiling_count} ${row.ceiling_count === 1 ? "entity" : "entities"}`
      : appliedToLabel(held, context)
  }

  const nameBudget = budgetLabeler(rows)
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
      // `ceiling_count` travels with the budget; the rows naming the entities
      // are a second read. Until that one lands the count is all there is, and
      // saying how many is the answer that cannot be wrong: deriving the cell
      // from the rows alone reads a budget applied to a dozen workspaces as
      // applied to nothing, for as long as the walk takes.
      cell: appliedLabel,
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
            onPress={() => openEdit(row)}
          />
          {/* Carries no danger color: this opens the confirmation rather than
              deleting, and the dialog's own confirm is what wears it. */}
          <RowAction
            icon={FiTrash2}
            label="Delete"
            ariaLabel={`Delete ${nameBudget(row)}`}
            onPress={() => setPendingDelete(row)}
          />
        </RowActionRow>
      ),
    },
  ]

  // The read has to have landed before an empty table means an empty
  // organization: a failed read leaves the same no rows behind, and the banner
  // is the only thing that knows which happened.
  const hasLanded = budgets.data !== undefined && !budgets.isPending
  const isEmpty = hasLanded && rows.length === 0

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Budgets"
        action={
          <Button variant="primary" onPress={openAdd}>
            New Budget
          </Button>
        }
      >
        A budget is a limit, a reset cycle, and the entities it applies to. Each
        entity draws on its own allowance of the limit, and a request is refused
        when any budget covering it is out of headroom.
      </PageIntro>

      <ErrorBanner error={budgets.error ?? ceilings.error} />

      {isEmpty ? (
        <EmptyState
          title="No budgets created"
          description="Create your first budget now."
          actionLabel="New Budget"
          onAction={openAdd}
        />
      ) : (
        <TableScrollFrame className="otari-org-budgets-table">
          <DataTable
            ariaLabel="Budgets"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.budget_id}
            isLoading={budgets.isPending && !budgets.data}
            emptyContent="No budgets created. Create your first budget now."
          />
        </TableScrollFrame>
      )}

      {/* Keyed on the organization too: the picker's entities are the
          organization's, and a remount is what re-seeds them. */}
      <OrganizationBudgetDialog
        key={`${organizationId}:${openCount}`}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        organizationId={organizationId}
        organizationName={organization.organization.name}
        onSaved={() => setDialogOpen(false)}
      />

      <DeleteBudgetDialog
        key={pendingDelete?.budget_id}
        budget={pendingDelete}
        budgetName={pendingDelete ? nameBudget(pendingDelete) : ""}
        appliedTo={pendingDelete ? appliedLabel(pendingDelete) : ""}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        onDeleted={() => setPendingDelete(undefined)}
      />
    </div>
  )
}
