import { Button, Card } from "@heroui/react"
import { useState } from "react"

import type { OrganizationBudget } from "@/client"
import {
  useCreateOrganizationBudget,
  useDeleteOrganizationBudget,
  useOrganizationBudgets,
  useUpdateOrganizationBudget,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { ErrorBanner } from "@/shared/components/ui"
import {
  OrganizationBudgetDialog,
  type OrganizationBudgetDraft,
} from "./OrganizationBudgetDialog"
import { budgetLabel, limitLabel, periodLabel } from "./organizationBudget"

// The organization's own budgets: the figures, without yet saying where they
// apply. The ceilings card below is what applies them.
//
// Deliberately no spend column. The deployment's budgets page shows one, summed
// over the gateway's `users` table, which is deployment-wide and carries no
// tenancy: the same figure here would be a cross-tenant read. What an
// organization has spent is Usage's question; what a *ceiling* has spent against
// its own counters is on the ceilings table, where it belongs.

export function OrganizationBudgetsCard() {
  const budgets = useOrganizationBudgets()
  const create = useCreateOrganizationBudget()
  const update = useUpdateOrganizationBudget()
  const remove = useDeleteOrganizationBudget()

  const [isDialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState<OrganizationBudget>()
  const [pendingDelete, setPendingDelete] = useState<OrganizationBudget>()

  const rows = budgets.data ?? []

  const openAdd = () => {
    setEditing(undefined)
    setDialogOpen(true)
  }

  const openEdit = (budget: OrganizationBudget) => {
    setEditing(budget)
    setDialogOpen(true)
  }

  const submit = (draft: OrganizationBudgetDraft) => {
    const onDone = { onSuccess: () => setDialogOpen(false) }
    if (editing) {
      update.mutate({ id: editing.budget_id, body: draft }, onDone)
      return
    }
    create.mutate(draft, onDone)
  }

  const columns: DataTableColumn<OrganizationBudget>[] = [
    {
      id: "name",
      header: "Budget",
      isRowHeader: true,
      cell: (row) => <span className="text-body">{budgetLabel(row)}</span>,
    },
    {
      id: "limit",
      header: "Limit",
      align: "end",
      cell: (row) => limitLabel(row),
    },
    {
      id: "resets",
      header: "Resets",
      cell: (row) => periodLabel(row),
    },
    {
      id: "ceilings",
      header: "Held by",
      align: "end",
      // The organization-relevant fact, and the one that makes a delete refuse:
      // a budget nothing names is deletable, and one that is named is not.
      cell: (row) =>
        row.ceiling_count === 0
          ? "Nothing yet"
          : `${row.ceiling_count} ${row.ceiling_count === 1 ? "ceiling" : "ceilings"}`,
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        <div className="flex justify-end gap-1">
          <Button size="sm" variant="ghost" onPress={() => openEdit(row)}>
            Edit
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onPress={() => setPendingDelete(row)}
          >
            Delete
          </Button>
        </div>
      ),
    },
  ]

  return (
    <section className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-title">Budgets</h2>
        <Button size="sm" variant="primary" onPress={openAdd}>
          Add budget
        </Button>
      </div>

      <p className="text-sm text-muted">
        An amount and the period it is spent over. A budget caps nothing until a
        spend ceiling below points it at something, and editing one moves every
        ceiling that holds it.
      </p>

      <ErrorBanner error={budgets.error} />

      <Card>
        <Card.Content className="p-0">
          <DataTable
            ariaLabel="Organization budgets"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.budget_id}
            isLoading={budgets.isPending && !budgets.data}
            // Says nothing about the organization: an empty table is also what a
            // failed read leaves behind, and the banner above is the only thing
            // that knows which happened.
            emptyContent="No budget yet. Add one, then point a spend ceiling at it."
          />
        </Card.Content>
      </Card>

      <OrganizationBudgetDialog
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        isPending={create.isPending || update.isPending}
        error={editing ? update.error : create.error}
        onSubmit={submit}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete budget"
        body={
          pendingDelete
            ? pendingDelete.ceiling_count > 0
              ? `${budgetLabel(pendingDelete)} is held by ${pendingDelete.ceiling_count} spend ${pendingDelete.ceiling_count === 1 ? "ceiling" : "ceilings"}, so this will be refused. Remove or repoint them first.`
              : `${budgetLabel(pendingDelete)} stops existing. Nothing holds it, so no cap changes.`
            : null
        }
        confirmLabel="Delete budget"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete.budget_id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />
    </section>
  )
}
