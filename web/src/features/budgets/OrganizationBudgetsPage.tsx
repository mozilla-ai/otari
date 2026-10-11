import { useRef, useState } from "react"

import type { OrganizationBudget, OrganizationContext } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import {
  useDeleteOrganizationBudget,
  useOrganizationBudgets,
} from "@/shared/api/budgets"

import { budgetLabeler } from "./budgetLabel"
import { OrganizationBudgetDialog } from "./OrganizationBudgetDialog"
import { OrganizationBudgetsTable } from "./OrganizationBudgetsTable"
import { SpendCeilingsCard } from "./SpendCeilingsCard"

// Budgets, for an organization owner or admin: one table of the organization's
// budgets, each naming what it applies to.

export function OrganizationBudgetsPage({
  organization,
}: {
  organization: OrganizationContext
}) {
  const budgets = useOrganizationBudgets()
  const remove = useDeleteOrganizationBudget()
  // Where focus lands after the first create, when the empty state that opened
  // the dialog is gone.
  const createButtonRef = useRef<HTMLButtonElement>(null)

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in. Clearing it on close would blank the fields while the dialog
  // is still animating away.
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<OrganizationBudget>()
  const [pendingDelete, setPendingDelete] = useState<OrganizationBudget>()

  const rows = budgets.data ?? []

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

  const nameBudget = budgetLabeler(rows)

  // The read has to have landed before an empty table means an empty
  // organization: a failed read leaves the same no rows behind, and the banner
  // is the only thing that knows which happened.
  const hasLanded = budgets.data !== undefined
  const isEmpty = hasLanded && rows.length === 0

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Budgets"
        action={
          <Button ref={createButtonRef} variant="primary" onPress={openAdd}>
            Create budget
          </Button>
        }
      >
        A budget is a limit, a reset cycle, and the entities it applies to. Each
        entity draws on its own allowance of the limit, and a request is refused
        when any budget covering it is out of headroom.
      </PageIntro>

      <ErrorBanner error={budgets.error} />

      {isEmpty ? (
        <EmptyState
          title="No budgets yet"
          description="A budget is a spending limit and how often it resets. Create one, then apply it to the organization, a workspace, a member or an API key to cap what they may spend."
          actionLabel="Create budget"
          onAction={openAdd}
        />
      ) : (
        <OrganizationBudgetsTable
          budgets={rows}
          isLoading={budgets.isPending && !budgets.data}
          nameBudget={nameBudget}
          onEdit={openEdit}
          onDelete={setPendingDelete}
        />
      )}

      {/* Where a budget is applied is still edited here, until the create and
          edit journeys take the entity picker. Delete this with them. */}
      <SpendCeilingsCard
        organizationId={organization.organization.id}
        organizationName={organization.organization.name}
      />

      <OrganizationBudgetDialog
        key={openCount}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        returnFocusRef={createButtonRef}
        onSaved={() => setDialogOpen(false)}
      />

      {pendingDelete ? (
        <ConfirmDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) setPendingDelete(undefined)
          }}
          heading="Delete budget"
          body={
            pendingDelete.ceiling_count > 0
              ? `${nameBudget(pendingDelete)} is applied to ${pendingDelete.ceiling_count} ${pendingDelete.ceiling_count === 1 ? "entity" : "entities"}, so this will be refused. Remove them first.`
              : `${nameBudget(pendingDelete)} stops existing. It applies to nothing, so no cap changes.`
          }
          confirmLabel="Delete budget"
          isPending={remove.isPending}
          error={remove.error}
          onConfirm={() => {
            remove.mutate(pendingDelete.budget_id, {
              onSuccess: () => setPendingDelete(undefined),
            })
          }}
        />
      ) : null}
    </div>
  )
}
