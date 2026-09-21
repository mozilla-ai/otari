import { Button, Card, Chip } from "@heroui/react"
import { useState } from "react"

import type { OrganizationSpendCeiling } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  useDeleteOrganizationSpendCeiling,
  useOrganizationBudgets,
  useOrganizationSpendCeilings,
} from "@/shared/api/budgets"
import { useWorkspaces } from "@/shared/api/workspaces"
import { formatDate, formatUsd } from "@/shared/helpers/format"

import { limitLabel, periodLabel, scopeLabel } from "./organizationBudget"
import { SpendCeilingDialog } from "./SpendCeilingDialog"

// Where the organization's budgets actually apply, and what has been spent
// against each.
//
// Two kinds of row can appear that this card does not create. A member's ceiling
// is set on Members & roles, beside the person it caps; and a ceiling naming a
// budget set at the deployment level is what the otari-ai cutover writes. Both
// are listed, because both are enforcing against this organization today and
// leaving them out would let the page read as uncapped. The second is marked, so
// a figure nobody here can change does not look editable.

const DEFAULT_PAGE_SIZE = 25

function spentLabel(ceiling: OrganizationSpendCeiling): string {
  const spent = formatUsd(ceiling.current_spend)
  if (ceiling.reserved_spend === 0) return spent
  // Reserved is held against in-flight requests and settles into spend or is
  // released, so it counts towards the cap and is not spend yet. Both shown,
  // because a ceiling refuses on their sum.
  return `${spent} (+${formatUsd(ceiling.reserved_spend)} held)`
}

export function SpendCeilingsCard({
  organizationId,
  organizationName,
}: {
  organizationId: string
  organizationName: string
}) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const ceilings = useOrganizationSpendCeilings(page, pageSize)
  const budgets = useOrganizationBudgets()
  const workspaces = useWorkspaces()
  const remove = useDeleteOrganizationSpendCeiling()

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in rather than on the way out.
  const [openCount, setOpenCount] = useState(0)
  // The organization joins that key because the dialog seeds its target from
  // the id on mount: a remount is what re-seeds it.
  const dialogKey = `${organizationId}:${openCount}`
  const [editing, setEditing] = useState<OrganizationSpendCeiling>()
  const [pendingDelete, setPendingDelete] = useState<OrganizationSpendCeiling>()

  // Switching organization invalidates every query rather than remounting this
  // page, so both frames can outlive the rows they were opened on: the server
  // scopes each write to the caller's organization and refuses, but the frame
  // still names a ceiling the reader has just left. Dropped here rather than in
  // an effect, so nothing renders against the new organization holding the old
  // one's row.
  const [shownFor, setShownFor] = useState(organizationId)
  if (shownFor !== organizationId) {
    setShownFor(organizationId)
    setDialogOpen(false)
    setEditing(undefined)
    setPendingDelete(undefined)
    // The page too, or the new organization is asked for a window its shorter
    // list does not reach: that answers empty, the step-back below decrements,
    // and the two walk down a page per request until they meet zero.
    setPage(0)
  }

  const rows = ceilings.data?.data ?? []

  // Deleting the last ceiling on a page leaves it empty while earlier pages
  // still hold rows. Stepping back during render rather than in an effect, the
  // way `TablePagination` adjusts its own page box.
  if (page > 0 && ceilings.data && !ceilings.isFetching && rows.length === 0) {
    setPage(page - 1)
  }
  const workspaceRows = workspaces.data ?? []

  const openAdd = () => {
    setOpenCount((count) => count + 1)
    setEditing(undefined)
    setDialogOpen(true)
  }

  const openEdit = (ceiling: OrganizationSpendCeiling) => {
    setOpenCount((count) => count + 1)
    setEditing(ceiling)
    setDialogOpen(true)
  }

  const columns: DataTableColumn<OrganizationSpendCeiling>[] = [
    {
      id: "scope",
      header: "Capping",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex flex-col gap-0.5">
          <span className="text-body">
            {scopeLabel(row, { organizationName, workspaces: workspaceRows })}
          </span>
          {row.name ? <span className="text-caption">{row.name}</span> : null}
        </div>
      ),
    },
    {
      id: "provider",
      header: "Provider",
      cell: (row) => row.provider_key_id ?? "Every provider",
    },
    {
      id: "limit",
      header: "Limit",
      align: "end",
      cell: (row) => (
        <div className="flex flex-col items-end gap-0.5">
          <span>{limitLabel(row)}</span>
          {row.manageable ? null : (
            // Not a warning: the ceiling is correct and enforcing, it simply is
            // not this organization's figure to change. The word "deployment"
            // rather than any role, which is not a tenant's to see.
            <Chip size="sm" variant="secondary">
              Set at the deployment level
            </Chip>
          )}
        </div>
      ),
    },
    {
      id: "spent",
      header: "Spent this period",
      align: "end",
      cell: (row) => spentLabel(row),
    },
    {
      id: "resets",
      header: "Resets",
      cell: (row) => (
        <div className="flex flex-col gap-0.5">
          <span>{periodLabel(row)}</span>
          {row.period_end ? (
            <span className="text-caption">
              Next on {formatDate(row.period_end)}
            </span>
          ) : null}
        </div>
      ),
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
        <h2 className="text-title">Spend ceilings</h2>
        <Button size="sm" variant="primary" onPress={openAdd}>
          Add ceiling
        </Button>
      </div>

      <p className="text-sm text-muted">
        What each budget actually caps. A request has to pass every ceiling that
        applies to it, so an organization-wide cap and a workspace cap both
        bind. A member&rsquo;s own ceiling is set on Members &amp; roles, beside
        the person it caps.
      </p>

      {/* All three reads, not just the ceilings. A failed workspace roster
          leaves the dialog with no workspace to offer and the table naming
          "A workspace" for rows it cannot resolve, and a failed budget list
          leaves it with nothing to hold a ceiling to; without this the owner
          sees the consequence and never the cause. First error wins, because
          one banner saying something true beats three stacked. */}
      <ErrorBanner
        error={ceilings.error ?? workspaces.error ?? budgets.error}
      />

      <Card>
        <Card.Content className="p-0">
          <DataTable
            ariaLabel="Organization spend ceilings"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={ceilings.isPending && !ceilings.data}
            emptyContent="Nothing is capped yet. Add a ceiling to hold an organization or a workspace to a budget."
          />
          <div className="px-4 pb-3">
            <TablePagination
              page={page}
              pageSize={pageSize}
              total={ceilings.data?.count ?? null}
              rowsOnPage={rows.length}
              onPageChange={setPage}
              onPageSizeChange={(size) => {
                setPageSize(size)
                setPage(0)
              }}
              isFetching={ceilings.isFetching}
              label="spend ceilings"
            />
          </div>
        </Card.Content>
      </Card>

      <SpendCeilingDialog
        key={dialogKey}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        budgets={budgets.data ?? []}
        workspaces={workspaceRows}
        organizationId={organizationId}
        organizationName={organizationName}
        onSaved={() => setDialogOpen(false)}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete spend ceiling"
        body={
          pendingDelete
            ? `${scopeLabel(pendingDelete, { organizationName, workspaces: workspaceRows })} stops being capped by this budget from the next request. The budget itself is left alone, and so is every other ceiling holding it.`
            : null
        }
        confirmLabel="Delete ceiling"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete.id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />
    </section>
  )
}
