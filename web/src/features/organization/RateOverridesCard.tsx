import { Button } from "@heroui/react"
import { useState } from "react"
import { FiEdit2, FiTrash2 } from "react-icons/fi"

import type { OrganizationPricingOverride } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Dot } from "@/design-system/indicators/Dot"
import { Section } from "@/design-system/layout/Section"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useDeleteOrganizationPricing,
  useOrganizationPricing,
} from "@/shared/api/pricing"
import { formatCost, formatDateTime } from "@/shared/helpers/format"
import { PricingOverrideDialog } from "./PricingOverrideDialog"
import { overrideStatus } from "./pricingOverride"
import { canManage } from "./roles"

// What this organization pays for a model, above the catalog the rest of this
// page shows.
//
// A section of Model pricing rather than a destination of its own, because an
// operator asking "what does this model cost us" is asking one question. Two
// cards on that page and not one merged table, though: a request resolves the
// override first and the catalog row second (`services/pricing_service.py`), so
// a single list would hide which of the two a bill came from.
//
// A period is half-open, so two of them may meet at an instant without
// overlapping, and overlapping periods for one model are refused rather than
// shadowed. That is the server's rule; the dialog disables the save before a
// doomed request and the 409 is still the authority.

const STATUS_LABEL: Record<
  ReturnType<typeof overrideStatus>,
  { label: string; className: string }
> = {
  active: { label: "Active", className: "text-muted" },
  scheduled: { label: "Scheduled", className: "text-muted" },
  expired: { label: "Expired", className: "text-subtle" },
}

/**
 * The dot carries the state; the words stay quiet.
 *
 * Ink alone would spend the danger and success channels on three states none of
 * which is a problem: an override that has expired did what it was for. So the
 * words sit on the neutral rungs and the mark says which of the three this is,
 * the same division the rest of the surface uses. Expired is the one that
 * recedes, since it no longer bills anything.
 */
const STATUS_DOT: Record<ReturnType<typeof overrideStatus>, string> = {
  active: "bg-success",
  scheduled: "bg-accent",
  expired: "bg-text-subtle",
}

function rate(value: number | null | undefined): string {
  // `formatCost` is the page's one money formatter, shared with the catalog
  // table above so the same quantity cannot render two ways on one page. The
  // absent check stays in front of it rather than being folded into it: a blank
  // optional rate means the tokens are priced as fresh input, and `formatCost`
  // renders null as "$0.00", which would claim the organization negotiated a
  // free cache read. The em dash is the glyph the catalog column already uses
  // for the same "no rate stored" state.
  if (value === null || value === undefined) return "—"
  return formatCost(value)
}

function period(override: OrganizationPricingOverride): string {
  const from = formatDateTime(override.effective_from)
  if (!override.effective_to) return `From ${from}`
  return `${from} to ${formatDateTime(override.effective_to)}`
}

export function RateOverridesCard() {
  const context = useOrganizationContext()
  const overrides = useOrganizationPricing()
  const remove = useDeleteOrganizationPricing()

  const [isDialogOpen, setDialogOpen] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in rather than on the way out. These values set money.
  const [openCount, setOpenCount] = useState(0)
  const [editing, setEditing] = useState<OrganizationPricingOverride>()
  const [pendingDelete, setPendingDelete] =
    useState<OrganizationPricingOverride>()

  const canEdit = canManage(context.data)
  const rows = overrides.data ?? []

  const openAdd = () => {
    setOpenCount((count) => count + 1)
    setEditing(undefined)
    setDialogOpen(true)
  }

  const openEdit = (override: OrganizationPricingOverride) => {
    setOpenCount((count) => count + 1)
    setEditing(override)
    setDialogOpen(true)
  }

  const columns: DataTableColumn<OrganizationPricingOverride>[] = [
    {
      id: "model_key",
      header: "Model",
      isRowHeader: true,
      cell: (row) => <code className="text-xs">{row.model_key}</code>,
    },
    {
      id: "input",
      header: "Input / 1M",
      align: "end",
      cell: (row) => rate(row.input_price_per_million),
    },
    {
      id: "output",
      header: "Output / 1M",
      align: "end",
      cell: (row) => rate(row.output_price_per_million),
    },
    {
      id: "cache_read",
      header: "Cache read / 1M",
      align: "end",
      cell: (row) => rate(row.cache_read_price_per_million),
    },
    {
      id: "cache_write",
      header: "Cache write / 1M",
      align: "end",
      cell: (row) => rate(row.cache_write_price_per_million),
    },
    {
      id: "period",
      header: "Period",
      cell: (row) => <span className="text-caption">{period(row)}</span>,
    },
    {
      id: "status",
      header: "Status",
      cell: (row) => {
        const status = STATUS_LABEL[overrideStatus(row)]
        // The dot-and-word the rest of the surface uses, not a chip. The dot
        // carries the same severity the words already did, so the state is
        // legible without reading the label and without a box around it.
        return (
          <span
            className={`flex items-center gap-2 text-mono-caption ${status.className}`}
          >
            <Dot className={STATUS_DOT[overrideStatus(row)]} />
            {status.label.toUpperCase()}
          </span>
        )
      },
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        // Both controls stay mounted and disabled for a reader rather than
        // vanishing, so the page does not reflow between roles.
        <RowActionRow>
          <RowAction
            icon={FiEdit2}
            label="Edit"
            isDisabled={!canEdit}
            onPress={() => openEdit(row)}
          />
          <RowAction
            icon={FiTrash2}
            label="Delete"
            isDisabled={!canEdit}
            onPress={() => setPendingDelete(row)}
          />
        </RowActionRow>
      ),
    },
  ]

  return (
    <>
      {/* The group is introduced by its heading and closed by the rows' own
          separators. The rows stay a table rather than becoming a list of
          settings rows: four of the lanes are prices for the same model, and a
          price that cannot be read down the column against its neighbors is a
          number with nothing to compare it to. */}
      <Section
        aria-labelledby="rate-overrides-title"
        className="border-t border-border pt-6 pb-3"
        contentClassName="flex flex-col gap-2"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="rate-overrides-title" className="text-title">
            Rate overrides
          </h2>
          <Button
            size="sm"
            variant="primary"
            isDisabled={!canEdit}
            onPress={openAdd}
          >
            Add override
          </Button>
        </div>

        <p className="max-w-prose text-sm text-muted">
          This organization&rsquo;s own rate for a model, applied ahead of the
          catalog above. A model with no override here is priced by that
          catalog.
        </p>
      </Section>

      {canEdit ? null : (
        <InfoBanner>
          You can see the rates your requests are billed at. Only owners and
          admins can change them.
        </InfoBanner>
      )}

      <ErrorBanner error={overrides.error} />

      <TableScrollFrame className="otari-rate-overrides-table">
        <DataTable
          ariaLabel="Organization rate overrides"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.id}
          isLoading={overrides.isPending && !overrides.data}
          // Deliberately asserts nothing about the catalog: an empty table is
          // also what a failed request leaves behind, and the banner above is
          // the only thing that knows which of the two happened.
          emptyContent="No override yet. Add one to bill this organization at its own rate for a model."
        />
      </TableScrollFrame>

      {/* Keyed on the open count, so each open remounts a blank form. */}
      <PricingOverrideDialog
        key={openCount}
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        existing={rows}
        onSaved={() => setDialogOpen(false)}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete rate override"
        body={
          pendingDelete
            ? `${pendingDelete.model_key} returns to the catalog rate from the next request. Usage already billed at this rate keeps the cost it was charged.`
            : null
        }
        confirmLabel="Delete override"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete.id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />
    </>
  )
}
