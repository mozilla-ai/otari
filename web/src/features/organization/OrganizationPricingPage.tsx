import { Button, Chip } from "@heroui/react"
import { useState } from "react"

import type { OrganizationPricingOverride } from "@/client"
import {
  useCreateOrganizationPricing,
  useDeleteOrganizationPricing,
  useOrganizationContext,
  useOrganizationPricing,
  useReplaceOrganizationPricing,
} from "@/shared/api/hooks"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  EmptyState,
  ErrorBanner,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { formatDateTime } from "@/shared/helpers/format"
import {
  PricingOverrideDialog,
  type PricingOverrideDraft,
} from "./PricingOverrideDialog"
import { overrideStatus } from "./pricingOverride"
import { canManage } from "./roles"

// What this organization pays for a model, above the deployment price list the
// Models page edits.
//
// Two price lists, not one merged view. A request resolves the override first
// and the deployment row second, so an override is a row an operator manages in
// its own right rather than a variant of a deployment price, and showing them
// merged would hide which of the two a bill came from. The resolution order
// lives in `services/pricing_service.py`.
//
// A period is half-open, so two of them may meet at an instant without
// overlapping, and overlapping periods for one model are refused rather than
// shadowed. That is the server's rule; the dialog disables the save before a
// doomed request and the 409 is still the authority.

const STATUS_LABEL: Record<
  ReturnType<typeof overrideStatus>,
  { label: string; className: string }
> = {
  active: { label: "Active", className: "text-success" },
  scheduled: { label: "Scheduled", className: "text-info" },
  expired: { label: "Expired", className: "text-muted" },
}

function rate(value: number | null | undefined): string {
  // A blank optional rate is not zero: it means the tokens are priced as fresh
  // input, so a dash reads truer than "$0.00", which would claim the
  // organization negotiated a free cache read.
  if (value === null || value === undefined) return "–"
  return `$${value.toFixed(4)}`
}

function period(override: OrganizationPricingOverride): string {
  const from = formatDateTime(override.effective_from)
  if (!override.effective_to) return `From ${from}`
  return `${from} to ${formatDateTime(override.effective_to)}`
}

export function OrganizationPricingPage() {
  const context = useOrganizationContext()
  const overrides = useOrganizationPricing()
  const create = useCreateOrganizationPricing()
  const replace = useReplaceOrganizationPricing()
  const remove = useDeleteOrganizationPricing()

  const [isDialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState<OrganizationPricingOverride>()
  const [pendingDelete, setPendingDelete] =
    useState<OrganizationPricingOverride>()

  const canEdit = canManage(context.data)
  const rows = overrides.data ?? []

  const openAdd = () => {
    setEditing(undefined)
    setDialogOpen(true)
  }

  const openEdit = (override: OrganizationPricingOverride) => {
    setEditing(override)
    setDialogOpen(true)
  }

  const submit = (draft: PricingOverrideDraft) => {
    const onDone = { onSuccess: () => setDialogOpen(false) }
    if (editing) {
      // model_key is absent from the update body: the endpoint refuses to
      // repoint an override at another model.
      const { model_key: _unused, ...rest } = draft
      // The endpoint requires a start on a replacement, so that an omitted one
      // cannot silently move a stored period to the present. The dialog blocks a
      // blank start while editing; this narrows the type and is the belt to that
      // brace.
      if (rest.effective_from === null) return
      replace.mutate(
        {
          id: editing.id,
          body: { ...rest, effective_from: rest.effective_from },
        },
        onDone,
      )
      return
    }
    create.mutate(draft, onDone)
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
      cell: (row) => <span className="text-xs text-muted">{period(row)}</span>,
    },
    {
      id: "status",
      header: "Status",
      cell: (row) => {
        const status = STATUS_LABEL[overrideStatus(row)]
        return (
          <Chip size="sm" variant="secondary">
            <span className={status.className}>{status.label}</span>
          </Chip>
        )
      },
    },
    {
      id: "actions",
      header: "",
      cell: (row) => (
        // Both controls stay mounted and disabled for a reader rather than
        // vanishing, so the page does not reflow between roles.
        <div className="flex justify-end gap-2">
          <Button
            size="sm"
            variant="ghost"
            isDisabled={!canEdit}
            onPress={() => openEdit(row)}
          >
            Edit
          </Button>
          <Button
            size="sm"
            variant="ghost"
            isDisabled={!canEdit}
            onPress={() => setPendingDelete(row)}
          >
            Delete
          </Button>
        </div>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Rate overrides"
        description="What this organization pays for a model, above the deployment's own price list. A model with no override here is priced by that list, and then by the public pricing dataset."
        action={
          <Button variant="primary" isDisabled={!canEdit} onPress={openAdd}>
            Add override
          </Button>
        }
      />

      {canEdit ? null : (
        <InfoBanner>
          You can see the rates your requests are billed at. Only owners and
          admins can change them.
        </InfoBanner>
      )}

      <ErrorBanner error={overrides.error} />

      {/* Gated on the error too: a failed request also leaves `rows` empty, and
          the empty state's copy asserts that every model is priced by the
          deployment list, which the page cannot know when the list never
          arrived. */}
      {!overrides.isPending && !overrides.error && rows.length === 0 ? (
        <EmptyState
          title="No rate overrides"
          description="Every model is priced by the deployment price list. Add an override to bill this organization at its own negotiated rate for a model."
        />
      ) : (
        <DataTable
          ariaLabel="Organization rate overrides"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.id}
          isLoading={overrides.isPending && !overrides.data}
        />
      )}

      <PricingOverrideDialog
        isOpen={isDialogOpen}
        onOpenChange={setDialogOpen}
        editing={editing}
        existing={rows}
        isPending={create.isPending || replace.isPending}
        error={editing ? replace.error : create.error}
        onSubmit={submit}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(undefined)
        }}
        heading="Delete rate override"
        body={
          pendingDelete
            ? `${pendingDelete.model_key} returns to the deployment price list from the next request. Usage already billed at this rate keeps the cost it was charged.`
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
    </div>
  )
}
