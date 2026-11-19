import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { modelsUsed } from "@/features/people/modelsUsed"
import type { SpendWindow } from "@/features/spend/period"
import type { ModelGroup } from "@/features/usage/modelGroups"
import { useUsageSummary } from "@/shared/api/usage"
import { Band } from "@/shared/components/Band"
import { formatCost, formatTokens, formatUsd } from "@/shared/helpers/format"

const COLUMNS: DataTableColumn<ModelGroup>[] = [
  {
    id: "model",
    header: "Model",
    isRowHeader: true,
    cell: (row) => (
      <span
        className="block truncate text-mono-caption text-foreground"
        title={row.spellings.join("\n") || row.name}
      >
        {row.name}
      </span>
    ),
  },
  {
    id: "tokens",
    header: "Tokens",
    align: "end",
    cell: (row) => (
      <span className="text-mono-caption text-foreground">
        {formatTokens(row.tokens)}
      </span>
    ),
  },
  {
    id: "spend",
    header: "Spent",
    align: "end",
    cell: (row) => (
      <span className="text-mono-caption text-foreground">
        {row.cost >= 0.01 ? formatUsd(row.cost) : formatCost(row.cost)}
      </span>
    ),
  },
]

/**
 * What a person's month went to, model by model: tokens and spend on each,
 * the costliest first.
 */
export function PersonModels({
  userId,
  window,
}: {
  userId: string
  window: SpendWindow
}) {
  const summary = useUsageSummary(
    { start_date: window.start, user_id: [userId] },
    "day",
    ["model"],
    true,
    "organization",
  )
  const rows = modelsUsed(summary.data?.by_model ?? [])
  return (
    <Band title="Models this month">
      <ErrorBanner error={summary.error} />
      <TableScrollFrame className="">
        <DataTable
          ariaLabel="Models this month"
          columns={COLUMNS}
          rows={rows}
          isLoading={summary.isLoading}
          getRowKey={(row) => row.id}
          emptyContent={
            summary.error ? (
              ""
            ) : (
              <EmptyMessage>No model calls this month.</EmptyMessage>
            )
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
    </Band>
  )
}
