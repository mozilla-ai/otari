import type { SavingsByAgentType, SavingsBySwap } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  basisOf,
  formatSaved,
  pricedCount,
  spentOn,
} from "@/features/routerSavings/savingsModel"
import { formatNumber, formatSpend } from "@/shared/helpers/format"

const money = (value: number) => (
  <span className="text-mono-caption text-foreground">
    {formatSpend(value)}
  </span>
)

/** A group's two costs and its saving, or that it has no price to show. */
function moneyColumns<Row extends SavingsByAgentType | SavingsBySwap>(
  isMobile: boolean,
): DataTableColumn<Row>[] {
  const priced = (row: Row, value: number) =>
    pricedCount(row) > 0 ? (
      money(value)
    ) : (
      <span className="text-caption">{basisOf(row)}</span>
    )
  return [
    {
      id: "baseline",
      header: "Would have spent",
      align: "end",
      cell: (row) => priced(row, row.baseline_cost),
    },
    {
      id: "spent",
      header: "Spent",
      align: "end",
      cell: (row) => priced(row, spentOn(row)),
    },
    {
      id: "saved",
      header: "Saved",
      align: "end",
      cell: (row) =>
        pricedCount(row) > 0 ? (
          <span
            className={`text-mono-caption ${row.net_saved < 0 ? "text-danger" : "text-foreground"}`}
          >
            {formatSaved(row.net_saved)}
          </span>
        ) : (
          <span className="text-caption">{basisOf(row)}</span>
        ),
    },
    ...(isMobile
      ? []
      : ([
          {
            id: "basis",
            header: "Basis",
            cell: (row) => <span className="text-caption">{basisOf(row)}</span>,
          },
        ] satisfies DataTableColumn<Row>[])),
  ]
}

/** What each subagent type saved, largest saving first. */
export function AgentTypeBreakdown({ rows }: { rows: SavingsByAgentType[] }) {
  const isMobile = useIsMobile()
  const columns: DataTableColumn<SavingsByAgentType>[] = [
    {
      id: "agent",
      header: "Subagent type",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-foreground">{row.agent_type}</span>
          <span className="truncate text-caption">
            {formatNumber(row.decisions)}{" "}
            {row.decisions === 1 ? "recommendation" : "recommendations"}
          </span>
        </span>
      ),
    },
    ...moneyColumns<SavingsByAgentType>(isMobile),
  ]
  return (
    <section aria-label="By subagent type" className="flex flex-col gap-4">
      <h2 className="text-title">By subagent type</h2>
      <TableScrollFrame className="flex flex-col gap-2">
        <DataTable
          ariaLabel="Savings by subagent type"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.agent_type}
          mobileLayout="cards"
        />
      </TableScrollFrame>
    </section>
  )
}

/** What each swap of a baseline model for the model run on saved. */
export function SwapBreakdown({
  rows,
  windowWords,
}: {
  rows: SavingsBySwap[]
  /** The period in words, for the empty table. */
  windowWords: string
}) {
  const isMobile = useIsMobile()
  const columns: DataTableColumn<SavingsBySwap>[] = [
    {
      id: "swap",
      header: "From, to",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-col">
          <span className="text-mono-caption text-foreground [overflow-wrap:anywhere]">
            {row.ran_on}
          </span>
          <span className="text-caption [overflow-wrap:anywhere]">
            instead of {row.baseline_model}
          </span>
        </span>
      ),
    },
    ...moneyColumns<SavingsBySwap>(isMobile),
  ]
  return (
    <section aria-label="By model" className="flex flex-col gap-4">
      <h2 className="text-title">By model</h2>
      <TableScrollFrame className="flex flex-col gap-2">
        <DataTable
          ariaLabel="Savings by model"
          columns={columns}
          rows={rows}
          getRowKey={(row) => `${row.baseline_model} ${row.ran_on}`}
          emptyContent={
            <EmptyMessage>
              No subagent has reported what it consumed {windowWords}.
            </EmptyMessage>
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
    </section>
  )
}
