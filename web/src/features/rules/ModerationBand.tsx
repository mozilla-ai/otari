import { useNavigate } from "@tanstack/react-router"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { MandateConsequenceMark } from "@/features/guardrails/GuardrailStatus"
import { BandEmpty } from "@/features/rules/BandParts"
import { emptyCopy, RULE_TYPE_LABEL } from "@/features/rules/rules"
import { ScopeCell } from "@/features/rules/ScopeCell"
import type { Moderation, ModerationRow } from "@/features/rules/useModeration"
import { Band } from "@/shared/components/Band"

/**
 * The content moderation running in the selected workspace. A row opens the
 * guardrail; a broken one says what it is doing to requests under its name.
 */
export function ModerationBand({ moderation }: { moderation: Moderation }) {
  const navigate = useNavigate()
  const columns: DataTableColumn<ModerationRow>[] = [
    {
      id: "profile",
      header: "Profile",
      isRowHeader: true,
      // A broken guardrail says what it does to requests under its profile,
      // and its row grows to hold the line.
      cell: (row) => (
        <span className="flex min-w-0 flex-col">
          <code
            className="truncate font-mono text-foreground"
            title={row.profile}
          >
            {row.profile}
          </code>
          <MandateConsequenceMark consequence={row.consequence} />
        </span>
      ),
    },
    {
      id: "scope",
      header: "Applies to",
      cell: (row) => <ScopeCell scope={row.scope} />,
    },
    {
      id: "runs_on",
      header: "Runs on",
      cell: (row) => (
        <span className="block truncate text-muted" title={row.runsOn}>
          {row.runsOn}
        </span>
      ),
    },
    {
      id: "checks",
      header: "Checks",
      cell: (row) => (
        <span className="block truncate text-muted" title={row.checks}>
          {row.checks}
        </span>
      ),
    },
    {
      id: "mode",
      header: "Mode",
      cell: (row) => <span className="text-muted">{row.mode}</span>,
    },
    {
      id: "on_unavailable",
      header: "If it can't run",
      cell: (row) => <span className="text-muted">{row.ifItCantRun}</span>,
    },
  ]

  return (
    <Band
      title={RULE_TYPE_LABEL.data}
      count={moderation.isLoading ? undefined : moderation.rows.length}
      isLast
    >
      <TableScrollFrame className="otari-rules-table is-data">
        <DataTable
          ariaLabel={`${RULE_TYPE_LABEL.data} running in ${moderation.workspace?.name ?? "this workspace"}`}
          columns={columns}
          rows={moderation.rows}
          isLoading={moderation.isLoading}
          getRowKey={(row) => row.id}
          onRowAction={(id) =>
            void navigate({ to: "/guardrails/$id", params: { id } })
          }
          emptyContent={
            moderation.error ? (
              ""
            ) : (
              <BandEmpty
                copy={emptyCopy(
                  "data",
                  true,
                  undefined,
                  moderation.workspace?.name,
                )}
              />
            )
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
    </Band>
  )
}
