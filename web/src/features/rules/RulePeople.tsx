import { useNavigate } from "@tanstack/react-router"
import { useEffect, useRef, useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { PersonCell } from "@/features/rules/PersonCell"
import {
  peopleBandWords,
  peopleShown,
  type Rule,
  type RulePerson,
} from "@/features/rules/rules"
import { UsedCell } from "@/features/rules/UsedCell"
import { Band } from "@/shared/components/Band"
import { formatUsd } from "@/shared/helpers/format"

/**
 * Everyone a per-person rule reaches who is still in the organization, the
 * first ten unless asked for all. Owners and admins see each person's spend
 * against the limit; anyone else sees the team and the amount, which is all
 * the server gives them.
 */
export function RulePeople({
  rule,
  isManager,
}: {
  rule: Rule
  isManager: boolean
}) {
  const navigate = useNavigate()
  const [isAll, setAll] = useState(false)
  const people = rule.people ?? []
  const max = rule.spend?.max ?? 0
  const rows = peopleShown(people, isManager, isAll)
  const rowKey = (row: RulePerson) => row.memberId ?? row.name
  const tableRef = useRef<HTMLDivElement>(null)
  const [focusKey, setFocusKey] = useState<string>()

  // "Show all" leaves with the press, so focus goes to the first row it
  // added rather than falling to the page: the row in a grid, the card's
  // Open button on a phone. The table renders its new rows a pass after
  // this one, so they are waited for.
  useEffect(() => {
    const table = tableRef.current
    if (focusKey === undefined || !table) return
    const land = () => {
      const item = Array.from(
        table.querySelectorAll<HTMLElement>("[data-key]"),
      ).find((one) => one.dataset.key === focusKey)
      if (!item) return false
      const target =
        item.getAttribute("role") === "row"
          ? item
          : item.querySelector<HTMLElement>("button")
      target?.focus()
      return true
    }
    if (land()) return
    const observer = new MutationObserver(() => {
      if (land()) observer.disconnect()
    })
    observer.observe(table, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [focusKey])

  const person: DataTableColumn<RulePerson> = {
    id: "name",
    header: "Person",
    isRowHeader: true,
    cell: (row) => <PersonCell person={row} />,
  }
  const columns: DataTableColumn<RulePerson>[] = isManager
    ? [
        person,
        {
          id: "spent",
          header: "Spent",
          align: "end",
          cell: (row) => (
            <span className="text-mono-caption text-foreground tabular-nums">
              {formatUsd(row.peak?.spent ?? row.spent ?? 0)}
            </span>
          ),
        },
        {
          id: "used",
          header: "Used",
          align: "end",
          cell: (row) => (
            <UsedCell
              spent={row.peak?.spent ?? row.spent ?? 0}
              max={row.peak?.max ?? max}
              label={`${row.name}’s spend against the limit`}
              meterClassName="w-24"
            />
          ),
        },
      ]
    : [
        person,
        {
          id: "team",
          header: "Workspace",
          cell: (row) => (
            <span className="block truncate text-sm" title={row.team}>
              {row.team}
            </span>
          ),
        },
        {
          id: "limit",
          header: "Limit",
          align: "end",
          cell: () => (
            <span className="text-mono-caption text-foreground tabular-nums">
              {formatUsd(max)}
            </span>
          ),
        },
      ]

  return (
    <Band
      title="People under it"
      count={people.length}
      description={peopleBandWords(rule.formerCount ?? 0)}
    >
      <div ref={tableRef}>
        <TableScrollFrame className="otari-rules-table is-people">
          <DataTable
            ariaLabel={`People under ${rule.name}`}
            columns={columns}
            rows={rows}
            getRowKey={rowKey}
            onRowAction={(key) => {
              if (!rows.some((row) => row.memberId === key)) return
              void navigate({
                to: "/people/$personId",
                params: { personId: key },
              })
            }}
            rowClassName={(row) =>
              row.memberId ? undefined : "cursor-default!"
            }
            emptyContent={
              <EmptyMessage>
                Nobody on the roster is under this rule.
              </EmptyMessage>
            }
            mobileLayout="cards"
          />
        </TableScrollFrame>
      </div>
      {people.length > rows.length ? (
        <div>
          <Button
            size="sm"
            onPress={() => {
              const added = peopleShown(people, isManager, true)[rows.length]
              setFocusKey(added ? rowKey(added) : undefined)
              setAll(true)
            }}
          >
            Show all {people.length}
          </Button>
        </div>
      ) : null}
    </Band>
  )
}
