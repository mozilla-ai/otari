import { Link, useNavigate } from "@tanstack/react-router"
import { useMemo } from "react"
import type { TraceSummary } from "@/client"
import { DataTable } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import type { PersonRow } from "@/features/people/people"
import { type SpendWindow, sessionsSearchFor } from "@/features/spend/period"
import { columns } from "@/features/traces/sessionColumns"
import { useTraces } from "@/shared/api/traces"
import { Band } from "@/shared/components/Band"
import { anyHarnessMark } from "@/shared/components/marks/BrandMark"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/** As many as a glance takes; the rest are a click away in Sessions. */
const SHOWN = 10

/**
 * A person's sessions this month, latest first, as the Sessions page lists
 * them: its columns and cells, without the User column, which would name the
 * same person on every row. A row opens the session in Sessions; All sessions
 * opens Sessions on this person and this month.
 */
export function PersonSessions({
  person,
  userId,
  window,
}: {
  person: PersonRow
  userId: string
  window: SpendWindow
}) {
  const navigate = useNavigate()
  const { selected, memberships, select } = useSelectedWorkspace()
  const traces = useTraces(
    {
      workspaceId: "",
      start: window.start,
      q: "",
      userIds: [userId],
    },
    0,
    SHOWN,
  )
  // Each run's harness logo, as the Sessions list draws it, once any run has one.
  const withMarks = anyHarnessMark(traces.data?.items ?? [])
  const tableColumns = useMemo(
    () =>
      columns("", withMarks)
        .filter((column) => column.id !== "user")
        .map((column) => ({
          ...column,
          allowsSorting: false,
          // A playground run is named for a 40-character id; bounded here so
          // it truncates instead of pushing the table past the page.
          cell:
            column.id === "name"
              ? (session: TraceSummary) => (
                  <span className="block max-w-[18rem]">
                    {column.cell(session)}
                  </span>
                )
              : column.cell,
        })),
    [withMarks],
  )
  const search = { ...sessionsSearchFor(window), user_id: userId }

  // Sessions lists one workspace: open it on one of theirs the caller is in,
  // keeping the current one when it is.
  const isMine = (workspaceId: string) =>
    memberships.some((one) => one.workspace_id === workspaceId)
  const showWorkspace = (workspaceId: string | undefined) => {
    if (workspaceId && workspaceId !== selected?.workspace_id) {
      if (isMine(workspaceId)) select(workspaceId)
    }
  }
  const theirs = (person.member.workspaces ?? []).map(
    (place) => place.workspace_id,
  )
  const onOpenAll = () => {
    if (!selected || !theirs.includes(selected.workspace_id)) {
      showWorkspace(theirs.find(isMine))
    }
  }
  const rows = traces.data?.items ?? []

  return (
    <Band
      title="Sessions this month"
      count={person.unread.sessions ? undefined : person.sessions}
      action={
        <Link
          to="/sessions"
          search={search}
          onClick={onOpenAll}
          className="inline-flex min-h-11 items-center text-sm text-link underline"
        >
          All sessions
        </Link>
      }
      isLast
    >
      <ErrorBanner error={traces.error} />
      <TableScrollFrame className="">
        <DataTable
          ariaLabel={`${person.name}'s sessions this month`}
          columns={tableColumns}
          rows={rows}
          isLoading={traces.isLoading}
          getRowKey={(session) => session.trace_id}
          onRowAction={(traceId) => {
            const session = rows.find((one) => one.trace_id === traceId)
            if (!session) return
            showWorkspace(session.workspace_id)
            void navigate({
              to: "/sessions",
              search: {
                ...search,
                peek: session.trace_id,
                peek_workspace: session.workspace_id,
              },
            })
          }}
          emptyContent={
            traces.error ? (
              ""
            ) : (
              <EmptyMessage>No sessions this month.</EmptyMessage>
            )
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
    </Band>
  )
}
