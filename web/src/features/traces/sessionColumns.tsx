/**
 * The Sessions table's columns, in a module of their own so a person's page
 * can draw the same table without loading the Sessions page and the session
 * peek it carries.
 */

import type { TraceSummary } from "@/client"
import type { DataTableColumn } from "@/design-system/data/DataTable"
import { Chip } from "@/design-system/indicators/Chip"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { memberLabel } from "@/features/organization/roles"
import {
  formatDuration,
  formatTraceTimestamp,
  plural,
} from "@/features/traces/format"
import {
  sessionDurationMs,
  sessionName,
  shortId,
  traceKind,
} from "@/features/traces/traceModel"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { HarnessMark } from "@/shared/components/marks/BrandMark"
import { formatCost, formatNumber } from "@/shared/helpers/format"

const EMPTY = <span className="text-subtle">—</span>

/**
 * Who a session ran for, as the rest of the product shows a person: their
 * avatar and name. An id with no member behind it (a service
 * account, someone who left) stays the id, which is all there is to say.
 */
function SessionUser({ userId }: { userId: string }) {
  const members = useOrganizationMembers()
  const member = (members.data ?? []).find(
    (one) => one.attribution_user_id === userId || one.user_id === userId,
  )
  if (!member) {
    return (
      <span className="block max-w-48 truncate text-mono-caption text-muted">
        {userId}
      </span>
    )
  }
  return (
    <span className="flex max-w-56 min-w-0 items-center gap-2.5">
      <MemberAvatar
        userId={member.user_id ?? undefined}
        face={member.avatar_face}
        isCompact
      />
      <span className="truncate text-sm text-foreground">
        {memberLabel(member)}
      </span>
    </span>
  )
}

/** The Sessions table, and a person's sessions (`features/people/PersonSessions.tsx`). */
export function columns(
  peekId: string,
  withMarks: boolean,
): DataTableColumn<TraceSummary>[] {
  return [
    {
      id: "last_activity",
      allowsSorting: true,
      header: "Last activity",
      cell: (session) => (
        <span className="text-mono-caption whitespace-nowrap text-muted">
          {formatTraceTimestamp(session.last_activity_at)}
        </span>
      ),
    },
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (session) => (
        <span className="flex min-w-0 items-center gap-2">
          {withMarks ? (
            <HarnessMark
              harness={session.harness}
              language={session.client_language}
            />
          ) : null}
          <span className="truncate text-body">{sessionName(session)}</span>
          <span className="shrink-0 text-mono-caption whitespace-nowrap text-subtle">
            {shortId(session.trace_id)}
          </span>
          {session.trace_id === peekId ? (
            <span className="sr-only">, open</span>
          ) : null}
        </span>
      ),
    },
    {
      id: "user",
      header: "User",
      cell: (session) =>
        session.user_id ? <SessionUser userId={session.user_id} /> : EMPTY,
    },
    {
      id: "levels",
      allowsSorting: true,
      header: "Levels",
      cell: (session) =>
        session.error_count ? (
          <Chip tone="danger">{plural(session.error_count, "error")}</Chip>
        ) : (
          EMPTY
        ),
    },
    {
      id: "latency",
      allowsSorting: true,
      header: "Latency",
      align: "end",
      cell: (session) => (
        <span className="text-mono-caption">
          {formatDuration(sessionDurationMs(session))}
        </span>
      ),
    },
    {
      id: "tokens",
      allowsSorting: true,
      header: "Tokens",
      align: "end",
      cell: (session) =>
        session.input_tokens || session.output_tokens ? (
          <span className="text-mono-caption whitespace-nowrap">
            {formatNumber(session.input_tokens)} →{" "}
            {formatNumber(session.output_tokens)} (Σ{" "}
            {formatNumber(session.input_tokens + session.output_tokens)})
          </span>
        ) : (
          EMPTY
        ),
    },
    {
      id: "cost",
      allowsSorting: true,
      header: "Total cost",
      align: "end",
      cell: (session) =>
        session.cost ? (
          <span className="text-mono-caption">{formatCost(session.cost)}</span>
        ) : (
          EMPTY
        ),
    },
    {
      id: "tags",
      header: "Tags",
      cell: (session) => <Chip>{traceKind(session)}</Chip>,
    },
  ]
}
