import type { ContentAccess, ContentReaderKind } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { Badge } from "@/design-system/indicators/Badge"
import { formatDateTime } from "@/shared/helpers/format"

const KIND_LABEL: Record<ContentReaderKind, string> = {
  owner: "Owner",
  admin: "Organization admin",
  break_glass: "Break-glass",
}

const USER_PREFIX = "user:"

/**
 * Names the reader an audit line records: a signed-in user by the name the
 * organization knows them by where it can, and the deployment master key,
 * which names nobody, as itself.
 */
export function readerLabel(
  reader: string,
  names: ReadonlyMap<string, string>,
): string {
  if (reader === "master_key") return "Master key"
  if (!reader.startsWith(USER_PREFIX)) return reader
  const id = reader.slice(USER_PREFIX.length)
  return names.get(id) ?? `Identity ${id.slice(0, 8)}`
}

function rowKey(row: ContentAccess): string {
  return `${row.accessed_at}:${row.trace_id}:${row.span_id}:${row.reader}`
}

/** Who read a workspace's captured content, when, as what, and why. */
export function ContentReadsList({
  reads,
  names,
  isLoading,
}: {
  reads: ContentAccess[]
  /** Display names by user id, where the organization's roster is readable. */
  names: ReadonlyMap<string, string>
  isLoading: boolean
}) {
  const columns: DataTableColumn<ContentAccess>[] = [
    {
      id: "when",
      header: "When",
      isRowHeader: true,
      cell: (row) => (
        <span className="text-body tabular-nums">
          {formatDateTime(row.accessed_at)}
        </span>
      ),
    },
    {
      id: "reader",
      header: "Who",
      cell: (row) => (
        <span className="text-body">{readerLabel(row.reader, names)}</span>
      ),
    },
    {
      id: "kind",
      header: "As",
      cell: (row) => (
        <Badge tone={row.reader_kind === "break_glass" ? "warn" : "muted"}>
          {KIND_LABEL[row.reader_kind]}
        </Badge>
      ),
    },
    {
      id: "reason",
      header: "Reason",
      // Only a break-glass read states one; the others are reads the
      // reader's own role allows.
      cell: (row) =>
        row.reason ? (
          <span className="text-body whitespace-normal break-words">
            {row.reason}
          </span>
        ) : null,
    },
    {
      id: "session",
      header: "Session",
      cell: (row) => (
        <code title={row.trace_id} className="text-mono-caption">
          {row.trace_id.slice(0, 10)}
        </code>
      ),
    },
  ]
  return (
    // Scrolls inside its own frame on a phone rather than the page.
    <div className="overflow-x-auto">
      <div className="min-w-[40rem]">
        <DataTable
          ariaLabel="Content reads"
          columns={columns}
          rows={reads}
          getRowKey={rowKey}
          isLoading={isLoading}
          emptyContent={
            <EmptyMessage>
              No one has read this workspace's content yet.
            </EmptyMessage>
          }
        />
      </div>
    </div>
  )
}
