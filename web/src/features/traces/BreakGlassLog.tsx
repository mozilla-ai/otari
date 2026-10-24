import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { useBreakGlassLog } from "@/shared/api/traces"
import { formatDateTime } from "@/shared/helpers/format"

/**
 * Every break-glass opening an owner of the organization made, for the other owners to see.
 *
 * The point of recording an opening is that someone else can read the record,
 * so it is on the page owners already use rather than behind a request.
 */
export function BreakGlassLog() {
  const log = useBreakGlassLog(true)
  // Offset pages over a log that grows at its head, so an opening can appear on two; each is listed once.
  const items = [
    ...new Map(
      (log.data?.pages ?? [])
        .flatMap((page) => page.items ?? [])
        .map((read) => [read.id, read]),
    ).values(),
  ]

  return (
    <SettingsGroup title="Break-glass openings">
      <div className="flex flex-col gap-2 py-3">
        <ErrorBanner error={log.error} />
        {log.isPending && !log.data ? null : items.length === 0 ? (
          <p className="text-caption">
            No owner has opened a session's content by break-glass.
          </p>
        ) : (
          <ul className="flex flex-col gap-2" aria-label="Break-glass openings">
            {items.map((read) => (
              <li
                key={read.id}
                className="flex flex-col gap-1 border border-border p-3"
              >
                <span className="text-body">
                  {read.reader_email} opened session {read.trace_id} on{" "}
                  {formatDateTime(read.read_at)}
                </span>
                <span className="text-caption">“{read.reason}”</span>
              </li>
            ))}
            {log.hasNextPage ? (
              <li>
                <Button
                  variant="ghost"
                  onPress={() => void log.fetchNextPage()}
                  isDisabled={log.isFetchingNextPage}
                >
                  Show older openings
                </Button>
              </li>
            ) : null}
          </ul>
        )}
      </div>
    </SettingsGroup>
  )
}
