import type { ReactNode } from "react"

import { Skeleton } from "@/design-system/feedback/Skeleton"
import { RequestDetail } from "@/features/activity/RequestDetail"
import { RequestLogLink } from "@/features/activity/RequestLogLink"
import { useRequestGroups } from "@/shared/api/usage"

/**
 * A gateway LLM call as the request log records it: the call's usage row,
 * read through its request group, in full. Where that row cannot be read (the
 * caller's usage scope does not reach it, or it is gone), the span's own
 * summary stands in, with a link to look for the row on the log.
 */
export function GatewayCall({
  attemptId,
  requestGroupId,
  startMs,
  endMs,
  summary,
}: {
  /** The call's span id, which is its usage row's id. */
  attemptId: string
  requestGroupId: string
  /** When the call started and ended, in epoch milliseconds. */
  startMs: number
  endMs: number
  /** What the span says of itself, shown when the row cannot be read. */
  summary: ReactNode
}) {
  const group = useRequestGroups([requestGroupId])
  if (group.isPending && !group.data) {
    return <Skeleton className="h-24 w-full" ariaLabel="Loading the request" />
  }
  const entry = group.data?.find((row) => row.id === attemptId)
  if (entry) return <RequestDetail entry={entry} />
  return (
    <>
      <div className="flex">
        <RequestLogLink requestId={attemptId} startMs={startMs} endMs={endMs} />
      </div>
      {summary}
    </>
  )
}
