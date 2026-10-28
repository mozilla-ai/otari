import { useRef } from "react"

import type { UsageEntry } from "@/client"
import { PeekPanel } from "@/shared/components/PeekPanel"
import { formatDateTime } from "@/shared/helpers/format"
import { displayStatus } from "./activityModel"
import { RequestDetail } from "./RequestDetail"
import { RequestTraceLink } from "./RequestTraceLink"
import { StatusMark } from "./StatusMark"

/** One request of the log, open beside it. */
export function RequestPeek({
  entry,
  onPriceModel,
  onClose,
  onPrevious,
  onNext,
}: {
  entry: UsageEntry
  onPriceModel?: (model: string, endpoint: string) => void
  onClose: () => void
  onPrevious?: () => void
  onNext?: () => void
}) {
  const headingRef = useRef<HTMLHeadingElement>(null)
  return (
    <PeekPanel
      label="Request detail"
      kind="Request"
      noun="request"
      identity={
        <span className="truncate text-mono-caption text-muted">
          {formatDateTime(entry.timestamp)}
        </span>
      }
      recordKey={entry.id}
      focusRef={headingRef}
      onClose={onClose}
      onPrevious={onPrevious}
      onNext={onNext}
    >
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex min-w-0 flex-col gap-1">
          {/* Focused only by the panel opening, so it draws no ring: it is
              not a control. */}
          <h2
            ref={headingRef}
            tabIndex={-1}
            className="truncate text-title outline-none"
          >
            {entry.model}
          </h2>
          <StatusMark status={displayStatus(entry)} />
        </div>
        <RequestTraceLink entry={entry} />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <RequestDetail entry={entry} onPriceModel={onPriceModel} />
      </div>
    </PeekPanel>
  )
}
