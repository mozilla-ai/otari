import { buttonVariants } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { FiList } from "react-icons/fi"

// Either side of the call, so a row stamped at its start or at its end is
// inside the window.
const PAD_MS = 60_000

/**
 * Opens a gateway request on the request log, the way `RequestTraceLink` opens
 * a request's session: a gateway LLM span carries its usage row's id. The
 * window is the call's own, padded, so the row is on the log's first page.
 */
export function RequestLogLink({
  requestId,
  startMs,
  endMs,
}: {
  requestId: string
  /** When the call started and ended, in epoch milliseconds. */
  startMs: number
  endMs: number
}) {
  return (
    <Link
      to="/requests"
      search={{
        request: requestId,
        start_date: new Date(startMs - PAD_MS).toISOString(),
        end_date: new Date(endMs + PAD_MS).toISOString(),
      }}
      // The phone touch floor in globals.css reaches HeroUI's Button only.
      className={`${buttonVariants({ size: "sm", variant: "ghost" })} max-md:min-h-11`}
    >
      <FiList aria-hidden="true" className="size-4" />
      View request
    </Link>
  )
}
