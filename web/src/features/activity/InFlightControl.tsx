/**
 * Requests in flight, reported beside the Activity page's refresh control.
 *
 * A usage row is written when a request settles, so the log alone can only
 * describe the past: on a slow backend a 30-second call is invisible for its
 * whole duration. What is running right now is therefore reported as a count
 * that opens the list, rather than as rows pinned above the log.
 *
 * In-flight requests stay out of the table for the same reason the log is
 * frozen (see `useUsageLogs`): rows re-derived on a 2s poll reorder the top of
 * the table continuously on a gateway with real traffic, and an operator cannot
 * read a row before it moves. Off to one side, the same information costs the
 * table nothing, and an operator who wants the live view opens it deliberately.
 *
 * The trade: a request does not resolve in place from live row into settled
 * row. It leaves the list when it lands and appears in the log at the next
 * refresh.
 */

import { Button, Popover } from "@heroui/react"
import { useState } from "react"

import type { InFlightResponse } from "@/client"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { formatNumber } from "@/shared/helpers/format"
import { InFlightWait } from "./InFlightWait"

// The live count, and the list behind it. Reports the gateway as a whole and says
// so: the endpoint takes no filters, so scoping the label to the current view
// would claim a narrowing that was never applied.
//
// The list is rendered only while the popover is open, so the per-entry 1s ticks
// exist only for as long as someone is reading them; closed, this is one number
// that changes every couple of seconds well away from the table.
//
// The Button is a direct child of the popover root rather than wrapped in
// `Popover.Trigger`, which renders its own `role="button"` div and would nest one
// control inside another. HeroUI's Button is a react-aria Button, so the root
// wires it up through context.
//
// Open state is held here, and the whole control renders only while there is
// something to report *or* the list is open. An idle gateway therefore shows no
// control at all, but a list an operator has opened is not torn out from under
// them when the request they were watching lands: it stays, reading "0 in flight",
// until they close it. That is also why the count is controlled rather than left
// to `DialogTrigger`'s own state, which unmounting would discard.
export function InFlightControl({
  data,
  updatedAt,
}: {
  data: InFlightResponse
  updatedAt: number
}) {
  // An in-flight row carries no alias (the registry is in memory and never
  // touches the users table), so the roster is the only name available here.
  const memberLabels = useMemberAttributionLabels()
  const [isOpen, setIsOpen] = useState(false)
  const shown = data.requests
  const hidden = Math.max(0, data.total - shown.length)
  if (data.total === 0 && !isOpen) return null
  return (
    <Popover isOpen={isOpen} onOpenChange={setIsOpen}>
      <Button size="sm" variant="ghost">
        {/* Decorative: the count beside it carries the same meaning in text, so
            nothing is encoded in motion alone. That is also why it can stop
            outright under `prefers-reduced-motion`, being the one element on the
            page that would otherwise animate indefinitely. Still at zero, where
            a pulse would suggest activity that is not there. */}
        <span
          className={`mr-1.5 inline-block h-1.5 w-1.5 motion-reduce:animate-none ${
            data.total > 0 ? "animate-pulse bg-accent" : "bg-muted"
          }`}
          aria-hidden="true"
        />
        {formatNumber(data.total)} in flight
      </Button>
      <Popover.Content placement="bottom end">
        <Popover.Dialog>
          <div className="flex w-80 flex-col gap-2">
            <Popover.Heading className="text-title">In flight</Popover.Heading>
            <p className="text-caption">
              Running right now, across the whole gateway; longest-running
              first. Not narrowed by the filters above.
            </p>
            {shown.length === 0 ? (
              <p className="text-sm text-muted">
                Nothing running right now. Newly settled requests join the log
                at the next refresh.
              </p>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {shown.map((request) => (
                  <li
                    key={request.id}
                    className="flex items-baseline justify-between gap-3 text-sm"
                  >
                    <span className="min-w-0">
                      <span className="block truncate">{request.model}</span>
                      <span className="block truncate text-caption">
                        {request.user_id === null
                          ? "—"
                          : userDisplay(request.user_id, null, memberLabels)
                              .label}
                        {request.policy_name ? ` · ${request.policy_name}` : ""}
                      </span>
                    </span>
                    <InFlightWait
                      startedAtMs={updatedAt - request.elapsed_ms}
                    />
                  </li>
                ))}
              </ul>
            )}
            {/* Only when the endpoint's cap actually bit, which takes more
                concurrency than a live list can usefully show anyway. */}
            {hidden > 0 ? (
              <p className="text-caption">
                {formatNumber(hidden)} further{" "}
                {hidden === 1 ? "request is" : "requests are"} in flight beyond
                the {formatNumber(shown.length)} listed.
              </p>
            ) : null}
          </div>
        </Popover.Dialog>
      </Popover.Content>
    </Popover>
  )
}
