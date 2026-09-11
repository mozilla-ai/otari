import { Link } from "@tanstack/react-router"

import { Button } from "@/design-system/actions/Button"
import { SetupOrb, type SetupOrbPhase } from "@/features/onboarding/SetupOrb"
import type { SetupFailure } from "@/features/onboarding/setupFailureCopy"
import type { SetupSnippetId } from "@/features/onboarding/setupSnippets"
import { formatRelative } from "@/shared/helpers/format"

/** Shown under every cause but the fallback, which carries its own advice. */
const STILL_LISTENING = "Still listening. Send it again and we'll finish setup."

/**
 * The contents of the sheet's pinned band: the one part of the screen that is
 * about traffic, and so the one that reports a failed attempt.
 *
 * It draws no surface of its own. The band is `Dialog`'s `status` slot, which
 * owns the tint, the rule above it and the height, so this is a row of content
 * and nothing else.
 *
 * **A failure dresses this row rather than replacing it.** The guide is still
 * watching for the next request, so the orb keeps turning and "Check now" stays
 * exactly where it was; what changes is the copy and the ink on it. Moving the
 * news into a banner above the code sample would put it away from the thing it
 * is news about, and moving the button would ask an operator to find it again
 * on the one screen where they are least able to.
 */
export function ListeningPanel({
  failure,
  attemptAt,
  isChecking,
  checkFailed,
  onCheckNow,
  onOpenTab,
  onLeave,
}: {
  /** Set when the latest attempt failed. Absent while simply waiting. */
  failure?: SetupFailure
  /** When that attempt happened, ISO-8601. */
  attemptAt?: string
  isChecking: boolean
  /** Whether the last status check could not be completed. */
  checkFailed: boolean
  onCheckNow: () => void
  /** Switches the example above, for a hint whose answer is on this screen. */
  onOpenTab: (tab: SetupSnippetId) => void
  /** Closes the sheet, for a hint that sends the operator to a page. */
  onLeave: () => void
}) {
  const phase: SetupOrbPhase = isChecking
    ? "checking"
    : checkFailed
      ? "stalled"
      : "waiting"
  const hint = failure?.hint

  return (
    <div className="flex w-full flex-wrap items-center justify-between gap-4">
      <div className="flex min-w-0 flex-1 items-center gap-3.5">
        <SetupOrb phase={phase} />
        {/* `role="status"` on the copy column alone: the orb beside it is
            decorative, and a live region wrapped around a canvas that repaints
            every frame is a live region that never stops announcing. */}
        <div className="flex min-w-0 flex-col gap-0.5" role="status">
          {failure ? (
            <>
              <div className="flex min-w-0 flex-wrap items-baseline gap-2">
                <span className="text-mono-overline text-danger">
                  Request failed
                </span>
                {attemptAt ? (
                  <span className="text-mono-micro text-subtle truncate">
                    {formatRelative(attemptAt)}
                  </span>
                ) : null}
              </div>
              <span className="text-emphasis">{failure.cause}</span>
              <span className="text-caption text-subtle">
                {failure.hint === undefined && attemptAt === undefined
                  ? "Still listening. Check the request details, fix the issue, and send it again."
                  : STILL_LISTENING}
              </span>
              {hint?.tab !== undefined ? (
                <Button
                  size="sm"
                  variant="ghost"
                  className="self-start px-0"
                  onPress={() => onOpenTab(hint.tab)}
                >
                  {hint.label}
                </Button>
              ) : null}
              {hint?.to !== undefined ? (
                // The sheet is over the page this link goes to, so it closes on
                // the way rather than navigating behind its own backdrop.
                <Link
                  to={hint.to}
                  onClick={onLeave}
                  className="text-caption text-link hover:text-link-hover self-start"
                >
                  {hint.label}
                </Link>
              ) : null}
            </>
          ) : (
            <>
              <span className="text-emphasis">
                {checkFailed
                  ? "Status could not be checked"
                  : "Listening for your first request"}
              </span>
              <span className="text-caption text-subtle">
                {checkFailed
                  ? "Keep this tab open and try again."
                  : "Keep this tab open. We notice it within a few seconds."}
              </span>
            </>
          )}
        </div>
      </div>
      {/* Shortened on a phone, where "Check now" pushed the copy beside it to a
          third line and made the band a third taller. The accessible name stays
          "Check now" at both widths, so it does not change under a screen
          reader and it still contains the visible word for voice control. */}
      <Button
        size="sm"
        aria-label="Check now"
        isPending={isChecking}
        onPress={onCheckNow}
      >
        <span className="sm:hidden">Check</span>
        <span className="hidden sm:inline">Check now</span>
      </Button>
    </div>
  )
}
