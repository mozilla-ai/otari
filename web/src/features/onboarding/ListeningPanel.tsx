import { Link } from "@tanstack/react-router"

import { Button } from "@/design-system/actions/Button"
import { ScanBorder } from "@/design-system/feedback/ScanBorder"
import { SetupOrb, type SetupOrbPhase } from "@/features/onboarding/SetupOrb"
import type { SetupFailure } from "@/features/onboarding/setupFailureCopy"
import type { SetupSnippetId } from "@/features/onboarding/setupSnippets"
import { formatRelative } from "@/shared/helpers/format"

/** Shown under every cause but the fallback, which carries its own advice. */
const STILL_LISTENING = "Still listening. Fix it and send the request again."

/**
 * The one panel on the sheet that is about traffic, and so the one that reports
 * a failed attempt.
 *
 * **A failure dresses this panel rather than replacing it.** The guide is still
 * watching for the next request, so the orb keeps turning, the sweep keeps
 * running and "Check now" stays exactly where it was; what changes is the tone
 * of the edge and the copy in the middle. Moving the news into a banner above
 * the code sample would put it away from the thing it is news about, and moving
 * the button would ask an operator to find it again on the one screen where
 * they are least able to.
 *
 * The sweep stops only for `stalled`, which is a check that could not complete.
 * That is the one state where the product has genuinely lost the thread, and it
 * is a different thing from a request that arrived and failed.
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
  /** Switches the example below, for a hint whose answer is on this screen. */
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
    <ScanBorder
      isActive={!checkFailed}
      tone={failure ? "danger" : "accent"}
      className={`flex flex-wrap items-center justify-between gap-3 px-4 py-3 ${
        failure ? "bg-danger-subtle" : "bg-surface-alt"
      }`}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <SetupOrb phase={phase} />
        {/* `role="status"` on the copy column alone: the orb beside it is
            decorative, and a live region wrapped around a canvas that repaints
            every frame is a live region that never stops announcing. */}
        <div className="flex min-w-0 flex-col gap-0.5" role="status">
          {failure ? (
            <>
              <span className="text-sm font-medium text-danger">
                Request failed: {failure.cause}
              </span>
              <span className="text-caption">
                {failure.hint === undefined && attemptAt === undefined
                  ? "Still listening. Check the request details, fix the issue, and send it again."
                  : STILL_LISTENING}
                {attemptAt ? ` Last attempt ${formatRelative(attemptAt)}.` : ""}
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
                  className="text-xs font-medium text-link hover:text-link-hover"
                >
                  {hint.label}
                </Link>
              ) : null}
            </>
          ) : (
            <>
              <span className="text-body">
                {checkFailed
                  ? "The gateway could not be checked"
                  : "Listening for your first request"}
              </span>
              <span className="text-caption">
                {checkFailed
                  ? "Leave this open and try again."
                  : "This sheet notices it within a few seconds."}
              </span>
            </>
          )}
        </div>
      </div>
      <Button
        variant="ghost"
        size="sm"
        isPending={isChecking}
        onPress={onCheckNow}
      >
        Check now
      </Button>
    </ScanBorder>
  )
}
