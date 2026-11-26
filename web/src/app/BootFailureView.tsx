import { Spinner } from "@heroui/react"
import { BootFrame } from "@/app/BootFrame"
import type { BootFailure, BootFailureKind } from "@/app/boot"
import { Button } from "@/design-system/actions/Button"
import { SeverityMark } from "@/design-system/metrics/SeverityMark"
import { formatClockTime } from "@/shared/helpers/format"

type Action = "retryNow" | "reload" | "retry"

interface Copy {
  /** The word beside the status mark, and the page title's first half. */
  mark: string
  heading: string
  body: string
  action: Action
  /** The page title's first half, where it is not the mark's word. */
  title?: string
}

const UNREACHABLE: Copy = {
  mark: "Gateway unreachable",
  heading: "Can’t reach the gateway",
  body: "The dashboard needs an answer from the gateway before it can open. It keeps trying, and opens on its own as soon as one arrives.",
  action: "retryNow",
  title: "Can’t reach the gateway",
}

/**
 * The four failures that keep retrying share their wording on purpose: to the
 * person waiting they are one situation, and the Last try row is where they
 * differ (no response, the deadline, a status, the device offline).
 */
export const BOOT_FAILURE_COPY: Record<BootFailureKind, Copy> = {
  unreachable: UNREACHABLE,
  timedOut: UNREACHABLE,
  serverError: UNREACHABLE,
  offline: UNREACHABLE,
  signInRequired: {
    mark: "Sign-in required",
    heading: "Sign in to continue",
    body: "Something in front of the gateway, such as a proxy or an access gate, wants this browser to sign in first. Reload to open its sign-in page.",
    action: "reload",
  },
  accessRefused: {
    mark: "Access refused",
    heading: "This browser isn’t allowed in",
    body: "Something in front of the gateway refused this browser. Reload to sign in again, or ask whoever runs the gateway for access.",
    action: "reload",
  },
  unexpectedAnswer: {
    mark: "Unexpected answer",
    heading: "This address didn’t answer like a gateway",
    body: "Check that it points at an Otari gateway that serves its dashboard, then retry.",
    action: "retry",
  },
  unchosen: {
    mark: "No gateway chosen",
    heading: "Couldn’t choose a gateway",
    body: "The dashboard asked no gateway, because it couldn’t work out which one to use. Retry once your deployment is available.",
    action: "retry",
  },
}

const ACTION_LABEL: Record<Action, string> = {
  retryNow: "Retry now",
  reload: "Reload",
  retry: "Retry",
}

/**
 * The Last try row after a pressed Retry failed. A status takes the place of
 * "still no answer", because it is the difference between a gateway that
 * replied and one that did not, and the new time already shows it was tried
 * again; with no status, "still no answer" is what happened.
 */
function stillNoAnswer(failure: BootFailure): string {
  return failure.kind === "serverError" ? failure.reason : "still no answer"
}

export function bootFailureTitle(kind: BootFailureKind): string {
  const copy = BOOT_FAILURE_COPY[kind]
  return `${copy.title ?? copy.mark} · Otari`
}

/**
 * The boot screen for one failure, drawn from what is passed in and nothing
 * else, so every state can be rendered without a loader.
 *
 * The diagnostics are deliberately not a live region: they update after every
 * background attempt, and reading each one aloud is chatter. What a screen
 * reader hears is sent once, by the caller, when something changes that it
 * needs to know.
 */
export function BootFailureView({
  failure,
  failedAt,
  host,
  isPending,
  isStillNoAnswer = false,
  onAction,
}: {
  failure: BootFailure
  /** When the last attempt failed, in epoch milliseconds. */
  failedAt: number
  /** The address the dashboard was loaded from, `location.host`. */
  host: string
  /** A pressed Retry is waiting on its attempt. Never a background attempt. */
  isPending: boolean
  /** The last attempt was a pressed Retry, and it failed too. */
  isStillNoAnswer?: boolean
  onAction: () => void
}) {
  const copy = BOOT_FAILURE_COPY[failure.kind]
  return (
    // The sweep is for a screen that is waiting. One that asks the reader to
    // act (sign in, reload) holds still, even while the loader keeps asking.
    <BootFrame isScanning={failure.willRetry && copy.action === "retryNow"}>
      <div className="flex flex-col items-center gap-2.5 text-center">
        <div className="text-mono-overline leading-4">
          <SeverityMark severity={{ status: "alert", word: copy.mark }} />
        </div>
        <h1 className="text-display">{copy.heading}</h1>
      </div>
      <p className="text-center text-sm text-pretty text-muted">{copy.body}</p>
      {failure.reason === "" ? null : (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 border-t border-border pt-3 text-mono-caption leading-(--text-caption-step--line-height) tabular-nums">
          <dt className="text-subtle">Address</dt>
          <dd className="break-all text-muted">{host}</dd>
          <dt className="text-subtle">Last try</dt>
          <dd className={isStillNoAnswer ? "text-foreground" : "text-muted"}>
            {formatClockTime(failedAt)} ·{" "}
            {isStillNoAnswer ? stillNoAnswer(failure) : failure.reason}
          </dd>
        </dl>
      )}
      <Button
        variant="primary"
        fullWidth
        isPending={isPending}
        onPress={onAction}
        // Pending here is a press being answered, not a refusal, so it keeps
        // full ink and says so with the cursor rather than taking the 0.4 that
        // `[data-pending]` borrows from disabled. HeroUI also turns pointer
        // events off on a pending button, which leaves the pointer on the card
        // and the cursor never shows; they come back here, and React Aria's
        // `isPending` still ignores a press until the attempt settles.
        className="h-11 data-[pending=true]:pointer-events-auto data-[pending=true]:cursor-progress data-[pending=true]:opacity-100 md:h-9"
      >
        {isPending ? (
          <>
            <Spinner size="sm" color="current" aria-hidden="true" />
            Retrying…
          </>
        ) : (
          ACTION_LABEL[copy.action]
        )}
      </Button>
    </BootFrame>
  )
}
