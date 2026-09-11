import { useEffect, useRef } from "react"
import { FiCheck } from "react-icons/fi"
import type { ActivationAttempt } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { Dialog } from "@/design-system/feedback/Dialog"
import {
  confettiOriginOf,
  fireSetupConfetti,
} from "@/features/onboarding/setupConfetti"
import { formatCost } from "@/shared/helpers/format"

/**
 * The payoff, in place of the guide: the workspace's first request landed.
 *
 * Deliberately the quietest screen in the flow. One mark, one line, the receipt
 * in mono, two actions. The receipt is a single line rather than a row of stat
 * tiles, because at this moment the numbers are proof that the call was
 * observed rather than something to analyze, and the pages that do analyze them
 * are one press away.
 */
export function SetupSuccess({
  attempt,
  onDismiss,
  onOpenActivity,
}: {
  /** The request that succeeded. Absent if the gateway reported none. */
  attempt?: ActivationAttempt
  onDismiss: () => void
  onOpenActivity: () => void
}) {
  const markRef = useRef<HTMLSpanElement>(null)

  // Thrown from the mark outwards, once, on mount. A no-op under reduced
  // motion, which the helper enforces rather than this screen.
  useEffect(() => {
    const mark = markRef.current
    if (mark) void fireSetupConfetti(confettiOriginOf(mark))
  }, [])

  const receipt = [
    attempt?.model,
    attempt?.latency_ms != null ? `${Math.round(attempt.latency_ms)} ms` : null,
    attempt?.cost_usd != null ? formatCost(attempt.cost_usd) : null,
  ].filter(Boolean)

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onDismiss()
      }}
      size="sm"
      align="center"
      isAnnouncement
      title="Your first request went through"
      description="This workspace is serving traffic. Usage, spend and the activity log fill in from here."
      mark={
        <span
          ref={markRef}
          className="bg-success-subtle flex size-12 items-center justify-center"
        >
          <FiCheck aria-hidden className="text-success size-6" />
        </span>
      }
      actions={
        <>
          <Button onPress={onDismiss}>Dismiss</Button>
          {/* Scoped to gateway traffic: imported usage is somebody else's
              requests, and the one that just landed is the newest row of what
              is left. */}
          <Button variant="primary" onPress={onOpenActivity}>
            Open the activity log
          </Button>
        </>
      }
    >
      {receipt.length > 0 ? (
        <p className="text-caption text-center font-mono break-all">
          {receipt.join(" · ")}
        </p>
      ) : null}
    </Dialog>
  )
}
