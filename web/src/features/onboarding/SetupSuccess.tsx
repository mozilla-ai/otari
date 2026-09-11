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
 * The receipt: what the request that landed was, in three cells divided by
 * rules.
 *
 * A strip rather than one mono line, because the three are different kinds of
 * fact and an operator reads down to the one they want. Deliberately not
 * `KpiStrip`: that is a page band at 30px with a subgrid and five tracks, and
 * this is three small values inside a dialog. One consumer, so it lives here.
 *
 * A cell whose value the gateway did not report is dropped rather than shown
 * empty, and the model cell takes the slack because a model id is the long one.
 */
function Receipt({ attempt }: { attempt: ActivationAttempt }) {
  // Fixed widths on the two figures, and the model takes the slack. Sized to
  // content they clipped instead: a cost is six decimal places wide and the
  // cell was as narrow as the word above it.
  const cells = [
    { label: "Model", value: attempt.model, width: "min-w-0 flex-1 px-6" },
    {
      label: "Latency",
      value:
        attempt.latency_ms != null
          ? `${Math.round(attempt.latency_ms)} ms`
          : null,
      width: "w-28 shrink-0 px-4",
    },
    {
      label: "Cost",
      value: attempt.cost_usd != null ? formatCost(attempt.cost_usd) : null,
      width: "w-32 shrink-0 pr-6 pl-4",
    },
  ].filter((cell) => cell.value)

  if (cells.length === 0) return null

  return (
    <div className="border-border flex border-t">
      {cells.map((cell, index) => (
        <div
          key={cell.label}
          className={`flex flex-col gap-1 py-3 ${cell.width} ${
            index > 0 ? "border-border border-l" : ""
          }`}
        >
          <span className="text-mono-overline">{cell.label}</span>
          {/* Only the model truncates. A figure that did would be a wrong
              number rather than a shortened one. */}
          <span
            className={`text-mono-caption text-foreground ${
              cell.label === "Model" ? "truncate" : ""
            }`}
          >
            {cell.value}
          </span>
        </div>
      ))}
    </div>
  )
}

/**
 * The payoff, in place of the guide: the workspace's first request landed.
 *
 * Deliberately the quietest screen in the flow. One mark beside the heading,
 * the receipt, one action. The numbers are proof that the call was observed
 * rather than something to analyze, and the pages that do analyze them are one
 * press away.
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

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onDismiss()
      }}
      size="md"
      isAnnouncement
      title="Your first call went through"
      description="Otari observed the request and finished setup for this workspace."
      mark={
        <span ref={markRef} className="mt-0.5 flex shrink-0">
          <FiCheck aria-hidden className="text-success size-6" />
        </span>
      }
      actions={
        // Scoped to gateway traffic: imported usage is somebody else's
        // requests, and the one that just landed is the newest row of what is
        // left.
        <Button variant="primary" onPress={onOpenActivity}>
          Continue to the activity log
        </Button>
      }
    >
      {attempt ? <Receipt attempt={attempt} /> : null}
    </Dialog>
  )
}
