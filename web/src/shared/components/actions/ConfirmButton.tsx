import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useState } from "react"
import { useConfirmationFocus } from "@/shared/hooks/useConfirmationFocus"

/**
 * A destructive button that requires a second click to confirm, avoiding a
 * modal dependency for revoke/delete actions.
 *
 * Neutral, then danger. The first click is SAFE: it arms this control and
 * destroys nothing, so spending the danger hue on it would spend the loudest
 * signal in the product on a reversible action. Color marks the irreversible
 * step. That is also what keeps the two steps apart now that `danger-soft` is
 * gone: as two shades of danger they were the same variant at the same size,
 * and the only difference was the label.
 *
 * THE CANCEL THAT APPEARS WHEN ARMED IS LOAD-BEARING. Do not simplify it away.
 * The escalation this control now carries is hue-only, and hue is the one
 * channel a red-green deficiency removes: measured, the ghost edge against the
 * danger edge is 1.17 in light and 1.11 in dark, which is no luminance
 * difference at all, and under simulated protanopia in light it is 1.03, so the
 * armed and resting states are indistinguishable. What survives is structural:
 * a second control appears and the row's layout changes, which no color
 * deficiency hides. Drop the Cancel and the escalation goes to zero for those
 * operators and close to it for everyone.
 */
export function ConfirmButton({
  children,
  confirmLabel,
  onConfirm,
  isPending,
}: {
  children: ReactNode
  confirmLabel: string
  onConfirm: () => void
  isPending?: boolean
}) {
  const [armed, setArmed] = useState(false)
  const { triggerRef, confirmRef } = useConfirmationFocus(armed)

  if (armed) {
    return (
      <span className="inline-flex items-center gap-1">
        <Button
          ref={confirmRef}
          size="sm"
          variant="danger"
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          isDisabled={isPending}
          onPress={() => setArmed(false)}
        >
          Cancel
        </Button>
      </span>
    )
  }

  return (
    <Button
      ref={triggerRef}
      size="sm"
      variant="ghost"
      onPress={() => setArmed(true)}
    >
      {children}
    </Button>
  )
}
