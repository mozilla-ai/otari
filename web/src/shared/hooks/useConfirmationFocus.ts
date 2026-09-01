import { useEffect, useRef } from "react"

/**
 * Keeps focus on the two-step confirm pattern (a trigger that swaps itself for
 * a Confirm/Cancel pair, and back).
 *
 * The swap unmounts the element that has focus, and the browser answers that by
 * moving focus to `<body>`: a keyboard user pressing Delete loses their place,
 * and a screen reader announces nothing at all. These are the most destructive
 * controls on the page, so the arm step has to hand focus to Confirm and the
 * cancel step has to hand it back.
 */
export function useConfirmationFocus(armed: boolean): {
  triggerRef: React.RefObject<HTMLButtonElement | null>
  confirmRef: React.RefObject<HTMLButtonElement | null>
} {
  const triggerRef = useRef<HTMLButtonElement>(null)
  const confirmRef = useRef<HTMLButtonElement>(null)
  // Nothing has been unmounted before the first arm, so the mount render must
  // not pull focus away from wherever the page put it.
  const wasArmed = useRef(false)

  useEffect(() => {
    if (armed) {
      wasArmed.current = true
      confirmRef.current?.focus()
    } else if (wasArmed.current) {
      triggerRef.current?.focus()
    }
  }, [armed])

  return { triggerRef, confirmRef }
}
