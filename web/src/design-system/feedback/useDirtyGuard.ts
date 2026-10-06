import { useEffect, useState } from "react"

/**
 * Whether a dialog holding unsaved work is asking before it closes.
 *
 * Logic only: each dialog draws the guard in its own footer, because a dialog
 * never opens a dialog. `requestClose` is what every dismissal goes through
 * (Escape, the backdrop, a close control, Cancel): refused while a submit is in
 * flight or the frame cannot be dismissed, it arms the guard when there is work
 * to lose and closes otherwise.
 *
 * The guard clears once the dialog has closed, never on Discard. Cleared there,
 * the footer swaps back mid-press and React reuses the Discard button as the
 * submit, so the click that ends the press saves the form being discarded.
 */
export function useDirtyGuard({
  isOpen,
  onOpenChange,
  isDirty,
  isPending,
  isDismissable = true,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  isDirty: boolean
  isPending: boolean
  isDismissable?: boolean
}) {
  const [isGuarding, setIsGuarding] = useState(false)
  useEffect(() => {
    if (!isOpen) setIsGuarding(false)
  }, [isOpen])

  const requestClose = () => {
    if (isPending || !isDismissable) return
    if (isDirty) {
      setIsGuarding(true)
      return
    }
    onOpenChange(false)
  }

  return {
    isGuarding,
    requestClose,
    keepEditing: () => setIsGuarding(false),
    discard: () => onOpenChange(false),
  }
}
