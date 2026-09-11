import { type ReactNode, useState } from "react"
import { useConfirmationFocus } from "@/design-system/hooks/useConfirmationFocus"
import { RowAction } from "./RowAction"

/**
 * A destructive row action that asks twice, in text rather than in buttons.
 *
 * Armed, the label becomes the confirmation in danger ink with a plain Cancel
 * beside it. It replaces `ConfirmButton` on a table row for the same reason the
 * other actions lost their boxes; `ConfirmButton` stays for the forms and cards
 * where a destructive control is the only control and a button is right.
 *
 * Neither is for a delete: a record's deletion goes through `ConfirmDialog`,
 * because a confirmation armed inside the row reads as part of the table rather
 * than as a decision, and has nowhere to say what the deletion costs
 * (otari-ai#2110).
 */
export function ConfirmRowAction({
  confirmLabel,
  onConfirm,
  isPending,
  children,
}: {
  confirmLabel: string
  onConfirm: () => void
  isPending?: boolean
  children: ReactNode
}) {
  const [armed, setArmed] = useState(false)
  // Cancelling unmounts the focused Cancel button, which has no counterpart at
  // rest, and focus lands on `<body>`: the way out of a destructive action
  // costs a keyboard user their place. Arming does not have that problem, but
  // only by accident, because React reuses the trigger's own node as Confirm;
  // the hook covers both so the accident stops being load-bearing.
  const { triggerRef, confirmRef } = useConfirmationFocus(armed)
  if (armed) {
    return (
      <>
        <RowAction
          ref={confirmRef}
          isDanger
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </RowAction>
        <RowAction isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </RowAction>
      </>
    )
  }
  return (
    <RowAction ref={triggerRef} onPress={() => setArmed(true)}>
      {children}
    </RowAction>
  )
}
