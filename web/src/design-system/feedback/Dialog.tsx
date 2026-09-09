import { AlertDialog } from "@heroui/react"
import type { ReactNode } from "react"

/**
 * The shell every dialog in this product sits in.
 *
 * It exists because the nest does not fit in a head: `Backdrop` > `Container` >
 * `Dialog` > `Header` > `Heading`, then a `Body` and a `Footer`, six levels
 * before a call site says anything of its own. Five components had written it
 * out by hand, and the copies had already begun to differ in the things that
 * are easy to get silently wrong: the placement, the dismiss behavior, and
 * whether the body was the element carrying the gap.
 *
 * `ConfirmDialog` is the specialization of this for a destructive action, with
 * the two buttons and the error line built in. Reach for that one when the
 * dialog's whole job is "are you sure"; reach for this one when it holds a form.
 *
 * Controlled only. A dialog opens because something happened elsewhere on the
 * page (a row's Edit, a toolbar's Add), so an uncontrolled variant would need a
 * trigger inside it and none of these have one.
 */
export function Dialog({
  isOpen,
  onOpenChange,
  heading,
  children,
  footer,
  size = "md",
  isDismissable = true,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** The dialog's title. Rendered as its accessible name, so it is required. */
  heading: string
  children: ReactNode
  /** The action row. Its last button is the primary one, on the right. */
  footer?: ReactNode
  size?: "sm" | "md" | "lg"
  /**
   * Whether Escape and a click outside close it.
   *
   * On by default, and a call site should think before turning it off: the only
   * honest reason is a dialog holding unsaved work that would be lost, and even
   * then the fix is usually to keep the dismiss and confirm the discard.
   */
  isDismissable?: boolean
}) {
  return (
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {/* Mounted only while open, which is what the existing dialogs do and
          worth keeping: the body of a form dialog holds controlled inputs, and
          leaving them mounted would carry one row's draft into the next row's
          dialog. */}
      {isOpen ? (
        <AlertDialog.Backdrop
          isDismissable={isDismissable}
          isKeyboardDismissDisabled={!isDismissable}
        >
          <AlertDialog.Container placement="center" size={size}>
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>{heading}</AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                {children}
              </AlertDialog.Body>
              {footer ? (
                <AlertDialog.Footer>{footer}</AlertDialog.Footer>
              ) : null}
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  )
}
