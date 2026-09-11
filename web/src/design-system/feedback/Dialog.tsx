import { Modal } from "@heroui/react"
import { type ReactNode, useEffect, useId, useState } from "react"
import { FiX } from "react-icons/fi"

import { Button } from "../actions/Button"

/** `sm` 440px, `md` 520px (the default), `lg` 640px. */
export type DialogSize = "sm" | "md" | "lg"

/**
 * One band of a dialog's body, divided from the one above it by a hairline that
 * runs the full width of the frame.
 *
 * This is the whole of the dialog's internal layout, and it is a rule rather
 * than a choice: a frame presenting several things divides them the way a page
 * does, edge to edge, instead of floating a card per thing inside a padded
 * column. The padding lives here rather than on the body so the rule can reach
 * both edges.
 */
export function DialogSection({
  className,
  children,
}: {
  /** Layout only. The padding and the rule are the section's own. */
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={`border-border flex flex-col gap-2 border-t px-6 pt-4 pb-5 ${className ?? ""}`}
    >
      {children}
    </div>
  )
}

export interface DialogProps {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** Names what the frame is about. Becomes the dialog's accessible name. */
  title: string
  /** One line under the title. The consequence, not a restatement. */
  description?: ReactNode
  size?: DialogSize
  /**
   * A glyph at the head of the title row, on the title's own line.
   *
   * The success mark is the case this exists for. Beside the heading rather
   * than centered above it, because a frame reporting that something worked is
   * still a frame, and centering one screen of a flow that is otherwise
   * left-aligned makes the payoff read as a different product.
   */
  mark?: ReactNode
  /**
   * Whether this frame is the product's announcement moment, which takes
   * `text-display-sub` rather than the section head every other dialog wears.
   * The type scale reserves that role for exactly one thing per page, "a
   * get-started strip, a first-run panel", and a first-run sheet is it.
   */
  isAnnouncement?: boolean
  /** Whether Escape, a backdrop press and the close control dismiss it. */
  isDismissable?: boolean
  /**
   * A readout pinned between the scrolling body and the footer, on its own
   * tinted band.
   *
   * For the one thing in a frame that must stay on screen while the body
   * scrolls: a status the frame is actually about, rather than more of what it
   * is presenting. A frame with a long body and a live status would otherwise
   * push that status below the fold at exactly the moment it starts changing.
   * The tint is what separates it from the sections above without a second
   * border, and it is the only fill in the frame.
   */
  status?: ReactNode
  /** The footer's left slot: a caption saying what happens next. */
  footerStart?: ReactNode
  /**
   * The footer's controls. Stacked full width on a phone by the same rule
   * `FormDialog`'s footer uses, so DOM, visual and focus order agree at both
   * widths and the control nearest the thumb is the last one written.
   */
  actions?: ReactNode
  /** `DialogSection`s. Each draws the rule above itself. */
  children: ReactNode
}

/**
 * The plain modal frame: a dialog that is neither a form nor a question.
 *
 * The third of the three, and the one to reach for when the other two would be
 * a lie about what the frame does. `FormDialog` is a place to work and owns a
 * submit; `ConfirmDialog` is an `AlertDialog` that interrupts to ask one
 * question. This is a `Modal` divided into bands, and the bands are whatever
 * the caller is presenting: a guided step, a receipt, a thing to read and copy.
 *
 * **The body has no padding of its own.** Every band is a `DialogSection`,
 * which carries the padding and the hairline above it, so the divisions run
 * edge to edge the way a page's do.
 *
 * Controlled only, like every dialog here, and it mounts its body only while
 * open so one opening's state cannot survive into the next.
 *
 * It shares `FormDialog`'s geometry class family in `globals.css`, which is why
 * both sit at the same height, cap at the same viewport budget and become the
 * same full-screen sheet on a phone. A second set of numbers under a second
 * prefix would be one edit away from two dialogs that no longer match.
 */
export function Dialog({
  isOpen,
  onOpenChange,
  title,
  description,
  size = "md",
  mark,
  isAnnouncement = false,
  isDismissable = true,
  status,
  footerStart,
  actions,
  children,
}: DialogProps) {
  const descriptionId = useId()
  // Whether the body has been scrolled away from its top. The header's rule is
  // conditional where every other rule here is not: it is the only one that
  // would otherwise sit under nothing.
  const [isScrolled, setIsScrolled] = useState(false)
  useEffect(() => {
    if (!isOpen) setIsScrolled(false)
  }, [isOpen])

  const hasFooter = footerStart !== undefined || actions !== undefined

  return (
    <Modal
      isOpen={isOpen}
      onOpenChange={(next) => {
        if (next || isDismissable) onOpenChange(next)
      }}
    >
      {/* HeroUI renders a press responder for the trigger slot and warns when
          nothing fills it. These dialogs are driven from state, so the slot is
          filled and hidden rather than absent, and `aria-hidden` keeps the
          duplicate name out of the accessibility tree. Same as `FormDialog`. */}
      <Modal.Trigger aria-hidden className="hidden">
        {title}
      </Modal.Trigger>
      <Modal.Backdrop isDismissable={isDismissable}>
        <Modal.Container
          placement="top"
          className="otari-dialog__container p-0 sm:px-4 sm:pt-[7.5rem] sm:pb-[7.5rem]"
        >
          <Modal.Dialog
            aria-describedby={description ? descriptionId : undefined}
            className={`otari-dialog otari-dialog--${size} flex flex-col p-0`}
          >
            <header
              className={`flex shrink-0 items-start justify-between gap-4 px-6 pt-5 pb-4 ${
                isScrolled ? "border-border border-b" : ""
              }`}
            >
              <div className="flex min-w-0 items-start gap-3">
                {mark}
                <div className="flex min-w-0 flex-col gap-1">
                  <Modal.Heading
                    className={
                      isAnnouncement ? "text-display-sub" : "text-heading"
                    }
                  >
                    {title}
                  </Modal.Heading>
                  {description ? (
                    <p id={descriptionId} className="text-body text-muted">
                      {description}
                    </p>
                  ) : null}
                </div>
              </div>
              {/* Absent rather than dead when the frame cannot be dismissed: the
                  footer's action is the way out in that case. */}
              {isDismissable ? (
                <Button
                  aria-label="Close"
                  isIconOnly
                  size="sm"
                  className="relative -top-1 shrink-0 before:absolute before:-inset-1.5 before:content-['']"
                  onPress={() => onOpenChange(false)}
                >
                  <FiX aria-hidden className="text-muted size-4" />
                </Button>
              ) : null}
            </header>
            {/* `min-h-0` is what lets this scroll rather than push the footer
                off the viewport: a flex child's default `min-height: auto`
                refuses to shrink below its content. No padding: the sections
                own it, so their rules reach both edges. */}
            <div
              onScroll={(event) =>
                setIsScrolled(event.currentTarget.scrollTop > 0)
              }
              className="flex min-h-0 flex-1 flex-col overflow-y-auto"
            >
              {children}
            </div>
            {status !== undefined ? (
              // Outside the scrolling body and above the footer, so it holds
              // its place while the body moves under it.
              <div className="border-border bg-surface-alt flex min-h-18 shrink-0 items-center border-t px-6 py-3.5">
                {status}
              </div>
            ) : null}
            {hasFooter ? (
              <footer className="otari-dialog__footer border-border flex shrink-0 items-center justify-between gap-4 border-t px-6 py-3">
                <div className="min-w-0">{footerStart}</div>
                <div className="otari-dialog__actions flex shrink-0 items-center gap-2">
                  {actions}
                </div>
              </footer>
            ) : null}
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  )
}
