import { Modal, Spinner } from "@heroui/react"
import { type ReactNode, useEffect, useState } from "react"
import { FiX } from "react-icons/fi"

import { Button } from "../actions/Button"
import { ErrorBanner } from "./ErrorBanner"

/** `sm` 440px, `md` 520px (the default), `lg` 640px. */
export type FormDialogSize = "sm" | "md" | "lg"

/**
 * The widths, as an inline custom property rather than a class.
 *
 * `globals.css` pins `.modal__dialog` to `width: fit-content` with
 * `min-width: min(28rem, …)`, unlayered so it outranks `@layer utilities`. A
 * `w-[27.5rem]` here compiles, lints, ships and loses; and even where the width
 * did land, the 448px floor makes `sm` unreachable. The rule that reads this
 * property is unlayered beside that one, so the geometry is settled in CSS and
 * the size a call site picks travels as a value.
 */
const WIDTH: Record<FormDialogSize, string> = {
  sm: "27.5rem",
  md: "32.5rem",
  lg: "40rem",
}

export interface FormDialogProps {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** Names the object being made, not the action: "New key", not "Create key". */
  title: string
  /** One line under the title. The consequence, not a restatement of the title. */
  description?: ReactNode
  size?: FormDialogSize
  /**
   * The submit button's label, and the trigger's, word for word: the control
   * that opened this dialog and the one that completes it say the same thing.
   */
  submitLabel: string
  onSubmit: () => void
  isPending: boolean
  /** The caught value. Rendered by `ErrorBanner` at the top of the body. */
  error?: unknown
  /**
   * Whether the form holds work that closing would lose. Arms the footer guard,
   * so Escape and a click outside ask before discarding.
   */
  isDirty?: boolean
  /** The footer's left slot: a "Create another" `Checkbox`, or a caption. */
  footerStart?: ReactNode
  /** A `TabRow` under the header. `lg` only, where a form has two shapes. */
  tabs?: ReactNode
  /** `Field`, `SecretField`, `Select`, `Checkbox`. Each with `reserveMessage`. */
  children: ReactNode
}

/**
 * The surface every create and edit form opens in.
 *
 * `ConfirmDialog` is the sibling for a destructive "are you sure", and the
 * difference is not decorative: that one is an `AlertDialog`, which interrupts
 * to ask one question, and a form is not an alert. This is a `Modal`.
 *
 * Controlled only, like every dialog here: it opens from a trigger elsewhere on
 * the page (a heading row's Create, an empty state's CTA) and mounts its body
 * only while open, so one row's draft cannot survive into the next row's.
 */
export function FormDialog({
  isOpen,
  onOpenChange,
  title,
  description,
  size = "md",
  submitLabel,
  onSubmit,
  isPending,
  error,
  isDirty = false,
  footerStart,
  tabs,
  children,
}: FormDialogProps) {
  // The dirty guard lives in the footer rather than in a second dialog, because
  // a dialog never opens a dialog.
  const [isGuarding, setIsGuarding] = useState(false)
  // Whether the body has been scrolled away from its top, which is what puts a
  // rule between the pinned header and the content passing under it. The footer
  // keeps its rule in every state, so only this end of the body is conditional.
  const [isScrolled, setIsScrolled] = useState(false)
  useEffect(() => {
    if (!isOpen) {
      setIsGuarding(false)
      setIsScrolled(false)
    }
  }, [isOpen])

  const requestClose = () => {
    if (isPending) return
    if (isDirty) {
      setIsGuarding(true)
      return
    }
    onOpenChange(false)
  }

  return (
    <Modal
      isOpen={isOpen}
      onOpenChange={(next) => {
        if (next) {
          onOpenChange(true)
          return
        }
        requestClose()
      }}
    >
      {/* HeroUI renders a press responder for the trigger slot and warns when
          nothing fills it. These dialogs are driven from state, so the slot is
          hidden rather than absent; `WorkspaceSwitcher` does the same. */}
      <Modal.Trigger className="hidden">{submitLabel}</Modal.Trigger>
      {/* No `bg-backdrop/50` here, despite the two dialogs that carry it:
          globals.css dims `.modal__backdrop--opaque` to 50% in an unlayered
          rule, which outranks the utility. Measured in the built stylesheet,
          the class changes nothing. */}
      <Modal.Backdrop isDismissable={!isPending}>
        <Modal.Container
          placement="top"
          // 120px from the top on a desktop viewport: a dialog optically
          // centered sits high, and a form that grows as fields appear should
          // grow downward rather than creep up the screen. Below `sm` the
          // container is the sheet's own frame and takes no padding.
          className="p-0 sm:px-4 sm:pt-[7.5rem] sm:pb-10"
        >
          {/* No edge spelled here. `globals.css` gives every floating surface
              one opaque control-border hairline, unlayered, in the rule that
              replaced elevation; a `border-border` at this call site would
              compile, lint, ship and lose to it, and it argues against that
              tier by name. */}
          <Modal.Dialog
            className="otari-form-dialog flex flex-col p-0"
            style={
              { "--form-dialog-width": WIDTH[size] } as React.CSSProperties
            }
          >
            <header
              className={`flex shrink-0 items-start justify-between gap-4 px-6 pt-5 ${
                tabs ? "" : "pb-4"
              } ${!tabs && isScrolled ? "border-border border-b" : ""}`}
            >
              <div className="flex flex-col gap-1">
                <Modal.Heading className="text-heading">{title}</Modal.Heading>
                {description ? (
                  <p className="text-body text-muted">{description}</p>
                ) : null}
              </div>
              {/* 32px glyph box lifted 4px so it centers on the title's own
                  line rather than on the header block, with the 44px target
                  bled outward so raising it moves no layout. */}
              <Button
                aria-label="Close"
                className="relative -top-1 size-8 min-h-0 min-w-0 shrink-0 justify-center border-0 p-0 before:absolute before:-inset-1.5 before:content-['']"
                isDisabled={isPending}
                onPress={requestClose}
              >
                <FiX aria-hidden className="text-muted size-3.5" />
              </Button>
            </header>
            {tabs ? (
              <div className="border-border shrink-0 border-b px-6 pt-4 pb-3">
                {tabs}
              </div>
            ) : null}
            <form
              // A real form, so a label associates, a password manager sees a
              // submit, and Enter in a single-line field submits the way it
              // does everywhere else. Cmd/Ctrl+Enter is added on top, for the
              // fields where Enter is a newline.
              onSubmit={(event) => {
                event.preventDefault()
                if (!isPending) onSubmit()
              }}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return
                if (!event.metaKey && !event.ctrlKey) return
                event.preventDefault()
                if (!isPending) onSubmit()
              }}
              className="flex min-h-0 flex-col"
            >
              {/* `min-h-0` above and here is what lets this scroll rather than
                  push the footer off the viewport: a flex child's default
                  `min-height: auto` refuses to shrink below its content. */}
              <div
                onScroll={(event) =>
                  setIsScrolled(event.currentTarget.scrollTop > 0)
                }
                className="flex min-h-0 flex-col gap-4 overflow-y-auto px-6 pt-1 pb-6"
              >
                <ErrorBanner error={error} />
                {children}
              </div>
              <footer className="border-border flex shrink-0 items-center justify-between gap-2 border-t px-6 py-3">
                {isGuarding ? (
                  <>
                    <p className="text-caption">Unsaved changes</p>
                    <div className="flex items-center gap-2">
                      <Button onPress={() => setIsGuarding(false)}>
                        Keep editing
                      </Button>
                      <Button
                        variant="danger"
                        onPress={() => {
                          setIsGuarding(false)
                          onOpenChange(false)
                        }}
                      >
                        Discard
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="min-w-0">{footerStart}</div>
                    <div className="flex shrink-0 items-center gap-2">
                      <Button isDisabled={isPending} onPress={requestClose}>
                        Cancel
                      </Button>
                      {/* The spinner replaces the label in place rather than
                          sitting beside it, so the button keeps its resting
                          width and the footer keeps its height. HeroUI's
                          `isPending` draws no spinner of its own: it sets
                          `data-pending`, which the stylesheet answers with
                          `pointer-events: none` and nothing else. */}
                      <Button
                        type="submit"
                        variant="primary"
                        isPending={isPending}
                        className="relative"
                      >
                        <span className={isPending ? "invisible" : undefined}>
                          {submitLabel}
                        </span>
                        {isPending ? (
                          <span className="absolute inset-0 flex items-center justify-center">
                            <Spinner size="sm" aria-hidden="true" />
                          </span>
                        ) : null}
                      </Button>
                    </div>
                  </>
                )}
              </footer>
            </form>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  )
}
