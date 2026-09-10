import { Modal, Spinner } from "@heroui/react"
import { type ReactNode, useEffect, useRef, useState } from "react"
import { FiX } from "react-icons/fi"

import { Button } from "../actions/Button"
import { ErrorBanner } from "./ErrorBanner"

/** `sm` 440px, `md` 520px (the default), `lg` 640px. */
export type FormDialogSize = "sm" | "md" | "lg"

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
  /**
   * `Field`, `SecretField`, `Select`, `Checkbox`. A field with a description
   * takes `reserveMessage`, so an error replaces that line instead of moving
   * the footer; one with nothing to say under it reserves nothing.
   */
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
  const bodyRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!isOpen) {
      setIsGuarding(false)
      setIsScrolled(false)
    }
  }, [isOpen])
  // A failed submit mounts `ErrorBanner` at the top of the body, which in a
  // scrolled `lg` body is above the fold: `role="alert"` reaches a screen
  // reader, and a sighted operator watches the submit finish and sees nothing
  // change. So the body goes back to the top, where the banner is.
  useEffect(() => {
    if (error === undefined || error === null) return
    if (bodyRef.current) bodyRef.current.scrollTop = 0
    setIsScrolled(false)
  }, [error])

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
      {/* No `bg-backdrop/50`: globals.css dims `.modal__backdrop--opaque` in
          an unlayered rule that outranks the utility, so the class changes
          nothing (#1033). */}
      <Modal.Backdrop isDismissable={!isPending}>
        <Modal.Container
          placement="top"
          // 120px from the top on a desktop viewport: a dialog optically
          // centered sits high, and a form that grows as fields appear should
          // grow downward rather than creep up the screen. The same 120px below,
          // because the gap is symmetric by design and the height cap the body
          // scrolls at is the viewport minus both. Below `sm` the container is
          // the sheet's own frame and takes no padding.
          className="p-0 sm:px-4 sm:pt-[7.5rem] sm:pb-[7.5rem]"
        >
          {/* No edge spelled here: globals.css gives every floating surface one
              opaque control-border hairline, unlayered, and argues that tier by
              name, so a `border-border` here would lose to it. */}
          <Modal.Dialog
            className={`otari-form-dialog otari-form-dialog--${size} flex flex-col p-0`}
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
              {/* `isIconOnly` at `sm` is what makes this a 32px square with no
                  edge: the rule that drops a ghost's border is keyed on the
                  control being icon-only, so a `border-0` here would be a call
                  site trying to remember something the system already knows.
                  Lifted 4px to center on the title's own line rather than on
                  the header block, with the 44px target bled outward so
                  raising it moves no layout. */}
              <Button
                aria-label="Close"
                isIconOnly
                size="sm"
                className="relative -top-1 shrink-0 before:absolute before:-inset-1.5 before:content-['']"
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
                // `isGuarding` as well as `isPending`: the guard has taken the
                // footer, so the submit control is not on screen, and a
                // keyboard submit under it runs the mutation from a footer
                // offering only Keep editing and Discard.
                if (!isPending && !isGuarding) onSubmit()
              }}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return
                if (!event.metaKey && !event.ctrlKey) return
                event.preventDefault()
                // `requestSubmit`, not `onSubmit` directly: it runs the form's
                // constraint validation first, so the shortcut and the button
                // are one path rather than two, and a required field left empty
                // cannot reach the mutation through the keyboard alone.
                // Guarded here too, and not only in `onSubmit` where every
                // path funnels: `requestSubmit` runs the form's validation
                // first, which would focus an invalid field to answer a
                // shortcut the guard is refusing.
                if (!isPending && !isGuarding)
                  event.currentTarget.requestSubmit()
              }}
              aria-busy={isPending}
              className="flex min-h-0 flex-col"
            >
              {/* `min-h-0` above and here is what lets this scroll rather than
                  push the footer off the viewport: a flex child's default
                  `min-height: auto` refuses to shrink below its content. */}
              <div
                ref={bodyRef}
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
                      {/* Not `isPending`, and not `isDisabled`. Both land on
                          the product's one disabled treatment, which is 0.4
                          opacity and has to read as *denied*; a submit in
                          flight is working, not refused, and the design keeps
                          its fill. So the press is blocked by hand
                          (`pointer-events`, plus the guard in `onSubmit` for
                          the keyboard) and `aria-busy` says what is happening.

                          The spinner replaces the label in place rather than
                          sitting beside it, so the button keeps its resting
                          width and the footer keeps its height. `opacity-0`
                          rather than `invisible`: `visibility: hidden` would
                          take the label out of the accessibility tree and
                          leave the button with no name mid-submit.

                          The busy state is announced from the form's
                          `aria-busy` rather than the button's: react-aria
                          filters unrecognized ARIA off a Button, so an
                          `aria-busy` here reaches no DOM node at all. */}
                      <Button
                        type="submit"
                        variant="primary"
                        className={`relative ${isPending ? "pointer-events-none" : ""}`}
                      >
                        <span className={isPending ? "opacity-0" : undefined}>
                          {submitLabel}
                        </span>
                        {isPending ? (
                          <span className="absolute inset-0 flex items-center justify-center">
                            {/* `current`, not the default `accent`: the accent
                                is this button's own fill, so the default paints
                                teal on teal. `current` inherits the button's
                                ink, which is what makes it white here and would
                                make it right on any other ground. */}
                            <Spinner
                              size="sm"
                              color="current"
                              aria-hidden="true"
                            />
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
