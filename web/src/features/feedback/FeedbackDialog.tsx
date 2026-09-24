import { Modal, Spinner } from "@heroui/react"
import { type RefObject, useEffect, useId, useRef, useState } from "react"
import { FiX } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { RestoreFocus } from "@/design-system/feedback/FormDialog"
import { useDirtyGuard } from "@/design-system/feedback/useDirtyGuard"
import { ProductMark } from "@/design-system/ProductMark"
import { useSendFeedback } from "@/shared/api/feedback"

const NOT_SENT = new Error(
  "That didn’t reach us. Your message is still here; send it again.",
)

/** The gateway's own cap on `FeedbackSubmission.message`; keep the two together. */
const MAX_MESSAGE_LENGTH = 4000

const IS_APPLE = /Mac|iPhone|iPad/.test(navigator.platform)

/**
 * A message to the Otari team, opened from Feedback beside Documentation: in the
 * top bar from `md` up, and in the account menu below it.
 *
 * Not a `FormDialog`: that frame always renders a titled header, a footer rule
 * and Cancel, and this one is a single field and a single action. It wears the
 * same geometry classes, so it sits at the same offset and becomes the same
 * full-screen sheet below 640px, and it shares the dirty guard.
 *
 * Sending swaps the compose for a thank-you in the same frame, which stays
 * until it is dismissed: a timer would take the words away from a slow reader.
 */
export function FeedbackDialog({
  isOpen,
  onOpenChange,
  returnFocusRef,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  returnFocusRef?: RefObject<HTMLElement | null>
}) {
  const headingId = useId()
  const thanksId = useId()
  const thanksRef = useRef<HTMLDivElement>(null)
  const fieldRef = useRef<HTMLTextAreaElement>(null)
  const [message, setMessage] = useState("")
  const [isEmptySend, setIsEmptySend] = useState(false)
  const [isSent, setIsSent] = useState(false)
  const { mutate, reset, isPending, error } = useSendFeedback()
  const guard = useDirtyGuard({
    isOpen,
    onOpenChange,
    isDirty: !isSent && message.trim() !== "",
    isPending,
  })

  // Reset on the way in rather than on the way out, so the frame does not flash
  // back to the compose during its exit.
  useEffect(() => {
    if (!isOpen) return
    setMessage("")
    setIsEmptySend(false)
    setIsSent(false)
    reset()
  }, [isOpen, reset])

  // The compose unmounts under the focused field, so focus moves to the frame,
  // which a screen reader announces by its heading. Through the DOM, because
  // `Modal.Dialog` takes no ref.
  useEffect(() => {
    if (isSent)
      thanksRef.current?.closest<HTMLElement>('[role="dialog"]')?.focus()
  }, [isSent])

  function send() {
    if (isPending || guard.isGuarding) return
    const trimmed = message.trim()
    if (trimmed === "") {
      setIsEmptySend(true)
      return
    }
    mutate({ message: trimmed }, { onSuccess: () => setIsSent(true) })
  }

  return (
    <Modal
      isOpen={isOpen}
      onOpenChange={(next) => {
        if (next) onOpenChange(true)
        else guard.requestClose()
      }}
    >
      {/* Driven from state, so the trigger slot is filled and hidden, as in
          `FormDialog`. */}
      <Modal.Trigger aria-hidden className="hidden">
        Feedback
      </Modal.Trigger>
      <Modal.Backdrop isDismissable={!isPending}>
        <Modal.Container
          placement="top"
          className="otari-form-dialog__container p-0 sm:px-4 sm:pt-[7.5rem] sm:pb-[7.5rem]"
        >
          <Modal.Dialog
            aria-labelledby={headingId}
            aria-describedby={isSent ? thanksId : undefined}
            className="otari-form-dialog otari-form-dialog--md flex flex-col p-0 outline-none"
          >
            {returnFocusRef ? <RestoreFocus target={returnFocusRef} /> : null}
            {/* The sheet's way out on a touch screen, which has no Escape and
                no backdrop to press. Shown wherever the frame is the sheet,
                which is by height as well as by width. */}
            <div className="hidden shrink-0 justify-end pt-1.5 pr-2.5 pl-4 max-sm:flex [@media(height<=639px)]:flex">
              <Button
                aria-label="Close"
                isIconOnly
                className="size-11"
                isDisabled={isPending}
                onPress={guard.requestClose}
              >
                <FiX aria-hidden className="text-muted size-3.5" />
              </Button>
            </div>
            {isSent ? (
              <div
                ref={thanksRef}
                data-feedback-view="sent"
                className="flex min-h-[16.25rem] flex-1 flex-col items-center justify-center gap-6 px-12 py-10 max-sm:px-6 max-sm:pt-0 max-sm:pb-24"
              >
                {/* The hook is on a wrapper because `ProductMark` passes no
                    attribute but `className` through to its `<svg>`. */}
                <span data-feedback-part="mark" className="flex">
                  <ProductMark className="text-accent h-auto w-12" />
                </span>
                <div className="flex flex-col items-center gap-2 text-center">
                  <Modal.Heading
                    id={headingId}
                    data-feedback-part="title"
                    className="font-display text-foreground text-2xl font-normal"
                  >
                    Thank you!
                  </Modal.Heading>
                  <p
                    id={thanksId}
                    data-feedback-part="body"
                    className="text-muted max-w-100 text-base text-balance max-sm:max-w-80"
                  >
                    Every message reaches the people building Otari. We read
                    them all, and they shape what comes next.
                  </p>
                </div>
              </div>
            ) : (
              <form
                data-feedback-view="compose"
                noValidate
                aria-busy={isPending}
                onSubmit={(event) => {
                  event.preventDefault()
                  send()
                }}
                className="flex min-h-0 flex-1 flex-col"
              >
                <Modal.Heading id={headingId} className="sr-only">
                  Feedback
                </Modal.Heading>
                <div className="flex min-h-52 flex-1 flex-col gap-4 px-6 pt-6 pb-4 max-sm:px-4 max-sm:pt-1">
                  <ErrorBanner error={error ? NOT_SENT : undefined} />
                  {/* Native, as the playground composer's is: the shared
                      `TextArea` brings a visible label, a caption row and a
                      field edge, and this frame draws none of the three.
                      `field-sizing` grows the frame with the draft up to the
                      dialog's height cap, past which the field scrolls inside
                      this wrapper's bottom inset rather than against the
                      footer. */}
                  <textarea
                    ref={fieldRef}
                    aria-labelledby={headingId}
                    aria-required="true"
                    // Not on a touch screen, whose keyboard would cover the sheet
                    // the moment it opens.
                    autoFocus={window.matchMedia("(pointer: fine)").matches}
                    value={message}
                    onChange={(event) => {
                      setMessage(event.target.value)
                      setIsEmptySend(false)
                    }}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter") return
                      if (!event.metaKey && !event.ctrlKey) return
                      event.preventDefault()
                      send()
                    }}
                    aria-invalid={isEmptySend || undefined}
                    readOnly={isPending}
                    maxLength={MAX_MESSAGE_LENGTH}
                    placeholder="What’s on your mind?"
                    className="text-foreground placeholder:text-subtle min-h-0 w-full flex-1 resize-none bg-transparent text-base focus:outline-none [field-sizing:content]"
                  />
                </div>
                <footer
                  data-feedback-footer={guard.isGuarding ? "guard" : "send"}
                  className={`flex shrink-0 items-center gap-4 pr-4 pb-4 max-sm:flex-col max-sm:items-stretch max-sm:gap-2 max-sm:pl-4 ${
                    guard.isGuarding || isEmptySend
                      ? "justify-between pl-6"
                      : "justify-end pl-4"
                  }`}
                >
                  {guard.isGuarding ? (
                    <>
                      <p className="text-caption">Unsaved changes</p>
                      <div className="flex items-center gap-2 max-sm:flex-col max-sm:items-stretch max-sm:[&>button]:min-h-11 max-sm:[&>button]:w-full">
                        {/* Back to the draft, since the button pressed is
                            about to unmount under the focus. */}
                        <Button
                          onPress={() => {
                            guard.keepEditing()
                            fieldRef.current?.focus()
                          }}
                        >
                          Keep editing
                        </Button>
                        <Button variant="danger" onPress={guard.discard}>
                          Discard
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      {/* Mounted empty so the reason is announced when it
                          arrives: a live region inserted already filled is
                          often skipped. `contents` keeps it out of the flex
                          row until it has something in it. */}
                      <div role="status" className="contents">
                        {isEmptySend ? (
                          <p className="text-caption text-danger">
                            Write something first.
                          </p>
                        ) : null}
                      </div>
                      {/* Not `isPending`, for the reason `FormDialog` gives: a
                          send in flight is working, not refused, so it keeps
                          its fill and its width while the spinner stands in. */}
                      <Button
                        type="submit"
                        variant="primary"
                        aria-keyshortcuts={
                          IS_APPLE ? "Meta+Enter" : "Control+Enter"
                        }
                        data-feedback-send={isPending ? "pending" : "idle"}
                        className={`relative pr-3 pl-4 max-sm:min-h-11 max-sm:w-full max-sm:px-4 pointer-coarse:px-4 ${
                          isPending ? "pointer-events-none" : ""
                        }`}
                      >
                        <span
                          data-feedback-send-label
                          className="flex items-center gap-3"
                        >
                          Send to the Otari team
                          <span
                            aria-hidden="true"
                            className="text-(length:--text-caption-step) leading-(--text-caption-step--line-height) tracking-[0.06em] opacity-72 pointer-coarse:hidden"
                          >
                            {IS_APPLE ? "⌘↵" : "Ctrl ↵"}
                          </span>
                        </span>
                        {isPending ? (
                          <span
                            data-feedback-send-spinner
                            className="absolute inset-0 flex items-center justify-center"
                          >
                            <Spinner
                              size="sm"
                              color="current"
                              aria-hidden="true"
                            />
                          </span>
                        ) : null}
                      </Button>
                    </>
                  )}
                </footer>
              </form>
            )}
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  )
}
