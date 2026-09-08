import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useEffect, useId, useRef, useState } from "react"
import { CopyButton } from "@/shared/components/actions/CopyButton"

// An identifier the operator needs verbatim (a model id, an alias target, a
// request id), rendered so it can be taken either way: highlighted with the mouse
// like ordinary text, or copied in one press.
//
// Highlighting is what needs the help. Inside a react-aria table the row is a
// press target, and a press on it both toggles the row's selection and sets
// `user-select: none` on the row for the duration; the re-render that selection
// causes lands mid-drag and discards the selection the browser had started, so
// dragging across an id would otherwise select nothing at all (issue #478). Keeping the
// press from starting on the value itself is the fix: the pointer sequence stays
// with the browser, which selects text with it. `select-text` then beats the
// inherited `none` from any press elsewhere in the row (an own declaration
// outranks inheritance, so no `!important` is needed). The rest of the row keeps
// its behavior: DataTable still opens a drill-in for a plain click, including one
// on this value, and skips it for the click that ends a drag.
export function CopyableValue({
  value,
  label,
  className,
  children,
}: {
  /** The exact text a copy yields, which is not always what is rendered. */
  value: string
  label: string
  className?: string
  /** Defaults to `value`; pass children when the display form differs. */
  children?: ReactNode
}) {
  const keepPressFromRow = (event: { stopPropagation: () => void }) =>
    event.stopPropagation()
  return (
    <span className="inline-flex items-center gap-1">
      {/* biome-ignore lint/a11y/noStaticElementInteractions: the handlers only stop propagation so a text drag survives; there is no action to expose */}
      <span
        // Focusable, but not tabbable: pressing here focuses the value itself
        // instead of the react-aria table cell, whose focus bookkeeping re-renders
        // the row and (again) discards a drag that has only just begun. Without
        // this, the first drag in a freshly loaded table selected nothing and only
        // subsequent ones worked.
        tabIndex={-1}
        className={`select-text outline-none ${className ?? ""}`}
        onPointerDown={keepPressFromRow}
        onMouseDown={keepPressFromRow}
      >
        {children ?? value}
      </span>
      <CopyButton value={value} label={label} />
    </span>
  )
}

// A readonly, always-selectable field with a copy button: how a value an
// operator has to paste elsewhere is handed over. Shared by the Keys page's
// one-time reveal and the setup guide, which hand out the same key and the same
// snippets.
//
// The Clipboard API is undefined on the non-secure origins this dashboard is
// routinely served from, so the text is selected on click and Ctrl/Cmd-C always
// works even when the button cannot copy programmatically. "Copied" is only
// claimed when it truly copied.
//
// The label is a real `<label>` for the field, not a caption beside it: these
// values are handed over in pairs and threes (a key and two snippets), so
// "which field is this" has to be answerable by a screen reader and by a test
// that queries the way an operator reads.
export function CopyField({
  label,
  value,
  multiline = false,
  fieldRef,
}: {
  label: string
  value: string
  multiline?: boolean
  fieldRef?: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>
}) {
  const internalRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(
    null,
  )
  const ref = fieldRef ?? internalRef
  const fieldId = useId()
  const [copied, setCopied] = useState(false)
  const [selectHint, setSelectHint] = useState(false)
  // Same shape as CopyButton's below: the acknowledgement clears itself on a
  // timer, so the timer has to die with the component (and be replaced rather
  // than stacked when a second copy lands inside the window).
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )

  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const copy = async () => {
    ref.current?.focus()
    ref.current?.select()
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value)
        setCopied(true)
        setSelectHint(false)
        clearTimeout(resetTimer.current)
        resetTimer.current = setTimeout(() => setCopied(false), 2_000)
        return
      }
    } catch {
      // fall through to the manual path
    }
    // No Clipboard API (or it threw): the text is selected, so the operator can
    // press Ctrl/Cmd-C. Never claim it was copied.
    setSelectHint(true)
  }

  const shared =
    "w-full rounded-lg border border-border bg-surface-alt px-3 py-2 font-mono text-xs text-foreground"

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label htmlFor={fieldId} className="text-caption">
          {label}
        </label>
        <Button size="sm" variant="ghost" onPress={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      {multiline ? (
        <textarea
          id={fieldId}
          ref={ref as React.RefObject<HTMLTextAreaElement>}
          readOnly
          rows={value.split("\n").length}
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={`${shared} resize-none whitespace-pre`}
        />
      ) : (
        <input
          id={fieldId}
          ref={ref as React.RefObject<HTMLInputElement>}
          readOnly
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={shared}
        />
      )}
      {/* Announce only the "Copied" event, never the secret itself. */}
      <span aria-live="polite" className="text-xs text-success">
        {copied ? "Copied to clipboard." : ""}
      </span>
      {selectHint ? (
        <span className="text-caption">
          Selected. Press Ctrl/Cmd-C to copy.
        </span>
      ) : null}
    </div>
  )
}
