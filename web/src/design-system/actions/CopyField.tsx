import { Button } from "@heroui/react"
import type { ReactElement, ReactNode } from "react"
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react"
import { FiEye, FiEyeOff } from "react-icons/fi"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { copyToClipboard } from "@/design-system/helpers/clipboard"

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

/**
 * What a concealed field shows in place of a credential.
 *
 * One fixed run rather than a bullet per character: the length of a key is
 * itself something not to put on screen, and the same stand-in has to read as
 * hidden inside a snippet built around it.
 */
export const CONCEALED_SECRET = "••••••••••••••••"

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
//
// A value that is a credential is `concealed`: it is handed over without being
// put on screen, and the operator reveals it only if they need to read it.
type CopyFieldProps = {
  label: string
  value: string
  fieldRef?: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>
} & (
  | {
      multiline?: boolean
      /**
       * What the field shows until the operator asks for the value, for a value
       * that carries a credential: the plaintext reaches the DOM only once it
       * has been asked for, while Copy copies the real value either way, so a
       * key can be handed over without being read off the screen (otari-ai#2111).
       *
       * The one-time secret step is the exception and opens revealed, by product
       * ruling: that screen exists to hand the key over, and there is no second
       * chance to read it. The toggle and copying without reading survive there,
       * which is what otari-ai#2111 asked for.
       *
       * A whole-value field passes `CONCEALED_SECRET`. A snippet passes the same
       * snippet built around that stand-in, so what is hidden is the key rather
       * than the request that explains it.
       */
      concealed?: string
      /**
       * Whether the value is revealed, for a caller that owns the state.
       *
       * Passing it makes the field controlled: the toggle reports through
       * `onRevealChange` and shows what the caller says. Several fields sharing
       * one credential share one of these, so they reveal and conceal together
       * rather than one at a time.
       *
       * **A controlled field does not re-conceal on a new value.** Uncontrolled,
       * the reveal is keyed to the value it was asked for (see the comment on
       * `revealedValue` below), so a rotation arriving into a revealed field
       * conceals itself. A controlled caller owns that instead: conceal on a
       * value change, or the replacement is on screen without anyone having
       * asked to see it.
       */
      isRevealed?: boolean
      /** The toggle's press, for a controlled field. */
      onRevealChange?: (next: boolean) => void
      /**
       * Whether an uncontrolled field starts revealed. Concealed by default,
       * and a later value arrives concealed whatever this said.
       */
      defaultRevealed?: boolean
      /**
       * Excluded here rather than guarded at runtime, so a call site that passes
       * both fails to compile. The multiline field is a `<textarea>`, where the
       * right padding an in-field control needs indents every line instead of
       * making room on the first, which is a different design.
       */
      action?: never
    }
  | {
      multiline?: false
      /** Nothing hands out a credential beside a domain-verification action. */
      concealed?: never
      isRevealed?: never
      onRevealChange?: never
      defaultRevealed?: never
      /**
       * A control to sit beside the field, which moves the copy affordance
       * inside the field and drops the button from the label row. Absent, the
       * field renders the arrangement it always has.
       *
       * `ReactElement` rather than `ReactNode`: the latter admits `false`, so
       * `action={enabled && <Button />}` compiled and then silently fell back
       * to the default arrangement, which is the one this exists to replace.
       */
      action: ReactElement
    }
)

export function CopyField({
  label,
  value,
  multiline = false,
  fieldRef,
  action,
  concealed,
  isRevealed,
  onRevealChange,
  defaultRevealed = false,
}: CopyFieldProps) {
  const internalRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(
    null,
  )
  const ref = fieldRef ?? internalRef
  const fieldId = useId()
  const [copied, setCopied] = useState(false)
  // Which value each of these is about, rather than a bare flag, because both
  // outlive the value they were asked for. A second credential rendered into
  // this field is concealed with nothing having to reset it, and a copy that
  // fails after the swap cannot reveal or advertise a credential it was not
  // copying: a rotation with the previous key still on screen would otherwise
  // publish the replacement without anyone asking to see it.
  const [revealedValue, setRevealedValue] = useState<string | undefined>(
    defaultRevealed ? value : undefined,
  )
  const [selectHintFor, setSelectHintFor] = useState<string | undefined>(
    undefined,
  )
  const revealed = isRevealed ?? revealedValue === value
  const setRevealed = (next: boolean) => {
    onRevealChange?.(next)
    if (isRevealed === undefined) setRevealedValue(next ? value : undefined)
  }
  const selectHint = selectHintFor === value
  // Same shape as CopyButton's below: the acknowledgement clears itself on a
  // timer, so the timer has to die with the component (and be replaced rather
  // than stacked when a second copy lands inside the window).
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  // The value on screen right now, which is not what `copy` below closes over:
  // that is the value of the render the press happened in, and a credential can
  // arrive while the attempt is in flight. Controlled mode needs it because
  // `onRevealChange` carries a bare boolean and cannot say which value a reveal
  // was asked for; uncontrolled mode keys on the value itself through
  // `revealedValue`.
  const latestValue = useRef(value)
  // `useLayoutEffect`, not `useEffect`: the reader is a rejected-promise
  // handler, so it runs as a microtask, and a passive effect is scheduled
  // rather than run at commit. A rejection landing after the render that
  // carries the new credential but before that effect would read the old one
  // here and pass the gate below, which is the case this exists to refuse.
  useLayoutEffect(() => {
    latestValue.current = value
  }, [value])
  // The value a failed copy asked to have selected, once revealing it has put
  // it on the field: `select()` in the failure's own tick would span the
  // stand-in, and the value React writes on the revealing render discards a
  // selection made before it anyway.
  const selectOnReveal = useRef<string | undefined>(undefined)

  useEffect(() => () => clearTimeout(resetTimer.current), [])

  useEffect(() => {
    const wanted = selectOnReveal.current
    if (wanted === undefined) return
    selectOnReveal.current = undefined
    // Only once the reveal has landed on the value the copy was for. Another
    // credential arriving in the meantime is concealed, and selecting its
    // stand-in is exactly what this is here to avoid.
    if (!revealed || value !== wanted) return
    ref.current?.focus()
    ref.current?.select()
  }, [revealed, value, ref])

  const isConcealed = concealed !== undefined && !revealed
  const shown = isConcealed ? concealed : value

  const acknowledgeCopy = () => {
    setCopied(true)
    setSelectHintFor(undefined)
    clearTimeout(resetTimer.current)
    resetTimer.current = setTimeout(() => setCopied(false), 2_000)
  }

  const copy = async () => {
    // A concealed field cannot answer a refused copy with "it is selected,
    // press Ctrl/Cmd-C": what is selected is the stand-in, so the hint would be
    // an invitation to copy bullets. It takes the path with the `execCommand`
    // fallback instead, which copies from an offscreen textarea and so needs
    // nothing of the value on screen. Only when that refuses too does it reveal
    // and select, which is the first moment Ctrl/Cmd-C could reach the key.
    if (concealed !== undefined) {
      // The credential this attempt is for. Another can arrive while the copy
      // is in flight, and everything below is keyed on this one so a failure
      // cannot land on its successor.
      const copying = value
      if (await copyToClipboard(copying)) {
        acknowledgeCopy()
        return
      }
      if (ref.current?.value === copying) {
        ref.current.focus()
        ref.current.select()
      } else if (latestValue.current === copying) {
        selectOnReveal.current = copying
        setRevealed(true)
      }
      setSelectHintFor(copying)
      return
    }
    ref.current?.focus()
    ref.current?.select()
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value)
        acknowledgeCopy()
        return
      }
    } catch {
      // fall through to the manual path
    }
    // No Clipboard API (or it threw): the text is selected, so the operator can
    // press Ctrl/Cmd-C. Never claim it was copied.
    setSelectHintFor(value)
  }

  const shared =
    "w-full rounded-lg border border-border bg-surface-alt px-3 py-2 font-mono text-xs text-foreground"

  if (action !== undefined) {
    return (
      <div className="flex flex-col gap-1">
        {/* The label keeps its own row and stays a real `<label>`: it is what
            names the in-field control to a screen reader ("Copy TXT record for
            example.com"), not just a caption over the value. */}
        <label htmlFor={fieldId} className="text-caption">
          {label}
        </label>
        {/* Wraps below `xl`, where 757px of content does not fit beside the
            rail, so `action` drops to its own line rather than squeezing the
            field below the width the whole record needs. */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative w-full xl:w-[37.5rem] xl:shrink-0">
            <input
              id={fieldId}
              ref={ref as React.RefObject<HTMLInputElement>}
              readOnly
              value={value}
              onFocus={(e) => e.currentTarget.select()}
              // Right padding clears the control rather than the value running
              // under it, and the field grows below `md` so the 44px touch
              // floor fits between its borders. The value is never truncated:
              // it scrolls inside the input and stays wholly selectable.
              className={`${shared} min-h-[2.875rem] pr-14 md:min-h-0 md:pr-11`}
            />
            {/* `CopyButton` rather than this component's own button: it falls
                back to `execCommand`, so it actually copies on the plain-HTTP
                origins where the async Clipboard API does not exist. It owns the
                acknowledgement too, which is why the field's own `aria-live`
                line and select hint are absent from this arrangement. */}
            <span className="absolute top-1/2 right-1 -translate-y-1/2">
              <CopyButton value={value} label={label} selectOnFailure={ref} />
            </span>
          </div>
          {action}
        </div>
      </div>
    )
  }

  const selectUnlessConcealed = (
    event: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    // Selecting a stand-in would invite a Ctrl/Cmd-C that copies bullets.
    if (isConcealed) return
    event.currentTarget.select()
  }

  // A credential, so nothing here invites a password manager to remember the
  // field or a spellchecker to underline it.
  const credentialProps =
    concealed === undefined
      ? {}
      : {
          autoComplete: "off",
          autoCorrect: "off",
          autoCapitalize: "off",
          spellCheck: false,
          "data-1p-ignore": true,
          "data-lpignore": "true",
        }

  const revealToggle = (
    <Button
      size="sm"
      variant="ghost"
      isIconOnly
      aria-label={`${revealed ? "Hide" : "Show"} ${label}`}
      onPress={() => setRevealed(!revealed)}
    >
      {revealed ? (
        <FiEyeOff aria-hidden="true" className="h-3.5 w-3.5" />
      ) : (
        <FiEye aria-hidden="true" className="h-3.5 w-3.5" />
      )}
    </Button>
  )

  const field = multiline ? (
    <textarea
      id={fieldId}
      ref={ref as React.RefObject<HTMLTextAreaElement>}
      readOnly
      rows={shown.split("\n").length}
      value={shown}
      onFocus={selectUnlessConcealed}
      className={`${shared} resize-none whitespace-pre`}
      {...credentialProps}
    />
  ) : (
    <input
      id={fieldId}
      ref={ref as React.RefObject<HTMLInputElement>}
      readOnly
      value={shown}
      onFocus={selectUnlessConcealed}
      // Concealed, the right padding clears the toggle rather than the value
      // running under it, and the field grows below `md` so the 44px touch
      // floor fits between its borders.
      className={
        concealed === undefined
          ? shared
          : `${shared} min-h-[2.875rem] pr-14 md:min-h-0 md:pr-11`
      }
      {...credentialProps}
    />
  )

  return (
    <div className="flex flex-col gap-1">
      {/* `gap-2` so a long label and the controls beside it cannot meet: a
          concealed snippet puts two of them in this row. */}
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={fieldId} className="text-caption">
          {label}
        </label>
        <div className="flex shrink-0 items-center gap-1">
          {/* A snippet's toggle sits in the label row rather than in the field,
              for the reason `action` is barred from the multiline variant at
              all: right padding on a textarea indents every line of it. */}
          {concealed !== undefined && multiline ? revealToggle : null}
          <Button size="sm" variant="ghost" onPress={copy}>
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>
      {concealed !== undefined && !multiline ? (
        <div className="relative">
          {field}
          <span className="absolute top-1/2 right-1 -translate-y-1/2">
            {revealToggle}
          </span>
        </div>
      ) : (
        field
      )}
      {/* Announce only the "Copied" event, never the secret itself. */}
      <span aria-live="polite" className="text-xs text-success">
        {copied ? "Copied to clipboard." : ""}
      </span>
      {selectHint ? (
        <span className="text-caption">
          {concealed === undefined
            ? "Selected. Press Ctrl/Cmd-C to copy."
            : "Revealed and selected. Press Ctrl/Cmd-C to copy."}
        </span>
      ) : null}
    </div>
  )
}
