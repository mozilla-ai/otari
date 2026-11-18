import { type RefObject, useEffect, useRef } from "react"

const ROW = "tr, li, [role=row]"
const CONTROL = "button:not(:disabled), a[href], input, [tabindex]"
const HEADING = "h1, h2, h3, h4, h5, h6"
// How long a watch outlives its arm: a delete confirmed and then dismissed
// (Escape) is still in flight when the dialog closes.
const WATCH_MS = 10_000

const rowsOf = (parent: Element) =>
  [...parent.children].filter((child) => child.matches(ROW))

/**
 * Remembers where `from` sits in its row, and when that row leaves the page
 * with focus dropped on `<body>`, moves focus to the next row's same control,
 * else the previous row's, else the nearest heading.
 */
function watchRowRemoval(from: HTMLElement) {
  const row = from.closest(ROW)
  const parent = row?.parentElement
  if (!row || !parent) return undefined
  const lane = [...row.querySelectorAll(CONTROL)].indexOf(from)
  const index = rowsOf(parent).indexOf(row)
  let heading: HTMLElement | null = null
  for (let a: Element | null = parent; a && !heading; a = a.parentElement) {
    heading = a.querySelector<HTMLElement>(HEADING)
  }
  let timer: ReturnType<typeof setTimeout> | undefined
  const observer = new MutationObserver(() => settle())
  observer.observe(document.body, { childList: true, subtree: true })

  function stop() {
    observer.disconnect()
    clearTimeout(timer)
  }
  function settle() {
    if (row?.isConnected) return
    // A table puts focus on a row of its own when the focused one leaves.
    const active = document.activeElement
    if (active && active !== document.body && !active.matches(ROW)) return
    const rows = parent?.isConnected ? rowsOf(parent) : []
    const control = [rows[index], rows[index - 1]]
      .map((r) => r?.querySelectorAll<HTMLElement>(CONTROL)[lane])
      .find(Boolean)
    if (control) control.focus()
    else if (heading?.isConnected) {
      heading.tabIndex = -1
      heading.focus()
      heading.addEventListener(
        "blur",
        () => heading?.removeAttribute("tabindex"),
        {
          once: true,
        },
      )
    }
    stop()
  }
  return {
    settle,
    stop,
    expire: () => {
      timer = setTimeout(stop, WATCH_MS)
    },
  }
}

/**
 * Keeps focus on the two-step confirm pattern (a trigger that swaps itself for
 * a Confirm/Cancel pair, and back).
 *
 * The swap unmounts the element that has focus, and the browser answers that by
 * moving focus to `<body>`: a keyboard user pressing Delete loses their place,
 * and a screen reader announces nothing at all. These are the most destructive
 * controls on the page, so the arm step has to hand focus to Confirm and the
 * cancel step has to hand it back. A confirmed delete unmounts the whole row,
 * so the arm step also watches for that (`watchRowRemoval`).
 */
export function useConfirmationFocus(isArmed: boolean): {
  triggerRef: RefObject<HTMLButtonElement | null>
  confirmRef: RefObject<HTMLButtonElement | null>
} {
  const triggerRef = useRef<HTMLButtonElement>(null)
  const confirmRef = useRef<HTMLButtonElement>(null)
  // Nothing has been unmounted before the first arm, so the mount render must
  // not pull focus away from wherever the page put it.
  const wasArmed = useRef(false)
  const watch = useRef<ReturnType<typeof watchRowRemoval>>(undefined)

  useEffect(() => {
    if (isArmed) {
      wasArmed.current = true
      confirmRef.current?.focus()
      // A table builds its cells a commit late, so the trigger can be absent
      // here and present a frame on.
      const start = () => {
        const from = confirmRef.current ?? triggerRef.current
        if (!from) return false
        watch.current?.stop()
        watch.current = watchRowRemoval(from)
        return true
      }
      const frame = start() ? 0 : requestAnimationFrame(start)
      // Not a cleanup that stops: the row can unmount this very commit.
      return () => {
        cancelAnimationFrame(frame)
        watch.current?.expire()
      }
    }
    if (wasArmed.current) {
      triggerRef.current?.focus()
      // The row may already be gone while a dialog held focus.
      requestAnimationFrame(() => watch.current?.settle())
    }
  }, [isArmed])

  return { triggerRef, confirmRef }
}
