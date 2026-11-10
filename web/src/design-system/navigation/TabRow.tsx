import { type ReactNode, useEffect, useRef } from "react"

/** How far a scrolling row's edge fades, in rem; `TabRow`'s mask spells it too. */
const EDGE_FADE_REM = 3

/**
 * The classes of a tab, for a `Tab` and for a tab that has to be another
 * element, such as a router link to a view with its own URL. One helper so the
 * two cannot drift; focus comes from the global `:focus-visible` ring either way.
 *
 * A segment never shrinks and never wraps its label: a tab row that squeezed
 * would put the same control at two widths on one page, and a wrapped label
 * would break the row's height. The row scrolls instead.
 */
export function tabClassName(isActive: boolean): string {
  return `shrink-0 px-2.5 py-[0.3125rem] text-sm whitespace-nowrap transition-colors motion-reduce:transition-none aria-disabled:cursor-not-allowed aria-disabled:opacity-50 ${
    isActive
      ? "bg-surface-subtle text-foreground"
      : "text-muted not-aria-disabled:hover:text-foreground"
  }`
}

/**
 * A tab in a row of them, which is what this product's segmented choices are
 * now: an active tab takes the `surface-subtle` fill and the foreground ink, an
 * inactive one is bare muted text. Square, and no accent anywhere, because the
 * accent is data ink and fills rather than a way to say "this one".
 *
 * A `<button>` with `aria-pressed` rather than a real tablist: these switch what
 * a panel shows without being a tab widget's roving-focus contract, and the
 * pages using them already announce the choice through the content beneath.
 *
 * Disabled through `aria-disabled` rather than `disabled`, so the tab stays
 * focusable and its reason reachable: `describedBy` names the id of visible
 * text saying why, which a natively disabled button could never point anyone to.
 */
export function Tab({
  isActive,
  onPress,
  isDisabled = false,
  describedBy = "",
  children,
}: {
  isActive: boolean
  onPress: () => void
  /** Unavailable right now; pair it with `describedBy`. */
  isDisabled?: boolean
  /** The id of the text that explains the tab, typically why it is disabled. */
  describedBy?: string
  children: ReactNode
}) {
  const ref = useRef<HTMLButtonElement>(null)
  // The row scrolls rather than wrapping, so on a phone the active tab can sit
  // past its edge, opened from a link, with nothing to say it is there.
  //
  // The row's own scroll and nothing else: `scrollIntoView` scrolls every
  // ancestor too, which would scroll the page to a tab row below the fold. The
  // tab also clears the row's edge fade, so it is never drawn half transparent.
  useEffect(() => {
    const tab = ref.current
    const row = tab?.parentElement
    if (!isActive || !tab || !row) return
    const fade =
      EDGE_FADE_REM *
      (Number.parseFloat(getComputedStyle(document.documentElement).fontSize) ||
        16)
    const tabBox = tab.getBoundingClientRect()
    const rowBox = row.getBoundingClientRect()
    if (tabBox.left < rowBox.left + fade)
      row.scrollLeft -= rowBox.left + fade - tabBox.left
    else if (tabBox.right > rowBox.right - fade)
      row.scrollLeft += tabBox.right - (rowBox.right - fade)
  }, [isActive])
  return (
    <button
      ref={ref}
      type="button"
      aria-pressed={isActive}
      aria-disabled={isDisabled || undefined}
      aria-describedby={describedBy || undefined}
      onClick={isDisabled ? undefined : onPress}
      className={tabClassName(isActive)}
    >
      {children}
    </button>
  )
}

/**
 * The row a set of `Tab`s sits in.
 *
 * A plain `<div>` with no role, deliberately. `role="group"` needs a `<fieldset>`
 * to be valid and a fieldset drags form semantics and a `<legend>` in with it,
 * and `role="tablist"` would promise a roving-focus contract these do not
 * implement. Each tab is a button with its own name and `aria-pressed`, which is
 * what a screen reader needs; the row is only spacing.
 */
export function TabRow({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  // A row that scrolls has to say so: the edge with more beyond it fades out,
  // and an edge with nothing past it stays sharp, so a row that fits is drawn
  // as it always was.
  useEffect(() => {
    const row = ref.current
    if (!row) return
    const sync = () => {
      const start = row.scrollLeft > 1
      const end = row.scrollLeft + row.clientWidth < row.scrollWidth - 1
      row.dataset.fade =
        start && end ? "both" : start ? "start" : end ? "end" : "none"
    }
    sync()
    row.addEventListener("scroll", sync, { passive: true })
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(sync)
    observer?.observe(row)
    return () => {
      row.removeEventListener("scroll", sync)
      observer?.disconnect()
    }
  }, [])
  return (
    <div
      ref={ref}
      className="inline-flex max-w-full items-center gap-1 overflow-x-auto overscroll-x-contain scroll-px-12 data-[fade=both]:[mask-image:linear-gradient(to_right,transparent,black_3rem,black_calc(100%-3rem),transparent)] data-[fade=end]:[mask-image:linear-gradient(to_right,black_calc(100%-3rem),transparent)] data-[fade=start]:[mask-image:linear-gradient(to_left,black_calc(100%-3rem),transparent)]"
    >
      {children}
    </div>
  )
}
