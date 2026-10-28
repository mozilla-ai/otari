import type { ReactNode } from "react"

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
  return (
    <button
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
  return (
    <div className="inline-flex max-w-full items-center gap-1 overflow-x-auto">
      {children}
    </div>
  )
}
