import type { ReactNode } from "react"

/**
 * A tab in a row of them, which is what this product's segmented choices are
 * now: an active tab takes the `surface-subtle` fill and the foreground ink, an
 * inactive one is bare muted text. Square, and no accent anywhere, because the
 * accent is data ink and fills rather than a way to say "this one".
 *
 * A `<button>` with `aria-pressed` rather than a real tablist: these switch what
 * a panel shows without being a tab widget's roving-focus contract, and the
 * pages using them already announce the choice through the content beneath.
 */
export function Tab({
  isActive,
  onPress,
  children,
}: {
  isActive: boolean
  onPress: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={onPress}
      // A segment never shrinks and never wraps its label: a tab row that
      // squeezed would put the same control at two widths on one page, and a
      // wrapped label would break the row's height. The row scrolls instead.
      className={`shrink-0 px-2.5 py-[0.3125rem] text-sm whitespace-nowrap transition-colors motion-reduce:transition-none ${
        isActive
          ? "bg-surface-subtle text-foreground"
          : "text-muted hover:text-foreground"
      }`}
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
