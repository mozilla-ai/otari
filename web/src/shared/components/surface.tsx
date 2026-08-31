import type { ReactNode } from "react"

/**
 * The shared vocabulary of the divided surface: the full-bleed rule, the square
 * status dot, the KPI strip and its cells, the share meter, and the tab row.
 *
 * Here rather than in a feature because the second page to want them was the
 * proof that they are the system rather than one screen's layout, and a copy on
 * each page is how two pages come to disagree about what a KPI cell is.
 */

/**
 * Every section on this page breaks out of `<main>`'s column padding so its
 * rules reach the page edge. Kept as one constant rather than repeated, because
 * a section that forgets it does not look broken, it looks like a card.
 */
export const FULL_BLEED = "-mx-4 md:-mx-6"
/** The padding a full-bleed section puts back inside its own rules. */
export const BLEED_INSET = "px-4 md:px-6"

/** A 6px square. The page's one status mark, in every place it appears. */
export function Dot({ className }: { className: string }) {
  return <span aria-hidden className={`h-1.5 w-1.5 shrink-0 ${className}`} />
}

/**
 * Five equal cells divided by vertical rules, between horizontal ones. Equal
 * rather than content-sized so the divisions land on a rhythm rather than
 * wherever the longest label happens to end.
 */
export function KpiStrip({
  children,
  empty,
}: {
  children: ReactNode
  empty: boolean
}) {
  return (
    <section
      // Auto rows in groups of four, one group per row of cells, so the cells
      // below can subgrid onto them and line their four parts up with each
      // other. Without it a label that wraps makes its own cell taller and
      // drops its value below the others'.
      className={`${FULL_BLEED} ${BLEED_INSET} grid grid-cols-2 border-y border-border sm:grid-cols-3 xl:grid-cols-5`}
      // The graphic row is dropped uniformly in the empty state, so the strip
      // gets shorter without any cell changing shape relative to its neighbors.
      data-empty={empty ? "true" : undefined}
    >
      {children}
    </section>
  )
}

export function KpiCell({
  label,
  value,
  severity,
  subline,
  delta,
  graphic,
}: {
  label: string
  value: string
  severity?: Severity
  /**
   * Why the value is what it is. Carried in every page state and not only in
   * the empty one: an em dash with nothing under it makes the reader guess
   * whether the number is missing or zero, and the empty state is simply the
   * case where every cell has something to say.
   */
  subline?: string
  delta?: ReactNode
  graphic?: ReactNode
}) {
  return (
    // `grid-rows-subgrid` over `row-span-4` is what keeps the five values on one
    // baseline. The cells are equal-height grid items, so a wrapped label used
    // to make its own cell taller and push its value down while the other four
    // stayed put (measured with a 1180px column: value tops 241 against 223).
    // Subgridding the cell's four parts onto the strip's own rows lines the
    // labels, values, deltas and graphics up with their neighbors instead, and
    // reserves nothing when no label wraps, so it costs no dead space wide.
    // `items-end` on the label is what makes a one-line label sit on the same
    // baseline as the last line of a two-line one.
    <div className="grid row-span-4 grid-rows-subgrid gap-1.5 border-border px-7 py-[18px] not-last:border-r">
      <span className="flex items-end text-overline">{label}</span>
      {/* 400, deliberately, where the rest of the page's emphasis is 550: at
          30px the size is already the hierarchy, and a heavier numeral here
          would out-weigh the page title above it. */}
      <span className="font-mono text-[30px] leading-[36px] font-normal text-foreground tabular-nums">
        {value}
      </span>
      {/* One line, always present, so no cell is shorter than its neighbors.
          A severity, a subline and a delta all share it where a cell has more
          than one: dropping any of them to make room would lose information the
          strip is there to carry, and a cell with a spare line would break the
          shared baseline the subgrid buys. */}
      <span className="flex min-h-[18px] flex-wrap items-center gap-2 text-xs text-muted">
        {severity ? <SeverityMark severity={severity} /> : null}
        {severity && (subline || delta) ? <Separator /> : null}
        {subline ? <span>{subline}</span> : null}
        {subline && delta ? <Separator /> : null}
        {delta}
      </span>
      {/* Reserved rather than conditional: an absent sparkline must not make one
          cell shorter than the four beside it. Dropped in the empty state by
          the caller passing none to any cell, which shortens the whole strip. */}
      {graphic ? (
        <span className="flex h-10 items-center">{graphic}</span>
      ) : null}
    </div>
  )
}

/** The middot between two things sharing the line under a value. */
function Separator() {
  return (
    <span aria-hidden className="text-subtle">
      ·
    </span>
  )
}

export type Severity = { status: "ok" | "warn" | "alert"; word: string }

/**
 * A severity as a square dot plus its word, never as a color alone. `warn` and
 * `alert` share the danger dot and differ in their ink, because the dot answers
 * "is anything wrong here" and the word answers "how much".
 */
export function SeverityMark({ severity }: { severity: Severity }) {
  const { status, word } = severity
  return (
    <span
      className={`flex items-center gap-2 font-mono ${
        status === "alert"
          ? "text-danger"
          : status === "warn"
            ? "text-muted"
            : "text-subtle"
      }`}
    >
      <Dot className={status === "ok" ? "bg-success" : "bg-danger"} />
      {word.toUpperCase()}
    </span>
  )
}

/** The budget meter: a 140x3 track with an accent fill. */
export function Meter({
  fraction,
  ariaLabel,
}: {
  fraction: number
  ariaLabel: string
}) {
  const pct = Math.max(0, Math.min(1, fraction)) * 100
  return (
    <span
      role="img"
      aria-label={ariaLabel}
      className="block h-[3px] w-[140px] bg-surface-subtle"
    >
      <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
    </span>
  )
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
      className={`shrink-0 px-2.5 py-[5px] text-sm whitespace-nowrap transition-colors motion-reduce:transition-none ${
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
