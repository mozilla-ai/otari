import {
  type HTMLAttributes,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useRef,
  useState,
} from "react"
import { FiX } from "react-icons/fi"

import { useConfirmationFocus } from "@/shared/hooks/useConfirmationFocus"

/**
 * The shared vocabulary of the divided surface: the full-bleed rule, the square
 * status dot, the KPI strip and its cells, the share meter, and the tab row.
 *
 * Here rather than in a feature because the second page to want them was the
 * proof that they are the system rather than one screen's layout, and a copy on
 * each page is how two pages come to disagree about what a KPI cell is.
 */

/**
 * A band of the page: rules that run the full width of the scroll area, with
 * the content inside them still in the centered column.
 *
 * Two elements, and that is the point. A section that is both full-width and
 * centered cannot be one element, and the earlier single-class version only
 * escaped `<main>`'s padding, so on a wide viewport every rule stopped at the
 * column's edge and the page read as a stack of cards again. `.otari-bleed`
 * does the escaping (see globals.css for why it is container units and not
 * `100vw`); the inner element restores the column.
 *
 * `className` styles the band: its rules, its vertical padding, its own layout
 * if the content is a single row. `contentClassName` styles the column inside.
 *
 * `bleed={false}` for a band nested inside a column rather than sitting
 * directly in the scroll area. The escape is `100cqw` against `<main>`, so a
 * nested band does not stop at its column: measured inside a 360px grid cell it
 * came out 1464px wide and 552px past the right edge of the page, painting over
 * whatever shared the row. A nested band's rules run its own container instead,
 * and its content aligns to that container's edges.
 */
export function Section({
  className = "",
  contentClassName = "",
  bleed = true,
  children,
  ...rest
}: {
  className?: string
  contentClassName?: string
  bleed?: boolean
  children: ReactNode
} & Omit<HTMLAttributes<HTMLElement>, "className" | "children">) {
  return (
    <section
      className={bleed ? `otari-bleed ${className}` : className}
      {...rest}
    >
      <div
        className={
          bleed
            ? `mx-auto w-full max-w-[1800px] px-4 md:px-6 ${contentClassName}`
            : `w-full ${contentClassName}`
        }
      >
        {children}
      </div>
    </section>
  )
}

/**
 * The escape half on its own, for a band that is not a `<section>`: a header
 * row, a page-level notice. Pair it with `BLEED_INSET` on an inner element.
 */
export const FULL_BLEED = "otari-bleed"
/** The column a full-bleed band restores inside itself. */
export const BLEED_INSET = "mx-auto w-full max-w-[1800px] px-4 md:px-6"

/**
 * A page's opening: its title, the paragraph under it, and the one action that
 * belongs beside rather than below them.
 *
 * Shared because it was already written eight times. Every torn-down page spells
 * the same header by hand, down to the same arbitrary type values, so the type
 * of a page title is currently a string that has to be kept in step across eight
 * files by hand and will not be.
 *
 * The type is `text-display`, and the scale is what moved to meet it. Those
 * eight had each spelled 28/34 at semibold as literal utilities, which is the
 * size the direction draws a page title at and which rendered at weight 550
 * rather than the 600 it asked for, because the weight axis has two values and
 * 600 is not one of them. So the step became 28/34 at the semibold token and
 * every consumer converged on it, rather than nine call sites carrying an
 * arbitrary value that quietly disagreed with the scale.
 *
 * Those literals are described here rather than quoted, and that is not
 * fussiness: `foundation.test.ts`'s whole-tree rule matches `text-[` followed
 * by a digit against raw file contents and, unlike its sibling sweeps in the
 * same file, does not strip comments first. A docstring quoting the spelling it
 * replaced keeps this file on the offender list after every call site in it is
 * fixed, with nothing visibly wrong to find.
 *
 * `pb-5` rather than a gap on the parent, because a page is a stack of bands
 * that set their own rules and spacing, and a column gap would add air above
 * the first rule as well.
 */
export function PageIntro({
  title,
  action,
  descriptionClassName = "",
  children,
}: {
  title: string
  action?: ReactNode
  /**
   * Overrides the description's measure. One caller uses it: the guide, whose
   * own prose is 560px, so the paragraph introducing it should not be the
   * widest line on a page about measure.
   */
  descriptionClassName?: string
  children?: ReactNode
}) {
  return (
    <header className="flex flex-col gap-4 pb-5 sm:flex-row sm:items-start sm:justify-between">
      <div className="max-w-[620px]">
        <h1 className="text-display">{title}</h1>
        {children ? (
          <p className={`mt-1 text-sm text-muted ${descriptionClassName}`}>
            {children}
          </p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </header>
  )
}

/**
 * A settings list: a heading between rules, then its rows on the page ground
 * divided by row separators.
 *
 * Four blocks were spelling this by hand and had already drifted: two used an
 * 18px heading and two a 16px one, for groups of the same rank on the same
 * page. Shared so the rank of a group is decided once.
 *
 * Two bands rather than one, which is what puts the heading *between* rules
 * rather than above them: the first carries the rule over the heading, the
 * second the rule under it and the rule closing the last row.
 */
export function SettingsGroup({
  title,
  count,
  description,
  children,
}: {
  /**
   * Omitted where the page's own title already names the group, which happens
   * when a page is a filtered view of one service. The rows band keeps its
   * rules either way; what goes is the heading band above it.
   */
  title?: string
  /** Shown beside the title where a group's size is worth knowing up front. */
  count?: number
  /**
   * What the group is, under its heading and inside the same band. Capped to a
   * readable measure, because a band spans the page and a sentence should not.
   * A node rather than a string so a caller can put a link, or the group's own
   * error banner, in the same place.
   */
  description?: ReactNode
  children: ReactNode
}) {
  return (
    <>
      {title === undefined && description === undefined ? null : (
        <Section
          className="border-t border-border pt-6 pb-3"
          contentClassName="flex flex-col gap-2"
        >
          {title === undefined ? null : (
            <h2 className="text-title">
              {title}
              {count === undefined ? null : (
                <span className="font-normal text-subtle"> ({count})</span>
              )}
            </h2>
          )}
          {description ? (
            <div className="max-w-prose text-sm text-muted">{description}</div>
          ) : null}
        </Section>
      )}
      {/* `border-subtle` between the rows, `border` around the group. The two
          tiers are the structure: a section rule divides the page, a row
          separator divides repeated things inside one section, and using the
          section tier for both flattens the hierarchy into one weight. This is
          the third place that mis-assignment has been found, so it is fixed
          here rather than at a call site: every settings list in the app is
          this component now, and none of them names a tier. */}
      <Section
        className="border-y border-border"
        contentClassName="flex flex-col divide-y divide-border-subtle"
      >
        {children}
      </Section>
    </>
  )
}

/**
 * "There is nothing here", in one treatment.
 *
 * 14px muted, which is the table's, and it is the table's because that is the
 * one the user picked when two of these turned up on the same screen at
 * different sizes: a chart band's message at 12px above a table's at 14px, in a
 * product where the two mean the same thing. Shared so the size cannot drift
 * again, the same reason `SettingsGroup` is shared.
 *
 * `minHeight` because an empty chart band still has to hold the space its chart
 * would have taken, or the page reflows the moment data arrives.
 */
export function EmptyMessage({
  children,
  minHeight,
}: {
  children: ReactNode
  /** A CSS length, for the bands that must not collapse. */
  minHeight?: string
}) {
  return (
    <div
      className="flex items-center justify-center px-4 py-10 text-center text-sm text-muted"
      style={minHeight ? { minHeight, paddingBlock: 0 } : undefined}
    >
      {children}
    </div>
  )
}

/**
 * A row of filter controls above a table.
 *
 * It exists to name a *place*, which is what keeps field height from being a
 * per-site choice: a field is 40px everywhere, and the controls inside one of
 * these are 38px, because a filter sits on the table's own header row and a
 * 40px control outgrows it. A call site says "this row is a toolbar"; it never
 * says "this control is 38px".
 *
 * The height itself is in globals.css, on `.otari-toolbar`, since it has to
 * reach inside HeroUI's own DOM to find the select trigger.
 */
export function Toolbar({
  className = "",
  children,
}: {
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={`otari-toolbar flex flex-wrap items-center gap-2 ${className}`}
    >
      {children}
    </div>
  )
}

/**
 * A row's action: 13px of muted text, no border, no fill, no box.
 *
 * A row has three or four of these and boxing each one turned the last lane
 * into a control panel, louder than the row it acts on.
 *
 * `isDanger` is for the ARMED state and nothing else, which is worth spelling
 * out because it was got wrong at six sites on the first pass: a destructive
 * action at rest is the same muted text as its neighbors, and the ink arrives
 * only once the next click commits something. A row that paints Remove red
 * before anybody has touched it spends the colour on a state nothing is in, and
 * by the time it means something the reader has stopped seeing it. Same rule
 * the spend figure follows: the colour marks what you are about to do, never
 * the control that offers it.
 *
 * An action whose confirmation is a dialog rather than an inline arm stays
 * muted throughout: the dialog is where the danger lives.
 *
 * Shared once two pages had spelled it. Pair with `RowActionRow`, which sets the
 * 16px between them.
 */
export function RowAction({
  onPress,
  isDanger,
  isDisabled,
  ariaLabel,
  ref,
  children,
}: {
  onPress: () => void
  isDanger?: boolean
  isDisabled?: boolean
  /**
   * Forwarded to the button so a caller can move focus onto it. The two-step
   * confirm is the only caller: its swap unmounts a focused control, and a ref
   * is the only way to hand the caret to whatever replaced it.
   */
  ref?: RefObject<HTMLButtonElement | null>
  /**
   * Replaces the visible label for assistive tech, which is how a row action
   * says which row it acts on and why it is refused: a disabled control takes
   * no focus, so a tooltip reaches a pointer and nothing else, and the reason
   * has to be in the name.
   */
  ariaLabel?: string
  children: ReactNode
}) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={isDisabled}
      aria-label={ariaLabel}
      onClick={onPress}
      className={`text-caption whitespace-nowrap transition-colors motion-reduce:transition-none disabled:opacity-50 ${
        isDanger ? "text-danger" : "hover:text-foreground"
      }`}
    >
      {children}
    </button>
  )
}

/** The lane those sit in: right-aligned, 16px apart. */
export function RowActionRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center justify-end gap-4">{children}</div>
}

/**
 * A destructive row action that asks twice, in text rather than in buttons.
 *
 * Armed, the label becomes the confirmation in danger ink with a plain Cancel
 * beside it. It replaces `ConfirmButton` on a table row for the same reason the
 * other actions lost their boxes; `ConfirmButton` stays for the forms and cards
 * where a destructive control is the only control and a button is right.
 */
export function ConfirmRowAction({
  confirmLabel,
  onConfirm,
  isPending,
  children,
}: {
  confirmLabel: string
  onConfirm: () => void
  isPending?: boolean
  children: ReactNode
}) {
  const [armed, setArmed] = useState(false)
  // Cancelling unmounts the focused Cancel button, which has no counterpart at
  // rest, and focus lands on `<body>`: the way out of a destructive action
  // costs a keyboard user their place. Arming does not have that problem, but
  // only by accident, because React reuses the trigger's own node as Confirm;
  // the hook covers both so the accident stops being load-bearing.
  const { triggerRef, confirmRef } = useConfirmationFocus(armed)
  if (armed) {
    return (
      <>
        <RowAction
          ref={confirmRef}
          isDanger
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </RowAction>
        <RowAction isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </RowAction>
      </>
    )
  }
  return (
    <RowAction ref={triggerRef} onPress={() => setArmed(true)}>
      {children}
    </RowAction>
  )
}

/**
 * A value somebody chose, with the way to unchoose it.
 *
 * Identifier text with a remove control beside it, and no container: no border,
 * no fill, no shape of its own. Dismissability is a modifier on a kind rather
 * than a kind, so this wears the identifier kind and adds a control.
 *
 * It shipped as a bordered, filled chip first. That was the pill squared, and
 * squaring the pill was the wrong move on a pill that had been deleted: the
 * kinds here are told apart by dot, case and separator, not by boxes, and a
 * fourth boxed thing would have been the only one in the product. Corrected
 * before anything else grew a border to match it.
 *
 * The dismiss is a real 24px target holding a 12px glyph. It reads small and is
 * not, which is the point: the visible cross stays quiet while the thing a
 * finger lands on is the size a finger needs. That part came through the
 * retraction unchanged, as did the one shared component behind three call
 * sites.
 */
export function DismissChip({
  label,
  value,
  onDismiss,
  dismissLabel,
}: {
  /** The dimension, shown before the value and divided from it by a colon. */
  label?: string
  value: ReactNode
  onDismiss: () => void
  /** Names the target for assistive tech; falls back to the label and value. */
  dismissLabel?: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-mono-caption text-foreground">
      {label ? <span className="text-subtle">{label}:</span> : null}
      {value}
      <button
        type="button"
        onClick={onDismiss}
        aria-label={
          dismissLabel ?? `Remove ${label ? `${label} ` : ""}${value}`
        }
        className="inline-flex h-6 w-6 items-center justify-center text-muted hover:text-foreground"
      >
        <FiX aria-hidden="true" className="h-3 w-3" />
      </button>
    </span>
  )
}

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
    <Section
      // Auto rows in groups of four, one group per row of cells, so the cells
      // below can subgrid onto them and line their four parts up with each
      // other. Without it a label that wraps makes its own cell taller and
      // drops its value below the others'.
      className="border-y border-border"
      contentClassName="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5"
      // The graphic row is dropped uniformly in the empty state, so the strip
      // gets shorter without any cell changing shape relative to its neighbors.
      data-empty={empty ? "true" : undefined}
    >
      {children}
    </Section>
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
    // `minmax(0,1fr)` for the cell's own column, not the `auto` a grid gives
    // itself by default. The strip's tracks are already `minmax(0,1fr)`, so the
    // cell's box is its track; what was unbounded was the column INSIDE it,
    // which took its floor from the widest child. The subline never wraps, so
    // its min-content is the whole string: measured with a 59-character
    // subline, the inner column came out 356px inside a 283px cell and every
    // child sized to it, the sparkline included, painted 129px into the cell
    // beside it (218px at 1280, since the track shrinks and the string does
    // not). Bounding the column is what makes the truncation and the
    // sparkline's `w-full` mean anything.
    <div className="grid row-span-4 min-w-0 grid-rows-subgrid grid-cols-[minmax(0,1fr)] gap-1.5 border-border px-7 py-[18px] not-last:border-r">
      <span className="flex items-end text-overline">{label}</span>
      {/* 400, deliberately, where the rest of the page's emphasis is 550: at
          30px the size is already the hierarchy, and a heavier numeral here
          would out-weigh the page title above it. */}
      <span className="text-mono-figure font-normal text-foreground">
        {value}
      </span>
      {/* One line, always present, so no cell is shorter than its neighbors.
          A severity, a subline and a delta all share it where a cell has more
          than one: dropping any of them to make room would lose information the
          strip is there to carry, and a cell with a spare line would break the
          shared baseline the subgrid buys. */}
      {/* Most-judged to least, left to right: a severity, then the comparison,
          then the raw fact. It never wraps: a second line here made one cell
          taller than its siblings and silently undid the shared baseline the
          subgrid buys. The severity and the delta hold their width; the subline
          is the only thing that gives, and it truncates, because its facts also
          live in the breakdown tables while a delta lives nowhere else. */}
      <span className="flex min-h-[18px] items-center gap-2 text-xs text-nowrap text-muted">
        {severity ? <SeverityMark severity={severity} /> : null}
        {severity && delta ? <Separator /> : null}
        {delta}
        {(severity || delta) && subline ? <Separator /> : null}
        {subline ? (
          <span className="min-w-0 truncate" title={subline}>
            {subline}
          </span>
        ) : null}
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
    <span aria-hidden className="shrink-0 text-subtle">
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
      className={`flex shrink-0 items-center gap-2 font-mono ${
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

/** How far along its allocation a spend is, which decides how it is drawn. */
export type SpendState = "on-track" | "near-limit" | "over"

/**
 * The one place a spend is classified, so the bar and the figure beside it
 * cannot disagree about which of the three states this is.
 *
 * `nearLimitAt` is the share of the allocation past which spend stops being
 * unremarkable. It is a fraction of the limit and not an absolute, because
 * "nearly out" means the same thing on a $50 budget and a $5,000 one.
 */
export function spendState(
  spent: number,
  allocated: number,
  nearLimitAt = 0.8,
): SpendState {
  if (allocated <= 0) return "on-track"
  if (spent > allocated) return "over"
  return spent >= allocated * nearLimitAt ? "near-limit" : "on-track"
}

/**
 * A spend against its allocation, in three states rather than the two the code
 * carried before, where everything under the limit looked identical and the
 * first thing anyone learned was that they had already gone past.
 *
 * | state      | bar                                    | figure  |
 * | ---------- | -------------------------------------- | ------- |
 * | on-track   | accent to the spend                    | normal  |
 * | near-limit | accent to the threshold, danger beyond | normal  |
 * | over       | danger, full width                     | danger  |
 *
 * The middle state is two segments and not a second color for the whole bar,
 * which is what makes it legible without hue: the overshoot past the threshold
 * is a distinct block with a hairline of track showing between it and the
 * accent, so in grayscale, or to anyone who cannot separate teal from red, the
 * bar still says "there is a part of this that is past the mark". A single
 * recolored bar would say nothing at all under those conditions.
 *
 * The figure changing color is reserved for `over`, and it is the only number
 * anywhere in this product that changes color. That is the point: it has to be
 * worth something when it happens.
 */
export function SpendMeter({
  spent,
  allocated,
  ariaLabel,
  nearLimitAt = 0.8,
  className = "",
}: {
  spent: number
  allocated: number
  ariaLabel: string
  nearLimitAt?: number
  className?: string
}) {
  const state = spendState(spent, allocated, nearLimitAt)
  const share = allocated > 0 ? spent / allocated : 0
  const pct = Math.max(0, Math.min(1, share)) * 100
  const thresholdPct = nearLimitAt * 100
  return (
    <span
      role="progressbar"
      aria-label={ariaLabel}
      // Clamped, because a progressbar's value is documented to sit inside its
      // range and a widget that reports 137 out of 100 is malformed. The number
      // that matters is not lost: it goes in `aria-valuetext`, which is the
      // field for the human reading of a value, so a screen reader is told both
      // that the bar is full and that the spend is 37% past the limit. Neither
      // field is abused to carry the other's fact.
      aria-valuenow={Math.min(100, Math.round(share * 100))}
      aria-valuetext={
        state === "over"
          ? `${Math.round(share * 100)}% of limit — over budget`
          : `${Math.round(share * 100)}% of limit`
      }
      aria-valuemin={0}
      aria-valuemax={100}
      className={`flex h-[3px] w-full bg-surface-subtle ${className}`}
    >
      {state === "over" ? (
        <span className="block h-full w-full bg-danger" />
      ) : state === "near-limit" ? (
        <>
          <span
            className="block h-full bg-accent"
            style={{ width: `${thresholdPct}%` }}
          />
          {/* The hairline that makes the two segments read as two. It is track,
              not a border, so it cannot pick up a color of its own. */}
          <span className="block h-full w-px shrink-0" />
          <span
            className="block h-full bg-danger"
            style={{ width: `${Math.max(pct - thresholdPct, 0)}%` }}
          />
        </>
      ) : (
        <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
      )}
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
 * One choice out of a small closed set, as a bounded segmented control.
 *
 * A tab row and this are the same shape doing different jobs, which is why
 * they are two components: a tab CHANGES WHAT YOU SEE and is unbounded, while
 * a segment RECORDS A CHOICE the form will submit and is bounded, because the
 * track is what says "these are the alternatives and there are no others".
 *
 * It exists because the alternative was worse in a specific way. Rendering the
 * options as buttons and filling the chosen one primary puts the CTA's own
 * treatment on a value: it reads as an action to take rather than an option
 * chosen, and it sat beside a real submit button wearing exactly the same
 * fill.
 *
 * The selected segment is a surface step and foreground ink rather than a
 * fill, for the reason the whole surface is: a fill here would be the only
 * filled thing in a form that has none.
 *
 * Native radios rather than buttons with `role="radio"`: the semantics are
 * exactly a radio group's, and taking the real element brings the arrow-key
 * navigation and the one-tab-stop-per-group behavior with it instead of
 * reimplementing them. Each input is visually hidden and its label is the
 * segment, so the focus ring has to be drawn on the label.
 */
export function Segmented({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: { value: string; label: string }[]
  value: string
  onChange: (next: string) => void
}) {
  const name = useId()
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex w-fit max-w-full overflow-x-auto border border-control-border"
    >
      {options.map((option) => {
        const selected = option.value === value
        return (
          <label
            key={option.value}
            // Never shrinks and never wraps, the same rule a tab follows: a
            // segment that squeezed would put one control at two widths in one
            // row. The track scrolls instead.
            // The divider is a leading border on every segment but the first,
            // so the count of rules is always one less than the count of
            // segments, however many there are.
            className={`shrink-0 cursor-pointer border-l border-control-border px-3 py-[5px] text-sm whitespace-nowrap transition-colors first:border-l-0 has-[:focus-visible]:otari-focus-ring motion-reduce:transition-none ${
              selected
                ? "bg-surface-subtle text-foreground"
                : "text-muted hover:text-foreground"
            }`}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={selected}
              onChange={() => onChange(option.value)}
              className="sr-only"
            />
            {option.label}
          </label>
        )
      })}
    </div>
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

/**
 * Marks its subtree `data-scrolled` while the table inside it is scrolled off
 * its left edge, so the pinned lane can draw its boundary only then.
 *
 * A cue for a state disappears with the state: at rest a table has no internal
 * verticals, because its columns are not regions, which is exactly what
 * separates it from the KPI strip. The same principle the mid-column clip
 * follows.
 *
 * It reaches for `.table__scroll-container` because that element is HeroUI's
 * and no call site can put a listener on it any other way.
 */
export function TableScrollFrame({
  className,
  children,
}: {
  className: string
  children: ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const root = ref.current
    const scroller = root?.querySelector<HTMLElement>(
      ".table__scroll-container",
    )
    if (!root || !scroller) return
    const sync = () => {
      root.dataset.scrolled = scroller.scrollLeft > 0 ? "true" : "false"
    }
    sync()
    scroller.addEventListener("scroll", sync, { passive: true })
    return () => scroller.removeEventListener("scroll", sync)
  })
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  )
}
