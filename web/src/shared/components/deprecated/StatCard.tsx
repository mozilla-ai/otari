import { Card, Chip } from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"

// A tile's attention status, on the foundation's three status roles. Color is
// never the only signal: a status tile also carries a word/icon via `statusLabel`.
export type StatStatus = "ok" | "warn" | "alert"

// The accent bar is `!`-important because the design foundation gives every
// HeroUI Card a 1px outline (`.card:not(.card--transparent)` in globals.css),
// and that rule is unlayered while a Tailwind utility sits in @layer utilities;
// unlayered always wins, whatever the specificity. Without the bang the
// shorthand `border:` resets this tile's left edge back to a hairline.
//
// The pill beside the value is HeroUI's own Chip, for the reason TrendChip is:
// `globals.css` aliases the status bases the chip's CSS reads (`--success`,
// `--warning`, `--danger`) onto our tokens, so naming a status here is all it
// takes for both themes to follow the foundation. A hand-rolled <span> carrying
// its own border/bg/text triple per status would restate what the library
// already derives, and leave the status pill and the trend chip on the same
// tile as two different shapes.
const STAT_STATUS: Record<
  StatStatus,
  { accent: string; chip: "success" | "warning" | "danger" }
> = {
  ok: { accent: "border-l-success!", chip: "success" },
  warn: { accent: "border-l-warning!", chip: "warning" },
  alert: { accent: "border-l-danger!", chip: "danger" },
}

export function StatCard({
  label,
  value,
  hint,
  trend,
  status,
  statusLabel,
  chart,
  to,
}: {
  label: string
  value: ReactNode
  // Supporting context under the value: what the number is made of ("5.8%
  // errors", "311.2k read"), not how it moved. The movement is `trend`, and the
  // two share one row.
  hint?: ReactNode
  // Period-over-period change, as a <TrendChip>. It leads the aside row under
  // the value, ahead of `hint`: a pill sharing the value's baseline competes
  // with the number for the first glance, while a second row of its own spends
  // a line saying what belongs beside the hint anyway.
  trend?: ReactNode
  status?: StatStatus
  // A short word (and/or icon) shown as a pill beside the value. Required to be a
  // non-color signal so status is legible without hue (colorblind operators).
  statusLabel?: ReactNode
  // An optional trend visual (e.g. a <Sparkline>) rendered under the value/hint,
  // for KPI tiles that have a bucketed series on the wire.
  chart?: ReactNode
  // When set, the whole tile is a keyboard-focusable link to this route.
  to?: LinkProps["to"]
}) {
  const accent = status ? `border-l-4! ${STAT_STATUS[status].accent}` : ""
  const body = (
    // p-0 on the Card zeroes HeroUI's own card padding so it doesn't stack with
    // Card.Content's, which otherwise doubled the tile's height (most visible at
    // two-up on mobile). Content owns the padding: tighter on mobile, roomier up.
    <Card.Content className="flex flex-col gap-1 p-4 sm:p-5">
      <span className="text-overline">{label}</span>
      <span className="flex flex-wrap items-center gap-2">
        {/* text-xl (22px), deliberately a step *below* the page title's
            text-display (28px), and flat across breakpoints: rising to
            text-2xl at `sm` would make a number inside a card the largest text
            on the page, bigger than the name of the page itself.
            `tabular-nums` so a column of these aligns. */}
        <span className="text-xl font-semibold tabular-nums text-foreground">
          {value}
        </span>
        {status && statusLabel ? (
          // `soft` and `sm` are TrendChip's defaults too, so a tile carrying
          // both (the error-rate tile carries a status word and a delta) draws
          // one shape at one weight rather than two.
          <Chip variant="soft" color={STAT_STATUS[status].chip} size="sm">
            <Chip.Label>{statusLabel}</Chip.Label>
          </Chip>
        ) : null}
      </span>
      {/* One row under the value carrying both the movement (the chip) and the
          composition (the hint): they are two halves of the same aside, and
          stacked they read as two, costing a line the tile does not have.
          `flex` also makes the chip hug its text, which it does not do as a
          direct child of this column: HeroUI's Chip is inline-flex, and a
          stretched child would run the pill the full width of the tile.

          `flex-wrap` because at five-up the pair does not always fit; the wrap
          is the fallback, not the layout. `items-center` aligns the hint's
          x-height to the middle of the pill rather than to its box.

          The wrap is reserved for so it does not misalign a row of tiles: one
          tile wrapping while its neighbors stay on one line would move its
          sparkline up relative to theirs. min-h-10.5 is 42px, which is what the
          wrap costs: a 20px chip line, `gap-y-1`, and one 18px hint line. The
          chip's line is 20px and not the 18px its `text-xs` implies, because
          HeroUI's `.chip` sets `--tw-leading` to `leading-5` on the element and
          `.chip--sm` does not reset it, so the size modifier's
          `line-height: var(--tw-leading, var(--text-xs--line-height))` resolves
          to the base 20px rather than to our token. #807 reserved 36px here for
          two lines of plain text, which that overruns. A hint long enough to
          wrap on its own still overruns this, so it is a floor and not a
          guarantee; the type is deliberately not sized to the longest string,
          which would be sizing it by accident. Reserved for a charted tile even
          with neither chip nor hint, since the chart is what makes the
          misalignment visible; a tile with no chart and nothing to say reserves
          nothing, so a lone tile carries no dead space. */}
      {trend || hint || chart ? (
        <span className="flex min-h-10.5 flex-wrap items-center gap-x-2 gap-y-1 text-caption tabular-nums">
          {trend}
          {/* Its own element, not a bare text node beside the chip: the two
              are separate statements, and a node keeps the hint addressable
              (by a test, and by a reader's selection) rather than merged into
              the chip's text. */}
          {hint ? <span>{hint}</span> : null}
        </span>
      ) : null}
      {chart ? <div className="mt-2">{chart}</div> : null}
    </Card.Content>
  )
  // min-w-0 (not a fixed min) so the tile fits its grid track: with
  // grid-cols-2's minmax(0,1fr) columns, a larger min-width would overflow the
  // track and overlap the neighboring tile on narrow (mobile) viewports.
  const cardClass = `flex-1 min-w-0 p-0 ${accent}`
  if (to) {
    return (
      <Card
        // Same reason as the accent above: the foundation's card outline is
        // unlayered, so the hover tint needs the bang to be seen at all.
        className={`${cardClass} transition-colors hover:border-accent!`}
      >
        <Link to={to} className="block rounded-[inherit]">
          {body}
        </Link>
      </Card>
    )
  }
  return <Card className={cardClass}>{body}</Card>
}
