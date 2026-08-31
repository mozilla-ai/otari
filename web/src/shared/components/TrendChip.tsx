import { FiArrowDown, FiArrowUp } from "react-icons/fi"

import { formatPct } from "@/shared/helpers/format"

// Period-over-period change, as an arrow and a number on the page ground.
//
// It was a HeroUI `Chip` with a tinted fill and a rounded shape until the
// divided-surface direction removed both from the product: a pill is a small
// card, and a tint on a flat plane reads as a raised object. The name is kept
// because every call site says `TrendChip` and it is still the same idea; only
// the drawing changed.
//
// The color rule is deliberately asymmetric and is not "green for good, red for
// bad". A delta that is bad for its metric paints `danger`; a delta that is
// good, and one whose metric has no good direction, both paint `muted`. Good
// news does not need a color, because the number beside it is already the news;
// what a color buys is catching an eye that was not looking, and only one of
// the two directions is worth that.
//
// What this adds over a bare span is the part a generic one cannot know: which
// way the arrow points, which direction is the *good* one for the metric, and
// tabular figures that keep a column of these from changing width as it ticks.

// Which way is up for the metric being described. Spend, latency, and error rate
// improve by falling, so a rise on those reads as danger even though it is a
// rise. The default is `neutral` rather than a guess: most of this dashboard's
// numbers are cost and error rate, and a forgotten prop that painted rising
// spend green would be worse than no color at all.
export type TrendPolarity = "up-is-good" | "down-is-good" | "neutral"

export type TrendSize = "sm" | "md" | "lg"

export type TrendDirection = "up" | "down" | "flat"

export type TrendState = {
  direction: TrendDirection
  // Kept as three values even though only `danger` is drawn in color, because
  // this is what the metric's polarity *means* and it is what `announce` reads
  // to say "better" or "worse". Collapsing it to what is painted would make the
  // screen-reader phrase depend on a styling decision.
  color: "success" | "danger" | "default"
}

// Sized to the text beside it rather than to the chip's box, which is what keeps
// the arrow reading as a glyph in the line and not as an icon parked next to it.
const ARROW_SIZE: Record<TrendSize, string> = {
  sm: "size-3",
  md: "size-3.5",
  lg: "size-4",
}

// `formatPct` prints one decimal place of a percent, so anything under half of
// one of those rounds to "0.0%". Calling that flat rather than a direction is
// what stops a chip from claiming a fall while reading "-0.0%".
const FLAT_THRESHOLD = 0.0005

const DIRECTION_WORD: Record<TrendDirection, string> = {
  up: "up",
  down: "down",
  flat: "no change",
}

const JUDGMENT_WORD = { success: "better", danger: "worse" } as const

// What the chip says to a reader who sees neither the arrow nor the hue.
// Direction alone will not do it: the sign already carries direction, while
// `polarity` puts good and bad in color, so a fall that is an improvement and a
// fall that is a regression would otherwise announce identically. The judgment
// is appended only where there is one to make, which is why a `neutral` metric
// still says just the direction.
function announce({ direction, color }: TrendState): string {
  if (color === "default") return DIRECTION_WORD[direction]
  return `${DIRECTION_WORD[direction]}, ${JUDGMENT_WORD[color]}`
}

/**
 * Which way the number went, and whether that is good for this metric. Exported
 * so the mapping can be tested as a function rather than through the rendered
 * chip's classes.
 */
export function trendState(
  fraction: number,
  polarity: TrendPolarity,
): TrendState {
  const direction: TrendDirection =
    Number.isNaN(fraction) || Math.abs(fraction) < FLAT_THRESHOLD
      ? "flat"
      : fraction > 0
        ? "up"
        : "down"
  if (direction === "flat" || polarity === "neutral") {
    return { direction, color: "default" }
  }
  const good = polarity === "up-is-good" ? "up" : "down"
  return { direction, color: direction === good ? "success" : "danger" }
}

export function TrendChip({
  // null (or undefined) when there is nothing to compare against, the shape
  // `deltaFraction` returns for an unbounded range or a previous value of zero.
  // Nothing renders, so a call site can hand the value straight over.
  fraction,
  polarity = "neutral",
  size = "sm",
  // Replaces the formatted percentage when the change reads better as an
  // absolute ("+$1,234"). `fraction` still decides the arrow, the color and the
  // announced phrase, so it has to be the same change this string describes: a
  // fraction inside the flat threshold announces "no change" whatever the string
  // beside it says, so agreeing on the sign alone is not enough. Named `text`
  // rather than `value` because `value` on a component means the datum it is
  // about, and this is only what gets printed.
  text,
  // Trailing context inside the chip, e.g. "vs last month". Part of the chip's
  // text rather than a tooltip, so it is read out with the number. A string and
  // not a node: it shares one line inside a pill, which arbitrary JSX would
  // break.
  caption,
  // Layout and position at the call site (`ml-auto` in a value row,
  // `justify-self-end` in a grid cell), which is what className is for here. Not
  // for restyling: the ink comes from the metric's own polarity.
  className,
}: {
  fraction: number | null | undefined
  polarity?: TrendPolarity
  size?: TrendSize
  text?: string
  caption?: string
  className?: string
}) {
  if (fraction == null) return null
  const state = trendState(fraction, polarity)
  const { direction, color } = state
  const Arrow = direction === "up" ? FiArrowUp : FiArrowDown
  // A flat chip prints a bare zero: the sign of a change too small to render is
  // noise ("-0.0%"), and it would contradict the missing arrow beside it.
  const printed =
    text ??
    `${direction === "up" ? "+" : ""}${formatPct(direction === "flat" ? 0 : fraction)}`
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs ${
        color === "danger" ? "text-danger" : "text-muted"
      }${className ? ` ${className}` : ""}`}
    >
      {direction === "flat" ? null : (
        <Arrow className={`${ARROW_SIZE[size]} shrink-0`} aria-hidden="true" />
      )}
      {/* The arrow is decoration a screen reader never sees, and the judgment
          lives in hue alone for a sighted reader: a -2.1% that is an
          improvement and a -2.1% that is a regression print the same text and
          draw the same arrow. So `announce` says the judgment as well as the
          direction. It sits outside the visible span so it does not join that
          text, and `sr-only` is rendered (clipped) rather than hidden, so
          `select-none` keeps it out of a drag selection, as in ModelsPage. */}
      <span className="sr-only select-none">{announce(state)}</span>
      <span className="tabular-nums">
        {caption ? `${printed} ${caption}` : printed}
      </span>
    </span>
  )
}
