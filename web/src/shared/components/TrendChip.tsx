import { Chip } from "@heroui/react"
import { FiArrowDown, FiArrowUp } from "react-icons/fi"

import { formatPct } from "@/shared/helpers/format"

// Period-over-period change as a pill, from the "Trend chips" design canvas.
//
// It is built on HeroUI's own Chip rather than a hand-rolled <span>: v3 already
// ships the shape, the four fill treatments (primary/secondary/tertiary/soft),
// and the three sizes the canvas draws. `globals.css` aliases the status bases
// the chip's CSS reads (`--success`, `--danger`, `--default`) onto our tokens;
// the `-soft` fills and inks on top of those are HeroUI's own derivations from
// `themes/default/variables.css`, so a HeroUI bump can move them without a
// token change here. Either way nothing below names a color, which is what
// makes this theme-aware for free. What this adds is the part a generic chip
// cannot know: which way the arrow points, which direction is the *good* one
// for the metric, and tabular figures that keep a column of chips from
// changing width as it ticks.

// Which way is up for the metric being described. Spend, latency, and error rate
// improve by falling, so a rise on those reads as danger even though it is a
// rise. The default is `neutral` rather than a guess: most of this dashboard's
// numbers are cost and error rate, and a forgotten prop that painted rising
// spend green would be worse than no color at all.
export type TrendPolarity = "up-is-good" | "down-is-good" | "neutral"

// The canvas' four treatments, loudest first: `primary` is a solid status fill,
// `secondary` status ink on the neutral chrome fill, `tertiary` ink alone, and
// `soft` status ink on a tint of itself. `soft` is the default because a trend
// is a supporting detail beside a headline number, not the headline.
export type TrendVariant = "primary" | "secondary" | "tertiary" | "soft"

export type TrendSize = "sm" | "md" | "lg"

export type TrendDirection = "up" | "down" | "flat"

export type TrendState = {
  direction: TrendDirection
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

const DIRECTION_LABEL: Record<TrendDirection, string> = {
  up: "up",
  down: "down",
  flat: "no change",
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
  variant = "soft",
  size = "sm",
  // Replaces the formatted percentage when the change reads better as an
  // absolute ("+$1,234"). `fraction` still decides the arrow and the color, so
  // pass the signed value even when its text is not what is shown. Named `text`
  // rather than `value` because `value` on a component means the datum it is
  // about, and this is only what gets printed.
  text,
  // Trailing context inside the chip, e.g. "vs last month". Part of the chip's
  // text rather than a tooltip, so it is read out with the number. A string and
  // not a node: it shares one line inside a pill, which arbitrary JSX would
  // break.
  caption,
  // Layout and position at the call site (`ml-auto` in a tile's value row,
  // `justify-self-end` in a grid cell), which is what className is for here. Not
  // for restyling the chip: the fill and the ink come from `variant` and the
  // metric's own polarity.
  className,
}: {
  fraction: number | null | undefined
  polarity?: TrendPolarity
  variant?: TrendVariant
  size?: TrendSize
  text?: string
  caption?: string
  className?: string
}) {
  if (fraction == null) return null
  const { direction, color } = trendState(fraction, polarity)
  const Arrow = direction === "up" ? FiArrowUp : FiArrowDown
  // A flat chip prints a bare zero: the sign of a change too small to render is
  // noise ("-0.0%"), and it would contradict the missing arrow beside it.
  const printed =
    text ??
    `${direction === "up" ? "+" : ""}${formatPct(direction === "flat" ? 0 : fraction)}`
  return (
    <Chip variant={variant} color={color} size={size} className={className}>
      {direction === "flat" ? null : (
        <Arrow className={`${ARROW_SIZE[size]} shrink-0`} aria-hidden="true" />
      )}
      {/* The arrow is decoration a screen reader never sees, and `polarity`
          puts the good/bad distinction in hue alone, which no reader of the text
          can recover: a -2.1% that is good and a -2.1% that is bad differ only
          in green vs red. So direction is also said in a word. It sits outside
          `Chip.Label` so it does not join the visible label's text, and
          `sr-only` is rendered (clipped) rather than hidden, so `select-none`
          keeps it out of a drag selection over the chip, as in ModelsPage. */}
      <span className="sr-only select-none">{DIRECTION_LABEL[direction]}</span>
      <Chip.Label className="tabular-nums">
        {caption ? `${printed} ${caption}` : printed}
      </Chip.Label>
    </Chip>
  )
}
