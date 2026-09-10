import type { ReactNode } from "react"

/**
 * A short, closed-vocabulary label on a fill: a state, a scope, a plan.
 *
 * Its own element rather than HeroUI's `Chip`, for the reason `colors.md`
 * spells out about text on a tint. HeroUI's chip pairs a status color with its
 * own tint and its own ink, and two of those pairings measure under AA here:
 * the accent on its own subtle fill is 3.8:1, which is why brand text on the
 * brand tint takes `text-primary-subtle-foreground` instead. Wrapping the
 * library component would mean overriding its ink per tone, which is a rule
 * into vendor DOM for every tone; naming the pairs here is the same decision
 * made once, in the open.
 *
 * A chip is not a status mark. Where the thing being labeled is healthy or
 * broken, `Dot` and `SeverityMark` carry a word beside a square and read
 * correctly under a color deficiency. Reach for a chip when the label *is* the
 * value ("read-only", "Free plan", "every workspace") rather than a judgement
 * about it.
 */
export type ChipTone =
  | "neutral"
  | "accent"
  | "success"
  | "warning"
  | "danger"
  | "info"

/** Each tone as its own fill and the one ink that clears AA on that fill. */
const TONE: Record<ChipTone, string> = {
  neutral: "bg-surface-subtle text-muted",
  // Not `text-primary`: the accent on its own tint is 3.8:1. This is the token
  // colors.md exists to point at, and the single most missed line in that file.
  accent: "bg-primary-subtle text-primary-subtle-foreground",
  success: "bg-success-subtle text-success",
  warning: "bg-warning-subtle text-warning",
  danger: "bg-danger-subtle text-danger",
  info: "bg-info-subtle text-info",
}

const SIZE: Record<"sm" | "md", string> = {
  sm: "px-1.5 py-0.5 text-xs",
  md: "px-2 py-1 text-xs",
}

export function Chip({
  tone = "neutral",
  size = "sm",
  className = "",
  children,
}: {
  tone?: ChipTone
  size?: "sm" | "md"
  /** Position at the call site (`ml-auto`). Not for a fill of its own. */
  className?: string
  children: ReactNode
}) {
  return (
    <span
      className={`inline-flex w-fit shrink-0 items-center gap-1 ${SIZE[size]} ${TONE[tone]} ${className}`}
    >
      {children}
    </span>
  )
}
