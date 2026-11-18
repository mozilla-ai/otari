/**
 * The celebration's colors, read off the running theme: the two brand steps
 * and the one green. Read rather than written here, because a hex at a call
 * site cannot follow the theme, and the token rule is enforced over this file.
 */
const TOKENS = [
  "--color-primary",
  "--color-primary-hover",
  "--color-success",
  "--color-chart-ramp-4",
]

export function celebrationColors(): string[] {
  if (typeof document === "undefined") return []
  const styles = getComputedStyle(document.documentElement)
  return TOKENS.map((token) => styles.getPropertyValue(token).trim()).filter(
    (value) => value !== "",
  )
}
