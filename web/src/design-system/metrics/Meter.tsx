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
      className="block h-[0.1875rem] w-[8.75rem] bg-surface-subtle"
    >
      <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
    </span>
  )
}
