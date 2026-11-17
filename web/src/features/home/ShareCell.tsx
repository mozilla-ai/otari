import { wholePct } from "@/features/spend/limits"

/** A row's share of the table's spend, or "—" while the total is unknown. */
export function ShareCell({ share }: { share: number | undefined }) {
  return (
    <span className="flex items-center gap-3">
      {/* Fixed length: a share bar is a scale rows are compared against, so it
          does not stretch with its column. */}
      <span className="block h-[0.1875rem] w-[6.25rem] shrink-0 bg-surface-subtle">
        <span
          className="block h-full bg-accent"
          style={{ width: `${Math.min(100, (share ?? 0) * 100)}%` }}
        />
      </span>
      <span className="text-mono-caption text-muted">
        {share === undefined ? "—" : wholePct(share)}
      </span>
    </span>
  )
}
