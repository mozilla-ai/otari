import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { shareOf } from "@/features/rules/rules"
import { wholePct } from "@/features/spend/limits"

/** How much of a limit is gone: a short meter and the share beside it. */
export function UsedCell({
  spent,
  max,
  label,
  meterClassName = "w-14",
}: {
  spent: number
  max: number
  label: string
  /** The meter's width: 56px in the list, wider on a rule's own page. */
  meterClassName?: string
}) {
  return (
    <span className="flex items-center justify-end gap-2.5">
      <span className={`shrink-0 ${meterClassName}`}>
        <SpendMeter spent={spent} allocated={max} ariaLabel={label} />
      </span>
      <span className="w-9 shrink-0 text-right text-mono-caption text-foreground tabular-nums">
        {wholePct(shareOf(spent, max))}
      </span>
    </span>
  )
}
