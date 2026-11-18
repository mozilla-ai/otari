import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { Segmented } from "@/design-system/navigation/Segmented"
import { PERIODS, type PeriodKey } from "@/features/spend/period"

/**
 * Home's window. Six segments is past the four a `Segmented` is meant to
 * hold; the row is there so every window is one press away and the chosen one
 * is read at a glance, as analytics tools set it. Six do not fit a phone, so
 * there they are a select, which is what the design system gives a closed set
 * longer than a track.
 */
export function SpendPeriodPicker({
  value,
  onChange,
}: {
  value: PeriodKey
  onChange: (next: string) => void
}) {
  const isMobile = useIsMobile()
  return isMobile ? (
    <FilterSelect
      ariaLabel="Window"
      value={value}
      onChange={onChange}
      options={PERIODS}
    />
  ) : (
    <Segmented
      label="Window"
      size="md"
      options={PERIODS}
      value={value}
      onChange={onChange}
    />
  )
}
