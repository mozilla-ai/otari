import { Tab, TabRow } from "@/design-system/navigation/TabRow"

const MODE_VALUES = ["block", "monitor"] as const

/** A two-value mode switch. The codebase has no Select component and four
 *  hand-rolled `aria-pressed` groups, so this follows that pattern rather than
 *  introducing a fifth idiom. */
export function ModeToggle({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint?: string
  value: "block" | "monitor"
  onChange: (value: "block" | "monitor") => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-body">{label}</span>
      <TabRow>
        {MODE_VALUES.map((mode) => (
          <Tab
            key={mode}
            isActive={value === mode}
            onPress={() => onChange(mode)}
          >
            {mode}
          </Tab>
        ))}
      </TabRow>
      {hint === undefined ? null : <span className="text-caption">{hint}</span>}
    </div>
  )
}
