import { Checkbox } from "@/design-system/forms/Checkbox"

/** What a guardrail reads, as the API names it. */
export type GuardrailDirection =
  | "input"
  | "tool_results"
  | "tool_calls"
  | "output"

export const DIRECTION_OPTIONS: readonly {
  value: GuardrailDirection
  label: string
}[] = [
  { value: "input", label: "User input" },
  { value: "tool_results", label: "Tool results" },
  { value: "tool_calls", label: "Tool calls" },
  { value: "output", label: "Answers" },
]

/**
 * Which parts of a request a guardrail reads, one box each.
 *
 * At least one stays checked, since a guardrail that reads nothing never runs,
 * and the value keeps the options' order whatever order they were picked in.
 */
export function GuardrailReadsField({
  value,
  onChange,
  groupName = "What it checks",
}: {
  value: readonly GuardrailDirection[]
  onChange: (next: GuardrailDirection[]) => void
  /** The group's accessible name, for a page that renders one per guardrail. */
  groupName?: string
}) {
  return (
    <fieldset aria-label={groupName} className="flex flex-col gap-2">
      <legend className="text-body">What it checks</legend>
      <div className="flex flex-wrap gap-3">
        {DIRECTION_OPTIONS.map((option) => {
          const isSelected = value.includes(option.value)
          return (
            <span key={option.value} className="text-sm text-muted">
              <Checkbox
                isSelected={isSelected}
                isDisabled={isSelected && value.length === 1}
                onChange={(next) =>
                  onChange(
                    DIRECTION_OPTIONS.map((o) => o.value).filter((direction) =>
                      direction === option.value
                        ? next
                        : value.includes(direction),
                    ),
                  )
                }
              >
                {option.label}
              </Checkbox>
            </span>
          )
        })}
      </div>
    </fieldset>
  )
}
