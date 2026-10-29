import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import type { AccessMode } from "@/features/tools/mcpAccess"
import { ToolChecklist } from "@/features/tools/ToolChecklist"

const OPTIONS: RadioOption[] = [
  { value: "all", label: "Allow any tool" },
  { value: "some", label: "Allow specific tools…" },
  { value: "none", label: "Deny" },
]

/**
 * One rule's action as a choice: allow any tool, allow the ones ticked, or deny.
 *
 * Controlled on the mode and the ticked tools rather than on the rule's
 * patterns, because "some" with nothing ticked yet is `[]` on the wire, which
 * reads back as Deny. `toolsFor` turns the pair into the patterns to send.
 */
export function AccessChooser({
  label,
  mode,
  picked,
  onChange,
  tools,
  isDisabled = false,
}: {
  label: string
  mode: AccessMode
  picked: readonly string[]
  onChange: (mode: AccessMode, picked: string[]) => void
  /** What the server's last check listed, to tick from. */
  tools: readonly string[]
  isDisabled?: boolean
}) {
  return (
    <div className="flex flex-col gap-3">
      <RadioGroup
        label={label}
        value={mode}
        onChange={(next) => onChange(next as AccessMode, [...picked])}
        options={OPTIONS}
        isDisabled={isDisabled}
      />
      {mode === "some" ? (
        <div className="pl-6">
          <ToolChecklist
            label={`${label}: tools matched`}
            tools={tools}
            selected={picked}
            onChange={(next) => onChange("some", next)}
            isDisabled={isDisabled}
          />
        </div>
      ) : null}
    </div>
  )
}
