import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import type { AccessMode } from "@/features/tools/mcpAccess"
import { ToolChecklist } from "@/features/tools/ToolChecklist"

function options(readTools: readonly string[]): RadioOption[] {
  return [
    { value: "all", label: "Allow any tool" },
    {
      value: "read",
      label: "Read tools only",
      description:
        readTools.length > 0
          ? `Allows the ${readTools.length} ${readTools.length === 1 ? "tool" : "tools"} labeled read on this server.`
          : "Allows nothing yet: no tool is labeled read on this server.",
    },
    { value: "some", label: "Allow specific tools…" },
    { value: "none", label: "Deny" },
  ]
}

/**
 * One rule's action as a choice: allow any tool, allow the tools labeled read,
 * allow the ones ticked, or deny.
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
  descriptions,
  isKnown,
  readTools,
  isDisabled = false,
  onCheck,
  isChecking,
}: {
  label: string
  mode: AccessMode
  picked: readonly string[]
  onChange: (mode: AccessMode, picked: string[]) => void
  /** What the server's last check listed, to tick from. */
  tools: readonly string[]
  descriptions?: Readonly<Record<string, string>>
  /** Whether a passing check listed them; typed names only until one has. */
  isKnown: boolean
  /** Runs the server's check, offered while its tools are not known. */
  onCheck?: () => void
  isChecking?: boolean
  /** The tools labeled read on the server, which "read" stands for. */
  readTools: readonly string[]
  isDisabled?: boolean
}) {
  return (
    <div className="flex flex-col gap-3">
      <RadioGroup
        label={label}
        value={mode}
        onChange={(next) => onChange(next as AccessMode, [...picked])}
        options={options(readTools)}
        isDisabled={isDisabled}
      />
      {mode === "some" ? (
        <div className="pl-6">
          <ToolChecklist
            label={`${label}: tools matched`}
            tools={tools}
            descriptions={descriptions}
            isKnown={isKnown}
            onCheck={onCheck}
            isChecking={isChecking}
            selected={picked}
            onChange={(next) => onChange("some", next)}
            isDisabled={isDisabled}
          />
        </div>
      ) : null}
    </div>
  )
}
