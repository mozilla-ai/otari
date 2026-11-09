import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"

/**
 * Tool names to tick, from what the server's last check listed, plus any
 * pattern typed in (`create_*`). A ticked entry the server did not list stays
 * on the list, so a pattern or a name from an older check can be taken off.
 *
 * Controlled: `selected` is the patterns in order, and every change hands back
 * the whole new list.
 */
export function ToolChecklist({
  label,
  tools,
  selected,
  onChange,
  inputLabel = "Tool name or pattern",
  isDisabled = false,
}: {
  /** Names the list for assistive technology; the page's own heading shows it. */
  label: string
  tools: readonly string[]
  selected: readonly string[]
  onChange: (next: string[]) => void
  /**
   * The box for an entry the check did not list. A rule takes `prefix*`
   * patterns; the server's own allow-list takes names only, and says so here.
   */
  inputLabel?: string
  isDisabled?: boolean
}) {
  const [typed, setTyped] = useState("")
  const extra = selected.filter((name) => !tools.includes(name))
  const rows = [...tools, ...extra]
  const toggle = (name: string, isSelected: boolean) =>
    onChange(
      isSelected
        ? [...selected, name]
        : selected.filter((entry) => entry !== name),
    )
  const pattern = typed.trim()
  const add = () => {
    if (pattern === "" || selected.includes(pattern)) return
    onChange([...selected, pattern])
    setTyped("")
  }

  return (
    <div className="flex flex-col gap-2">
      {rows.length === 0 ? (
        <p className="text-caption">
          The server's last check listed no tools. Type a name below.
        </p>
      ) : (
        <ul
          aria-label={label}
          className="flex max-h-64 flex-col gap-1 overflow-y-auto"
        >
          {rows.map((name) => (
            <li key={name} className="min-w-0">
              <Checkbox
                isSelected={selected.includes(name)}
                onChange={(next) => toggle(name, next)}
                isDisabled={isDisabled}
                className="min-h-11 md:min-h-0"
              >
                <span className="font-mono text-sm break-all">{name}</span>
              </Checkbox>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1">
          <Field
            label={inputLabel}
            value={typed}
            onChange={setTyped}
            placeholder={inputLabel.includes("pattern") ? "create_*" : ""}
            isDisabled={isDisabled}
            shouldReserveMessage={false}
          />
        </div>
        <Button
          onPress={add}
          isDisabled={isDisabled || pattern === ""}
          className="min-h-11 md:min-h-0"
        >
          Add
        </Button>
      </div>
    </div>
  )
}
