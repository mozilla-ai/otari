import { useId } from "react"

import type { NavView } from "./NavHeader"

const OPTIONS: { value: NavView; label: string }[] = [
  { value: "tree", label: "Tree" },
  { value: "timeline", label: "Timeline" },
]

// Tree / Timeline as a flat track with a filled thumb, quieter than a bordered
// segmented control in a header this dense. Native radios, so arrow keys and a
// single tab stop come with them.
export function TraceViewSwitch({
  value,
  onChange,
}: {
  value: NavView
  onChange: (view: NavView) => void
}) {
  const name = useId()
  return (
    <div
      role="radiogroup"
      aria-label="Trace view"
      className="inline-flex items-stretch bg-[var(--trace-switch-track)] p-0.5 md:h-7"
    >
      {OPTIONS.map((option) => {
        const selected = option.value === value
        return (
          <label
            key={option.value}
            className={`flex min-h-11 cursor-pointer items-center px-2 text-caption whitespace-nowrap transition-colors duration-100 has-[:focus-visible]:otari-focus-ring motion-reduce:transition-none md:min-h-0 ${
              selected
                ? "bg-[var(--trace-switch-thumb)] text-foreground"
                : "text-subtle hover:text-foreground"
            }`}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={selected}
              onChange={() => onChange(option.value)}
              className="sr-only"
            />
            {option.label}
          </label>
        )
      })}
    </div>
  )
}
