import { useId } from "react"

/**
 * One choice out of a small closed set, as a bounded segmented control.
 *
 * A tab row and this are the same shape doing different jobs, which is why
 * they are two components: a tab CHANGES WHAT YOU SEE and is unbounded, while
 * a segment RECORDS A CHOICE the form will submit and is bounded, because the
 * track is what says "these are the alternatives and there are no others".
 *
 * It exists because the alternative was worse in a specific way. Rendering the
 * options as buttons and filling the chosen one primary puts the CTA's own
 * treatment on a value: it reads as an action to take rather than an option
 * chosen, and it sat beside a real submit button wearing exactly the same
 * fill.
 *
 * The selected segment is a surface step and foreground ink rather than a
 * fill, for the reason the whole surface is: a fill here would be the only
 * filled thing in a form that has none.
 *
 * Native radios rather than buttons with `role="radio"`: the semantics are
 * exactly a radio group's, and taking the real element brings the arrow-key
 * navigation and the one-tab-stop-per-group behavior with it instead of
 * reimplementing them. Each input is visually hidden and its label is the
 * segment, so the focus ring has to be drawn on the label.
 */
export function Segmented({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: { value: string; label: string }[]
  value: string
  onChange: (next: string) => void
}) {
  const name = useId()
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex w-fit max-w-full overflow-x-auto border border-control-border"
    >
      {options.map((option) => {
        const selected = option.value === value
        return (
          <label
            key={option.value}
            // Never shrinks and never wraps, the same rule a tab follows: a
            // segment that squeezed would put one control at two widths in one
            // row. The track scrolls instead.
            // The divider is a leading border on every segment but the first,
            // so the count of rules is always one less than the count of
            // segments, however many there are.
            className={`shrink-0 cursor-pointer border-l border-control-border px-3 py-[0.3125rem] text-sm whitespace-nowrap transition-colors first:border-l-0 has-[:focus-visible]:otari-focus-ring motion-reduce:transition-none ${
              selected
                ? "bg-surface-subtle text-foreground"
                : "text-muted hover:text-foreground"
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
