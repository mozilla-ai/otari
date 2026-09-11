/**
 * A keyboard key, in a shortcut hint.
 *
 * `<kbd>` rather than a styled span, which is the whole reason this is a
 * component: the element carries the meaning, and a screen reader announces
 * "K" in a shortcut differently from "K" in a sentence.
 *
 * The keys wear `text-mono-micro`, which is the role for a value read against
 * what is beside it rather than as prose: a shortcut is something you
 * reproduce, and mono is what every other reproducible value here already
 * wears. That role sets a family and a size and no ink, which is why the muted
 * rung is spelled beside it rather than being a repeat of what it already
 * carries.
 */
export function Kbd({
  keys,
  className = "",
}: {
  /**
   * The chord, in press order: `["Cmd", "K"]`. Written out rather than parsed
   * from a string, so a component never has to guess whether "+" is a
   * separator or a key.
   */
  keys: readonly string[]
  className?: string
}) {
  return (
    <span className={`inline-flex items-center gap-1 ${className}`}>
      {keys.map((key, index) => (
        <span className="inline-flex items-center gap-1" key={key}>
          {index > 0 ? (
            // The separator is decoration: the keys are already separate
            // elements, so a screen reader reads the chord without it.
            <span aria-hidden className="text-subtle">
              +
            </span>
          ) : null}
          <kbd className="border border-border-strong bg-surface-alt px-1.5 py-0.5 text-mono-micro text-muted">
            {key}
          </kbd>
        </span>
      ))}
    </span>
  )
}
