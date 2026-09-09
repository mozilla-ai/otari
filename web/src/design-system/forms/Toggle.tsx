/**
 * A boolean that takes effect on its own, with no Save behind it.
 *
 * That is the whole line between this and `Checkbox`: a switch commits when it
 * is flipped, a checkbox is part of a form somebody submits. Rendered as
 * `role="switch"` so assistive tech reads it as the former, and so it can be
 * targeted by its accessible name.
 *
 * Its props match `Checkbox`'s deliberately. They used to differ (`checked` and
 * `disabled` against `isSelected` and `isDisabled`), which forms.md called out
 * as a real inconsistency in the tree rather than a typo in the docs; two
 * controls an operator reads as a pair should not need two prop vocabularies,
 * so rehoming this into a shared library is where that closes.
 */
export function Toggle({
  isSelected,
  onChange,
  label,
  isDisabled,
}: {
  isSelected: boolean
  onChange: (next: boolean) => void
  /** The accessible name. A switch has no visible text of its own. */
  label: string
  isDisabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={isSelected}
      aria-label={label}
      disabled={isDisabled}
      onClick={() => onChange(!isSelected)}
      // 44x24 with a 1px edge, filled with the page ground rather than a
      // surface step: on a flat plane the track is a drawn outline, not a
      // raised trough, so the state is carried entirely by the knob's color.
      //
      // The visible track stays 24px, which is what the settings rows are drawn
      // around, while `before` carries the 44px touch floor the phone viewport
      // requires past it (responsiveness.md). Absolutely positioned on a
      // `relative` button, so it grows the hit area without moving a row. Same
      // device the master-key reveal toggle uses on the sign-in screen.
      className="relative inline-flex h-6 w-11 shrink-0 items-center border border-control-border bg-background before:absolute before:inset-x-0 before:-inset-y-2.5 disabled:opacity-(--disabled-opacity)"
    >
      <span
        // 20x20 inset 1px from the track's inner edge, which `items-center`
        // already gives vertically once the border is accounted for. No border,
        // no ring and no shadow on the knob in either theme: the edge it used
        // to carry was `shadow-elevation-sm`, which on dark was a 1px white
        // ring rather than a shadow, and both are gone with nothing in their
        // place. Contrast against the ground is what separates it now.
        //
        // `transition-transform` and not `transition-colors`: the fill changes
        // with the state and should read as instant, while the travel is what
        // benefits from being followed.
        className={`inline-block h-5 w-5 transform transition-transform duration-150 motion-reduce:transition-none ${
          isSelected
            ? "translate-x-[21px] bg-control-indicator"
            : "translate-x-px bg-control-thumb"
        }`}
      />
    </button>
  )
}
