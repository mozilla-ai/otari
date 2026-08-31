/**
 * The settings page's boolean control. `role="switch"` so it reads correctly to
 * assistive tech and can be targeted by its accessible name (a config key, or
 * the maintenance-mode switch's own label).
 *
 * Its own file because two things on the page render one: every settable `bool`
 * config field, and the maintenance-mode switch, which is not a config field at
 * all. A second copy of this markup would be the two drifting apart.
 */
export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      // 44x24 with a 1px edge, filled with the page ground rather than a
      // surface step: on a flat plane the track is a drawn outline, not a
      // raised trough, so the state is carried entirely by the knob's color.
      //
      // The visible track stays 24px, which is what the settings rows are drawn
      // around, while `before` carries the 44px touch floor the phone viewport
      // requires past it (responsiveness.md). Absolutely positioned on a
      // `relative` button, so it grows the hit area without moving a row. Same
      // device the master-key reveal toggle uses on the sign-in screen.
      className="relative inline-flex h-6 w-11 shrink-0 items-center border border-control-border bg-background before:absolute before:inset-x-0 before:-inset-y-2.5 disabled:opacity-50"
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
          checked
            ? "translate-x-[21px] bg-accent"
            : "translate-x-px bg-control-thumb"
        }`}
      />
    </button>
  )
}
