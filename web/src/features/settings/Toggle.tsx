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
      // The visible track stays 24px, which is what the settings rows are drawn
      // around, while `before` carries the 44px touch floor the phone viewport
      // requires past it (responsiveness.md). Absolutely positioned on a
      // `relative` button, so it grows the hit area without moving a row. Same
      // device the master-key reveal toggle uses on the sign-in screen.
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full outline-none transition-colors before:absolute before:-inset-y-2.5 before:inset-x-0 focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50 ${
        checked ? "bg-accent" : "bg-surface-subtle"
      }`}
    >
      <span
        className={`inline-block h-5 w-5 transform rounded-full bg-surface shadow transition-transform ${
          checked ? "translate-x-5" : "translate-x-0.5"
        }`}
      />
    </button>
  )
}
