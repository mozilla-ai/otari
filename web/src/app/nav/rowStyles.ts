/**
 * The one place a sidebar row's shape is decided.
 *
 * Every row in either rail goes through `navRowClass`: the leaves, a group's
 * expand trigger, the "Back to" link at the head of the organization rail, the
 * Organization row in the footer, and the collapsed icon buttons. Keeping them
 * in one module is what stops the rail drifting a row at a time, and it mirrors
 * `otari-ai/frontend/src/app/navigationRowStyles.ts` so the two shells stay
 * recognizably the same file at M5.
 *
 * Three decisions worth knowing before editing:
 *
 * **44px is the floor, everywhere.** `min-h-11` rather than vertical padding, so
 * a row with a longer label that wraps grows instead of squashing, and so a row
 * is a comfortable touch target on the mobile drawer without a second rule.
 *
 * **Hover and selection move in opposite directions off the rail.** The rail
 * itself is `--color-background-muted`, and every state is one or two steps
 * along the neutral ramp from there. A transient pointer state presses *into*
 * the chrome, a persistent selection lifts *out* of it, which is what lets each
 * step stay this subtle and still be tellable apart: hover and selection are
 * both one step off the rail, so if they moved the same way they would be
 * neighbors on the ramp and read as the same fill.
 *
 * | state            | light                | dark                 |
 * | ---------------- | -------------------- | -------------------- |
 * | resting          | no fill              | no fill              |
 * | hover / focused  | `surface-subtle`     | `surface-subtle`     |
 * | pressed          | `border`             | `border`             |
 * | selected         | `surface-alt`        | `surface`            |
 * | selected + hover | `surface`            | `background`         |
 *
 * The first three rows need no `dark:` override because the ramp is a mirror
 * image per theme: `surface-subtle` and then `border` sit one step and then two
 * steps *below* the rail in light, one and two *above* it in dark, and either
 * direction is away from the chrome. Selection is the pair that does need one.
 * `surface-alt` (that is `--color-surface-muted`) is lighter than the rail in
 * light mode, but in dark the two are within a couple of units of lightness, so
 * the chip would be invisible; `dark:bg-surface` is the same lift read the other
 * way, cards sitting below chrome in the dark stack (fields, chrome, cards,
 * body, deepest last). Selected + hover then takes the one further step each
 * theme has left in that direction: pure white in light, the body ground in
 * dark. This is the navigation design's ramp, not an invention, and the selected
 * row is a lifted chip rather than a tinted one, which is why it is no longer
 * `bg-primary-subtle`.
 *
 * A selected row deliberately has no pressed state. It is the current page, so
 * the press has nowhere to go; it keeps answering the pointer with the hover
 * fill and stops there.
 *
 * **A nested child is indented with padding, not a narrower box.** `3.125rem`
 * clears the parent's icon lane (0.75rem padding + 1rem icon + 0.75rem gap) and
 * adds 0.625rem on top, which is what tells the eye the row is one level down,
 * and the row's fill still spans the rail. This is the one measurement where
 * `otari-ai/frontend`'s rail and the Paper design disagree: the design draws the
 * child's label in exactly its parent's lane (2.5rem). The sibling repo wins,
 * because these two rails are meant to be the same rail at M5.
 */

/**
 * The rail's motion, taken from HeroUI rather than invented.
 *
 * `color`/`background-color` over 150ms on `--ease-smooth` (plain `ease`) is
 * what HeroUI gives its own selectable navigation rows (`tabs.css`) and its
 * popover triggers, and `box-shadow` rides along for the focus ring. `transform`
 * is in the list for the rows that are HeroUI `Button`s (the collapsed group
 * triggers, the account control, the switcher): a Tailwind `transition-*`
 * utility replaces the `transition` shorthand the `.button` class sets, so
 * without naming it here the press-scale those buttons carry lands instantly.
 */
export const NAV_TRANSITION =
  "transition-[color,background-color,border-color,box-shadow,transform] duration-150 ease-smooth motion-reduce:transition-none"

/**
 * The keyboard ring: 2px of `--color-focus` at 2px of offset, on every row
 * whatever else it is wearing, which is why it lives in the base and not in one
 * of the state constants below.
 *
 * An `outline` rather than the `ring-2 ring-focus` HeroUI draws on its own
 * focusable components, for one reason: a ring is box-shadow, so its offset is
 * an opaque band that has to be told the color of the ground behind it, and
 * these rows sit on two grounds (the rail, and `--color-surface` inside a
 * collapsed group's flyout). Any single `ring-offset-*` color is wrong on one of
 * them, while an outline's offset is transparent and shows whatever is actually
 * there. `ring-offset-transparent` is not the way out of that: box-shadow layers
 * composite, so a transparent offset band over the ring paints the ring's color
 * rather than the ground.
 *
 * The last three utilities are for the rows that are HeroUI `Button`s. `.button`
 * carries `outline-none`, which leaves `--tw-outline-style: none` for
 * `outline-2` to read, so the style has to be named back; and it draws its own
 * `focus-ring` off `data-focus-visible`, which would otherwise sit inside this
 * outline as a second mark. Both are in the components layer, so a utility wins.
 */
const ROW_FOCUS =
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus focus-visible:outline-solid focus-visible:ring-0 focus-visible:ring-offset-0"

/**
 * `font-sans` is load-bearing, not decoration. A group's row is a
 * `Disclosure.Trigger` wrapped in a `Disclosure.Heading`, which HeroUI renders
 * as a real `<h3>`, and the base reset gives every `h1`-`h6` the display face.
 * The button inherits it, so without this the rows that hold sub-items (Routing,
 * Tools) come out in Zilla Slab while their leaf siblings are in Mozilla Text.
 * Naming the family on the shared row is what keeps the rail one typeface
 * wherever a row is rendered, rather than patching the one heading that has a
 * control inside it.
 *
 * Deliberately no `cursor-pointer`: every element this dresses already resolves
 * to `pointer` on its own. The leaves and both footer rows are `Link`s, so the
 * `<a href>` takes it from the user agent, and the expand triggers and the
 * account control carry HeroUI's `.disclosure__trigger` or `.button`, which set
 * `cursor: var(--cursor-interactive)`. Naming it here would be a no-op on all
 * five call sites and would also outrank `status-disabled`'s `--cursor-disabled`
 * on the account control, so a disabled row would promise a click.
 */
const ROW_BASE = `flex min-h-11 w-full items-center gap-3 rounded-lg px-3 font-sans text-sm font-medium leading-[1.375rem] ${NAV_TRANSITION} ${ROW_FOCUS}`

/**
 * `data-pressed` alongside `active` because a rail row is three different
 * elements: react-aria swallows `:active` on the HeroUI `Button`s (the collapsed
 * group triggers, the account control) and reports the press as that attribute
 * instead, while the `Link`s and the plain expand buttons only have `:active`.
 * The text color is repeated on each rather than left to `hover:`, because a
 * press can arrive without a hover, from touch or from Space on a focused row.
 */
const ROW_PRESSED =
  "active:bg-border active:text-foreground data-[pressed]:bg-border data-[pressed]:text-foreground"

const ROW_RESTING = `text-muted hover:bg-surface-subtle hover:text-foreground focus-visible:bg-surface-subtle focus-visible:text-foreground ${ROW_PRESSED}`

const ROW_SELECTED =
  "bg-surface-alt text-foreground hover:bg-surface dark:bg-surface dark:hover:bg-background"

/** The class list for one sidebar row. */
export function navRowClass({
  isActive = false,
  collapsed = false,
  nested = false,
}: {
  isActive?: boolean
  collapsed?: boolean
  nested?: boolean
} = {}): string {
  return [
    ROW_BASE,
    isActive ? ROW_SELECTED : ROW_RESTING,
    nested ? "pl-[3.125rem]" : "",
    collapsed ? "min-w-11 justify-center px-0" : "",
  ]
    .filter(Boolean)
    .join(" ")
}

/** The heading above a group of rows. 32px of label supplies the group's air. */
export const NAV_SECTION_HEADING_CLASS =
  "flex min-h-8 items-center px-3 text-overline"

/**
 * A row's leading glyph: 16px, never shrinking, which is what
 * `otari-ai/frontend` passes every one of its `react-icons` marks and what the
 * design draws.
 */
export const NAV_ICON_CLASS = "size-4 shrink-0"

/**
 * A row's trailing chevron: the group triggers, the account control, and the
 * scope switcher, each pointing down until the thing it opens is open and then
 * turning over 150ms, which is `otari-ai/frontend`'s `NavigationBranch`.
 *
 * A function rather than a constant so the rotation cannot drift from the
 * transition that carries it: three call sites turning their own chevron is
 * three chances to write `-rotate-180` in one of them and animate the other way.
 */
export function navIndicatorClass({ open = false }: { open?: boolean } = {}) {
  return [
    "size-4 shrink-0 transition-transform duration-150 motion-reduce:transition-none",
    open ? "rotate-180" : "",
  ]
    .filter(Boolean)
    .join(" ")
}
