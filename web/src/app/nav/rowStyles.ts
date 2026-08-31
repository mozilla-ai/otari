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
 * **Hover and selection are steps off the rail, and press no longer is.** Every
 * state is one or two steps along the neutral ramp from whatever the rail is.
 * Hover and selection deliberately move by different amounts rather than in the
 * same direction by different degrees, so they cannot read as one fill.
 *
 * | state            | light                | dark                 |
 * | ---------------- | -------------------- | -------------------- |
 * | resting          | no fill              | no fill              |
 * | hover / focused  | `surface-subtle`     | `surface-subtle`     |
 * | pressed          | `background`         | `background`         |
 * | selected         | `surface-alt`        | `surface-alt`        |
 * | selected + hover | `surface-alt` (none) | `surface-alt` (none) |
 *
 * No `dark:` override is needed for any of them, because the surface family is
 * staggered one rung above the background family in both themes. An earlier
 * ladder put `surface` *below* chrome in dark and needed `dark:bg-surface` to
 * produce a lift; under the current mapping that override resolves to the
 * rail's own value and erases the selection, which is why it is gone.
 *
 * **The pressed fill currently does nothing, and that is a known open item
 * rather than an oversight here.** The rail was `--color-background-muted`, one
 * rung above the canvas, and a press was the canvas showing through it. The
 * shell is flat now: the rail *is* `--color-background`, so `active:bg-background`
 * paints the rail's own value. Measured on the running page as L* from the rail:
 *
 * |          | dark  | light |
 * | -------- | ----- | ----- |
 * | hover    | +9.83 | -5.18 |
 * | selected | +6.77 | -2.41 |
 * | pressed  |  0.00 |  0.00 |
 *
 * Hover and selection still read, and gained contrast from the rail dropping a
 * rung. Press has nowhere below the ground to go, so restoring it means picking
 * a different direction for it, which is a design decision and not one to make
 * from inside this file. The classes are left in place rather than deleted so
 * the state has somewhere to land when that is decided.
 *
 * The selected row is a lifted chip rather than a tinted one, which is why it
 * is no longer `bg-primary-subtle`.
 *
 * A selected row answers the pointer with neither hover nor press. It is the
 * current page, so clicking it is a no-op and there is nothing for an
 * affordance to promise. This is a CHANGE from the earlier rule, which kept the
 * hover fill: `ROW_RESTING` gives a hovered row `hover:text-foreground` as well
 * as a fill, so a selected row that also took the hover fill became
 * indistinguishable from a hovered resting one - and losing "is the row under
 * my pointer the page I am on" costs more than losing a hover response on a row
 * that does nothing when clicked.
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
 * The keyboard ring, on every row whatever else it is wearing, which is why it
 * lives in the base and not in one of the state constants below.
 *
 * The ring itself is `focus-ring`, defined once in globals.css. Its values are
 * not spelled here and must not be: nine call sites used to spell their own and
 * they disagreed about which token a ring comes from. What is still spelled
 * here is why this row cannot use the base rule instead.
 *
 * It is an outline and not the box-shadow ring HeroUI draws on its own
 * focusable components, for one reason: a shadow ring's offset is an opaque
 * band that has to be told the color of the ground behind it, and these rows
 * sit on two grounds (the rail, and `--color-surface` inside a collapsed
 * group's flyout). Any single offset color is wrong on one of them, while an
 * outline's offset is transparent and shows whatever is actually there. A
 * transparent offset band is not the way out of that: box-shadow layers
 * composite, so it would paint the ring's color rather than the ground.
 *
 * The two suppressions are for the rows that are HeroUI `Button`s, which draw
 * their own inner ring off `data-focus-visible`; without them it sits inside
 * this outline as a second mark. Both are in the components layer, so a utility
 * wins. `.button` also carries `outline-none` there, which is why the ring has
 * to arrive as a utility at all rather than from the base rule.
 */
const ROW_FOCUS =
  "focus-visible:otari-focus-ring focus-visible:ring-0 focus-visible:ring-offset-0"

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
 * Deliberately no `cursor-pointer`, and it is the one thing a call site is left
 * to name. Nearly every element this dresses already resolves to `pointer` on
 * its own: the leaves and the footer's desktop rows are `Link`s, so the
 * `<a href>` takes it from the user agent, and the expand triggers and the
 * account control carry HeroUI's `.disclosure__trigger` or `.button`, which set
 * `cursor: var(--cursor-interactive)`. The exceptions are the mobile drawer's
 * two bare `<button>`s, the row that opens the organization submenu and the row
 * that leaves it again, where the user agent gives a plain button the default
 * arrow; both add the utility themselves. It stays out of the base because here
 * it would also outrank `status-disabled`'s `--cursor-disabled` on the account
 * control, so a disabled row would promise a click.
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
  "active:bg-background active:text-foreground data-[pressed]:bg-background data-[pressed]:text-foreground"

/**
 * Focus takes the ring and nothing else. It used to take `bg-surface-subtle` as
 * well, the same fill as hover, which cost two things. A row that can be
 * selected cannot also spend the fill channel on focus: the central ring exists
 * so focus never has to borrow another state's paint. And because a fill
 * outlives the click that put focus on the row while a pointer's hover does not
 * outlive the pointer, a group trigger clicked open kept a fill afterwards and
 * read as selected when it was only focused.
 */
const ROW_RESTING = `text-muted hover:bg-surface-alt hover:text-foreground focus-visible:text-foreground ${ROW_PRESSED}`

/**
 * The selected row, and the two things that make it unmistakable.
 *
 * Its fill and hover's have swapped. Selection had `surface-alt` and hover had
 * `surface-subtle`, and `surface-subtle` is the louder of the two on both
 * themes (lighter on dark, darker on light), so the transient state was
 * shouting over the permanent one and moving the pointer made any row look more
 * current than the row that actually was. Selection takes the louder rung now,
 * which also realigns with the mapping table, where an active control's fill is
 * `surface-subtle`.
 *
 * The 2px left edge is what a hover can never borrow. Two adjacent rungs of one
 * ramp are a fragile way to carry "which page am I on", so the state also gets a
 * structural channel, in the rules-not-fills vocabulary the rest of the redesign
 * is built from. The left padding gives the 2px back so a selected row's label
 * stays in the same lane as its siblings'.
 */
const ROW_SELECTED =
  "bg-surface-subtle text-foreground border-l-2 border-foreground pl-[calc(0.75rem-2px)]"

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
