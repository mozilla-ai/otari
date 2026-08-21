# Components: HeroUI v3 + the shared UI primitives

The dashboard uses HeroUI **v3** (`@heroui/react`). v3 is a ground-up rewrite; v2 habits are
wrong here.

## v2 → v3

| Concern | v2 (wrong here) | v3 (correct) |
|---|---|---|
| Import | `@heroui/button`, `@heroui/card` | unified `@heroui/react` |
| Provider | `HeroUIProvider` | none needed for these components |
| Structure | flat `Card`, `Modal` | compound: `Card.Header` / `Card.Content` / `Card.Footer` |
| Styling override | `classNames={{ slot: "…" }}` | `className` on the subcomponent directly |
| Change handler | `onValueChange` | `onChange` (v3 ignores `onValueChange` silently) |
| Click handler | `onClick` | `onPress` |
| Button intent | `color="danger"` | `variant="danger"` |
| Disabled | `isLoading` for disabled | `isDisabled` / `isPending` |

Real example from `shared/components/ui.tsx`:

```tsx
import { Button, Card } from "@heroui/react";

<Card className="flex-1 min-w-[11.25rem]">
  <Card.Content className="flex flex-col gap-1 p-5">…</Card.Content>
</Card>

<Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
  {confirmLabel}
</Button>
```

`variant` values the dashboard uses: `primary`, `ghost`, `outline`, `danger`, `danger-soft`.
`secondary` and the other HeroUI v3 variants are available if a new need arises; stick to the
in-use set unless you're deliberately adding one.

## Customizing: which layer to reach for

**Start from a component, not from an element.** A HeroUI component (or one of the shared
primitives below) already resolves through the tokens and already carries its states: the
pointer, the focus ring, the disabled dimming and the press animation arrive with it and cost
nothing at the call site. A hand-rolled `<button>` or `<div>` starts from Tailwind's reset,
which is to say from nothing, so every one of those states becomes a class somebody has to
remember on every copy of that markup. The dashboard's own nav rail is the worked example: its
rows are a mix of router `Link`s, plain `<button>`s and HeroUI `Button`s, and the plain ones
answer the pointer with the default arrow, so the shared row string has to name
`cursor-pointer` itself, where a HeroUI `Button` takes it from `--cursor-interactive` and a
bare `<button>` takes it from no one. Reaching for a native element is sometimes right
(`FilterSelect` is a deliberately token-styled native `<select>`, and a nav row has to be the
router's `Link`), but it is a decision that buys styling work, not a neutral default.

Once you know what you are styling, there are four ways to change how it looks, in the order to
try them. The first three are reusable, they follow a retheme, and they survive a HeroUI
upgrade. The fourth is a rule written against another project's private DOM, which is why it is
the last resort and not the default.

**1. A variable: ours, or the library's aliased onto ours.** If the value is one of the design
system's roles, it is a token, and if the role is missing, add it rather than writing the value
down somewhere. If it is a value the library computes from a variable of its own, alias that
variable instead of overriding the rules that read it. Color already works this way here: each
theme block maps `--surface`, `--accent`, `--focus` and the rest onto our `--color-*` tokens,
which is what makes a bare `<Card>` wear the palette. The same handle exists for geometry and
interaction, and this repo has not claimed it yet. HeroUI declares `--radius`, `--field-radius`,
`--spacing`, `--border-width`, `--disabled-opacity`, `--cursor-interactive` /
`--cursor-disabled` and the `--scrollbar-*` family in
`@heroui/styles/dist/themes/default/variables.css`, and derives its whole radius ramp from
`--radius` in `dist/themes/shared/theme.css` (`--radius-2xl` is `calc(var(--radius) * 2)`, so a
16px arc a component draws is one alias away from being ours), and `globals.css` sets none of
them. Read the value out of `@heroui/styles/dist` before concluding that a rule is the only way
to reach it. See [design-tokens.md](./design-tokens.md).

**2. A shared utility, once the look repeats.** The second call site that wants the same look
gets a utility or a shared class string, not the same class list typed again: the
`@utility text-heading` family in `globals.css` carries the type roles, and `navRowClass` and
`NAV_TRANSITION` in `app/nav/rowStyles.ts` carry a chrome row. Two copies of a class list are
two surfaces that are meant to match and will stop matching one fix at a time, and the copy
that missed the fix is the one somebody notices.

**3. The component's own API.** `variant`, `size`, `isDisabled`, `isPending`, `fullWidth`,
`isInvalid`, and on a compound component the `className` of the subcomponent that owns the part
you mean. Read the variants before assuming there is no prop for what you want: `Table.Root`
takes one, and its `secondary` is documented as no background, padding, or rounding on the
root, which is a prop for something `.otari-table` currently neutralizes by hand. Reserve
`className` for layout and positioning (`flex`, `gap-*`, `min-w-[…]`, responsive prefixes), not
for re-skinning something HeroUI already styles.

**4. Last resort: a rule that reaches into the library's DOM.** `.otari-*` in `globals.css` is
the sanctioned namespace for it (see [design-tokens.md](./design-tokens.md)), and some cases
genuinely land here: a keyframe, something that has to outrank an inline style, or a value the
library paints somewhere it gives you no name for. It carries a standing cost, so it is worth
being sure none of the three above reach it. `.table__cell` and
`.table__body tr:first-child td:first-child` are internal names that can change in a patch
release with nothing in this repo failing; the rule is invisible from the call site, so the
next reader checks the component's props, believes them, and is wrong; and the rule wins the
cascade by being unlayered against `@heroui/styles`'s `@layer components`, not by its
specificity, which is a distinction worth stating correctly in the comment rather than
restating as folklore. When you write one, name in that comment which of the three rungs above
does not reach the value, so a reader can tell a deliberate last resort from a shortcut.

## Check the shared primitives before hand-rolling

`shared/components/ui/` is the rehomed design foundation's primitive directory and is where a
new primitive goes. It holds `SettingsSection` (a settings page section's header + body) and
`RowActions` (a table row's trailing button cluster) so far.

Everything in the table below lives one level up, in `shared/components/`. These are
hand-rolled rather than rehomed (they predate the foundation and have no otari-ai
counterpart), but they are on the semantic tokens like everything else, so reuse them rather
than duplicating their markup. See [design-tokens.md](./design-tokens.md).

| Need | Use |
|---|---|
| Labeled metric tile | `StatCard` |
| Error alert from an unknown thrown value | `ErrorBanner` (pairs with `errorMessage(error)`) |
| Info/warning callout | `InfoBanner` (`tone="info" \| "warning"`) |
| Page title + description + action | `PageHeader` |
| Destructive action without a modal | `ConfirmButton` (two-click arm/confirm) |
| Filter over a small fixed option set | `FilterSelect` (token-styled native `<select>`) |
| Filter over a large or open option set | `FilterMultiComboBox` (type-to-filter, holds a set of values; `allowsCustom` when the value space is not enumerable) |
| Applied filters, each removable | `FilterChips` (`shared/components/FilterChips.tsx`); one chip per value, and pass `clearLabel` so several chips of one dimension stay distinguishable |
| Form field wrapper | `Field` (`shared/components/Field.tsx`) |
| Tabular data | `DataTable` (`shared/components/DataTable.tsx`) |
| Settings page section (header + body) | `SettingsSection` (`shared/components/ui/`, rehomed) |
| Table row's trailing icon-button cluster | `RowActions` (`shared/components/ui/`, rehomed) |

`errorMessage(error)` centralizes turning an `ApiError`/`Error`/unknown into a display string;
use it rather than reaching into `error.message` yourself.

**When you add a shared primitive, add its row to that table in the same change.** The table
is only useful while it is complete, and duplication in this tree has never come from a
missing rule; it comes from not knowing the primitive already existed.

## Internal links go through the router

HeroUI's `Link` is a react-aria link with no knowledge of TanStack Router, so
`<Link href="/models">` triggers a **full page reload**: the bundle re-downloads, the query
cache is thrown away, and the bootstrap round trip runs again. For anything inside the app,
use the router's own link:

```tsx
import { Link } from "@tanstack/react-router"

<Link to="/models" className="text-link hover:text-link-hover">Models</Link>
```

For a control that looks like a button, call `navigate` from a `Button`'s `onPress` rather
than nesting: `<Link><Button/></Link>` renders an `<a>` around a `<button>`, which is invalid
HTML and gives the two elements conflicting keyboard behavior. HeroUI's `Link` stays correct
for genuinely external destinations (documentation, otari.ai).

## Layout and spacing

- Space siblings with `gap-*` on the flex/grid parent, not `m-*` on each child. A shared
  component never bakes in its own outside margin: what sits between two things is the
  parent's decision, and a component that decides it cannot be recomposed.
- Arbitrary values are in `rem`, not `px` (`h-[20rem]`, not `h-[320px]`), so they follow the
  reader's root font size. A `1px` border is the exception; a `text-[11px]` is not, because
  that size is the `text-overline` role. See [responsiveness.md](./responsiveness.md).
- Responsive via Tailwind breakpoints (`sm:`, `md:`, `lg:`) and flex/grid; avoid fixed pixel
  widths for anything that should reflow (`min-w-[11.25rem]` on a wrapping stat card is fine:
  it is a floor, not a fixed width).
- One component per file for pages and standalone components, colocated with its test.
  `shared/components/ui.tsx` is the one place several closely related primitives share a
  file; a new primitive under `shared/components/ui/` gets its own.
