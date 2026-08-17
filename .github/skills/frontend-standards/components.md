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

<Card className="flex-1 min-w-[180px]">
  <Card.Content className="flex flex-col gap-1 p-5">…</Card.Content>
</Card>

<Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
  {confirmLabel}
</Button>
```

`variant` values the dashboard uses: `primary`, `ghost`, `outline`, `danger`, `danger-soft`.
`secondary` and the other HeroUI v3 variants are available if a new need arises; stick to the
in-use set unless you're deliberately adding one.

## Props over `className`

If a component exposes a prop for what you want (`variant`, `size`, `isDisabled`, `isPending`,
`fullWidth`, `isInvalid`), use the prop. Reserve `className` for layout and positioning
(`flex`, `gap-*`, `min-w-[…]`, responsive prefixes), not for re-skinning something HeroUI
already styles.

## Check the shared primitives before hand-rolling

`shared/components/ui/` is the rehomed design foundation's primitive directory and is where a
new primitive goes. It holds `SettingsSection` (a settings page section's header + body) and
`RowActions` (a table row's trailing button cluster) so far.

Everything in the table below lives one level up, in `shared/components/`, and is a
**migration bridge**: hand-rolled primitives that predate the foundation and still carry
`--otari-*` colors. Reuse them rather than duplicating their markup, but don't extend them and
don't build a new page on one; each leaves with the last page that uses it. See
[design-tokens.md](./design-tokens.md).

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

`errorMessage(error)` centralizes turning an `ApiError`/`Error`/unknown into a display string;
use it rather than reaching into `error.message` yourself.

## Layout and spacing

- Space siblings with `gap-*` on the flex/grid parent, not `m-*` on each child.
- Responsive via Tailwind breakpoints (`sm:`, `md:`, `lg:`) and flex/grid; avoid fixed pixel
  widths for anything that should reflow (`min-w-[180px]` on a wrapping stat card is fine).
- One component per file for pages and standalone components, colocated with its test. The
  bridge's `shared/components/ui.tsx` is the one place several closely related primitives
  share a file; a new primitive under `shared/components/ui/` gets its own.
