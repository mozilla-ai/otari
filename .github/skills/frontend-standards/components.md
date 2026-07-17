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

Real example from `components/ui.tsx`:

```tsx
import { Button, Card } from "@heroui/react";

<Card className="flex-1 min-w-[180px]">
  <Card.Content className="flex flex-col gap-1 p-5">…</Card.Content>
</Card>

<Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
  {confirmLabel}
</Button>
```

`variant` values the dashboard uses: `primary`, `secondary`, `ghost`, `outline`, `danger`,
`danger-soft`. Stick to these unless you're deliberately adding a new one.

## Props over `className`

If a component exposes a prop for what you want (`variant`, `size`, `isDisabled`, `isPending`,
`fullWidth`, `isInvalid`), use the prop. Reserve `className` for layout and positioning
(`flex`, `gap-*`, `min-w-[…]`, responsive prefixes), not for re-skinning something HeroUI
already styles.

## Check `components/ui.tsx` before hand-rolling

Small shared primitives already exist. Reuse or extend them instead of duplicating markup:

| Need | Use |
|---|---|
| Labeled metric tile | `StatCard` |
| Error alert from an unknown thrown value | `ErrorBanner` (pairs with `errorMessage(error)`) |
| Info/warning callout | `InfoBanner` (`tone="info" \| "warning"`) |
| Page title + description + action | `PageHeader` |
| Destructive action without a modal | `ConfirmButton` (two-click arm/confirm) |
| Form field wrapper | `Field` (`components/Field.tsx`) |
| Tabular data | `Table` (`components/Table.tsx`) |

`errorMessage(error)` centralizes turning an `ApiError`/`Error`/unknown into a display string;
use it rather than reaching into `error.message` yourself.

## Layout and spacing

- Space siblings with `gap-*` on the flex/grid parent, not `m-*` on each child.
- Responsive via Tailwind breakpoints (`sm:`, `md:`, `lg:`) and flex/grid; avoid fixed pixel
  widths for anything that should reflow (`min-w-[180px]` on a wrapping stat card is fine).
- One component per file, colocated with its test.
