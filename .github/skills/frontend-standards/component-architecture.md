# Component architecture: how a page is put together

[web/AGENTS.md](../../../web/AGENTS.md) owns where a file goes (`features/`, `shared/`,
`app/`, `routes/`, `tests/`) and the three lint-enforced import rules between them. This file
is about the shape of the code inside a feature.

## File size is a design signal

The largest pages here are past two thousand lines (`features/activity/ActivityPage.tsx`,
`features/models/ModelsPage.tsx`), and their tests are larger still. Nothing about the domains
requires that: it is what happens when every dialog, row renderer, and derived table lives in
the file that renders the page.

The rule is not a line count, it is this: **a page component composes; it does not also
implement.** When you are in a page file and about to add a modal body, a second table, or a
twenty-line pure derivation, put it in its own file in the same feature directory and import
it. Splitting an existing giant is welcome as its own change; growing one is not.

What belongs where inside `features/<domain>/`:

- `Page.tsx`: the route's component. Reads hooks, owns the URL state, and composes.
- One file per dialog, panel, table, or card the page renders, colocated with its test.
- Pure derivations (sorting, grouping, summing, formatting a domain value) in their own
  module, because they are the part that is worth unit testing directly.

## Stateful pages, stateless parts

Data fetching, mutations, and URL state live in the page component. The parts it renders take
props and call back through `on*` handlers, which is what makes them testable without a
QueryClient and reusable in a second page.

```tsx
// features/keys/KeyRow.tsx: no hooks, no fetching, no router
interface KeyRowProps {
  apiKey: ApiKey
  onRevoke: () => void
}

export function KeyRow({ apiKey, onRevoke }: KeyRowProps) { … }
```

A part that needs to fetch is usually a part that should have been given the data.

## Do not copy structural markup

If the same wrapper (a page shell, a card layout, an empty-state panel, a filter bar) appears
in two files, extract it and let the callers pass the inside. Three shapes, in the order you
should reach for them:

- **`children`** when the layout is fixed and only the content varies. Most cases.
- **Named slot props** (`header`, `actions`, `footer`: `ReactNode`) when content goes into
  several distinct places.
- **A render prop** (`children` as a function) only when the layout has internal state the
  content needs, such as an open/closed disclosure.

A `variant` prop beats a second near-identical component. Two components that differ by 10% of
their markup will not stay in step.

## Check the primitives before writing one

`shared/components/` already holds the recurring pieces, in a directory per design topic:
`layout/`, `metrics/`, `feedback/`, `forms/`, `actions/`, `data/`, `navigation/`,
`indicators/`, `access/`, plus `deprecated/` for the four that must not be used in new code.
`web/design/DESIGN.md` maps every export to its module and is the inventory. Extend a
primitive rather than forking it, and add it to that table in the same change when you add
one.

## One component per file

A page or standalone component gets its own file, named for it, with its test beside it.
That now holds for the shared primitives too: `surface.tsx` and `ui.tsx` were the two
exceptions, 42 exports between them, and they are gone. A new primitive is a file of its own
under the topic directory it belongs to. Where a module carries a second export it is because
the two are useless apart (`RowAction` and the lane it sits in, `SpendMeter` and the
classifier that decides how it draws).

## No IIFEs in JSX

An immediately-invoked function inside markup hides real logic in the middle of a render tree
and cannot be tested. Compute it above the `return`, or make it a component.

```tsx
// Bad
<Cell>{(() => { const { tone, label } = statusChip(state); return <Chip tone={tone}>{label}</Chip> })()}</Cell>

// Good
const { tone, label } = statusChip(state)
return <Cell><Chip tone={tone}>{label}</Chip></Cell>
```

## Route files export `Route` and nothing else

This is the rule with a build consequence, and `src/routes.test.ts` enforces it. The Vite
plugin's `autoCodeSplitting` (`web/vite.config.ts`) can only lift a route's component into its
own chunk if the route module has one export. A second export, even a helper only a test
imports, pulls that route's whole component graph into the **entry** chunk, which every
visitor downloads on first paint, including for pages their deployment does not serve.

```tsx
// src/routes/models.tsx: the whole file
export const Route = createFileRoute("/models")({ component: ModelsPage })
```

The page component lives in `features/<domain>/`, and anything else the route needs
(`validateSearch`, a loader) stays inline in the route file. If a feature component needs the
route's search params or navigate, read them through the shared URL-state helpers
(`useUrlState`, `src/shared/helpers/urlState.ts`) rather than importing the route, which would
be circular.
