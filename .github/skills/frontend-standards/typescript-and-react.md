# TypeScript & React conventions: `web/`

TypeScript runs in `strict` mode; `npm --prefix web run typecheck` must pass. React 19.

## TypeScript

- **`undefined`, not `null`, for absent values** in your own types and props. The API layer may
  hand back `null` (it mirrors the server JSON), convert at the boundary rather than letting
  `null` spread through the component tree. (`ApiError`-style third-party shapes that
  explicitly use `null` are the exception.)
- **Named exports**, not default exports, for components/hooks/helpers, consistent names
  across imports, better tooling and tree-shaking. (`web/` already does this throughout.)
- **Named imports**, not namespace imports (`import * as …`).
- Type the API surface in `web/src/api/types.ts` and thread those types through
  `apiFetch<T>(…)`; don't fetch into `any`.
- Let inference work for locals; annotate function signatures and exported values.

## React

- **`onPress`, not `onClick`**, for HeroUI interactive components (see
  [components.md](./components.md)).
- **Correct dependency arrays** on `useEffect`/`useMemo`/`useCallback`. Clean up subscriptions,
  intervals, and event listeners in the effect's return.
- **Derive, don't duplicate.** Compute values from props/query data during render instead of
  copying them into `useState` and syncing with effects. Server state lives in TanStack Query,
  not in component state (see [data-fetching.md](./data-fetching.md)).
- **Stable `key`s** for lists, a stable id, not the array index.
- Reach for memoization only when a re-render is measurably a problem, not by reflex.
- Keep a component per file, colocated with its test.

## Testing

Vitest + Testing Library (`@testing-library/react`, `@testing-library/user-event`,
`@testing-library/jest-dom` via `web/src/test/setup.ts`). Run `npm --prefix web test`.

- **Colocate**: `Foo.tsx` → `Foo.test.tsx`, `format.ts` → `format.test.ts`.
- **Query the way a user would**: `getByRole`, `getByLabelText`, `getByText`. Avoid
  `getByTestId` and CSS/class selectors, they break the moment markup or tokens change.
- **Render real providers, mock the network boundary.** Wrap the component in a real
  `QueryClientProvider` (see the existing page tests) so the real hooks and formatters run;
  set the master key with `setMasterKey(…)` and stub `apiFetch`'s endpoints, not the hooks
  themselves. Mocking `useModels`/`usePricing` directly hides query-key, loading, and error
  regressions.
- **Drive interactions with `userEvent`** and assert on the user-visible result; don't poke
  component internals.
- **Wait on the thing, not the clock.** Use `findBy*`/`waitFor` for async UI; never
  `setTimeout`-sleep a test.
- Keep test data explicit and deterministic (fixed ids, absolute dates like the existing
  `ModelsPage.test.tsx` fixtures), not `Date.now()`.
