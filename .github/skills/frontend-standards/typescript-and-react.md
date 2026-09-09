# TypeScript & React conventions: `web/`

TypeScript runs in `strict` mode; `pnpm --dir web run typecheck` must pass. React 19.

## TypeScript

- **`undefined`, not `null`, for absent values** in your own types and props. The API layer may
  hand back `null` (it mirrors the server JSON), convert at the boundary rather than letting
  `null` spread through the component tree. (`ApiError`-style third-party shapes that
  explicitly use `null` are the exception.)
- **Named exports**, not default exports, for components/hooks/helpers, consistent names
  across imports, better tooling and tree-shaking. (`web/` already does this throughout.)
- **Named imports**, not namespace imports (`import * as …`).
- Take the API surface from the generated client (`import type { … } from "@/client"`,
  regenerated from the OpenAPI spec) and thread those types through `apiFetch<T>(…)`;
  don't fetch into `any` and don't hand-write a wire shape. The one sanctioned
  exception is `web/src/client/local.ts`, for the few shapes the spec does not
  describe (routing-policy bodies, `/dashboard-build.json`, `UsageFilters`), and each
  entry there says why; adding to it is a decision, not a shortcut. See
  [web/AGENTS.md](../../../web/AGENTS.md).
- Let inference work for locals; annotate function signatures and exported values.
- **`unknown`, not `any`, at a boundary you cannot type** (a thrown value, an opaque payload),
  and narrow it with a guard before use. `errorMessage(error: unknown)` in
  `shared/components/feedback/errorMessage.ts` is the pattern: one place turns an unknown
  into a display string.
  Biome's `noExplicitAny` is off in `web/biome.jsonc` because the tree still has older `any`s,
  which makes it a convention rather than a lint error; do not add to the pile.
- **A discriminated union beats a bag of optionals** for anything with states. `{ status:
  "error"; message: string } | { status: "success"; data: T }` makes `state.data` on the error
  branch a compile error, where `{ status, message?, data? }` makes it a runtime `undefined`.
- **`as const` on a literal table** that drives a union (`THEME_PREFERENCES` in
  `shared/hooks/useTheme.tsx`, the nav registry's `as const satisfies readonly NavSection[]`),
  so the values stay literals and the derived type is the set rather than `string[]`.

## React

- **`onPress`, not `onClick`**, for HeroUI interactive components (see
  [components.md](./components.md)).
- **Correct dependency arrays** on `useEffect`/`useMemo`/`useCallback`. Clean up subscriptions,
  intervals, and event listeners in the effect's return.
- **Derive, don't duplicate.** Compute values from props/query data during render instead of
  copying them into `useState` and syncing with effects. Server state lives in TanStack Query,
  not in component state (see [data-fetching.md](./data-fetching.md)).
- **Stable `key`s** for lists, a stable id, not the array index.
- **The React Compiler is enabled** (`babel-plugin-react-compiler`, wired up in
  `vite.config.ts`), so memoization is the build's job. Do not add `useMemo`, `useCallback`, or
  `React.memo` without a measurement or a specific reference the compiler cannot prove stable.
  It also means the rules of hooks are load-bearing: the compiler silently skips a component it
  cannot verify. See [performance.md](./performance.md).
- Keep a component per file, colocated with its test.

## Testing

Vitest and Testing Library, colocated with the code they cover, mocking the transport rather
than the hooks. The rules, the harnesses in `src/tests/`, and the two Playwright suites are in
[testing.md](./testing.md).
