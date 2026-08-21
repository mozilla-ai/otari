---
name: frontend-standards
description: Guidelines for the otari admin dashboard (`web/`), React 19 + TypeScript (strict) + HeroUI v3 + Tailwind v4 + TanStack Query. Use when writing or reviewing dashboard components, styling, data fetching, or tests.
---

# Frontend Standards: otari dashboard (`web/`)

`web/` is the standalone admin dashboard: a React SPA that talks to the gateway's management
API with the master key. It is an operator tool, not a general-purpose app; keep its footprint
small and its conventions consistent with what is already there. There is no analytics and no
marketing surface here; do not add either to match some other repo.

Stack: React 19 (with the React Compiler), TypeScript (`strict`), HeroUI v3 (`@heroui/react`),
Tailwind CSS v4, TanStack Query, TanStack Router (file-based, `web/src/routes/`), Vite,
Vitest + Testing Library, Playwright (`web/e2e/`, behavioral and screenshot suites). Package
manager is **pnpm**.

[web/AGENTS.md](../../../web/AGENTS.md) owns the structure and is worth reading first: the
`features/` / `shared/` / `app/` layout it mirrors from `otari-ai/frontend`, the three
lint-enforced import rules, the routing conventions, the generated API client, and the design
foundation. This file is the house style for writing code inside that structure.

Build and check from the repo root:

- `make dashboard` (installs from the lockfile if needed, then `pnpm --dir web run build`).
  Output goes to the gitignored `src/gateway/static/dashboard/`; there is nothing to commit.
  Build only when you need to run the dashboard locally; Docker builds it in its own Node stage.
- `pnpm --dir web run lint` (Biome: formatting, recommended rules, layer boundaries; `lint:fix` writes)
- `pnpm --dir web run typecheck`
- `pnpm --dir web test`
- `pnpm --dir web run e2e` (behavioral) and `pnpm --dir web run e2e:screenshots` (visual)

pnpm is pinned by `packageManager` in `web/package.json`; CI and Docker both take the version
from there. `pnpm-workspace.yaml` carries the build-script approvals, and it is why an install
can link Vite's esbuild binary at all.

## Critical rules

**Always:**

- Reach for a HeroUI component or a shared primitive before a native element: it arrives with
  the tokens and its states (pointer, focus ring, disabled dimming) already wired, where a
  hand-rolled `<button>` starts from Tailwind's reset and every state becomes a class somebody
  has to remember. Then change how it looks in this order: a variable (ours as a token, or one
  of HeroUI's own aliased onto ours; its documented knobs include `--radius`,
  `--cursor-interactive` and `--disabled-opacity`, not just color), a wrapper or utility once
  the look repeats, the component's own prop (`variant`, `size`, `isDisabled`, `isPending`,
  `fullWidth`, `isInvalid`), and last a rule against HeroUI's own classes. HeroUI supports that
  last one and it stays discouraged: a selector fixes one case where a variable fixes every rule
  that reads it, it is invisible from the call site, and because the rules in `globals.css` are
  unlayered they outrank a Tailwind class at the call site too. Write one only when nothing above
  reaches the value, and say so in its comment. `className` is for layout/position, not for restyling a component HeroUI
  already styles. See [components.md](./components.md).
- Style from the semantic tokens in `web/src/styles/globals.css`, through the utilities they
  back (`text-muted`, `bg-surface`, `border-border`, `text-danger`, `text-heading`). The
  tokens are the design system and HeroUI is a consumer of it: a utility that does not resolve
  to a token will not follow a retheme. Never a raw hex, never a numbered Tailwind palette
  class, never `bg-white` / `text-black`; `src/styles/foundation.test.ts` fails on all three,
  over the whole of `web/src`. [design-tokens.md](./design-tokens.md) has the families, the
  HeroUI mapping, the chart palettes, the type scale, and the near-synonym utilities HeroUI
  ships that look like ours and are not.
- Space siblings with `gap-*` on the parent, and write arbitrary values in `rem`. See
  [responsiveness.md](./responsiveness.md).
- Fetch server state through the TanStack Query hooks in `web/src/shared/api/hooks.ts`, guard
  on `isPending && !data`, keep the previous page with `placeholderData` on a filtered query,
  and bound every "fetch everything" walk with a hard page cap. See
  [data-fetching.md](./data-fetching.md).
- Design for the phone as well as the desk. The dashboard is installable to a home screen, the
  shell already switches to a drawer below `md`, and the screenshot matrix captures every page
  at 390px. Touch targets ≥44px, no hover-only controls, a table needs an answer below `md`.
  See [responsiveness.md](./responsiveness.md).
- Prefer `undefined` over `null` for absent values in your own types (the API layer may return
  `null`; convert at the boundary). See [typescript-and-react.md](./typescript-and-react.md).
- Gate a deployment-dependent surface through `useDeployment()` / `useSurfaces()`, the one place
  that knows which deployment served the page. Mind the vocabulary: a *surface* is the
  deployment axis, a *capability* is the entitlement axis.
- Declare a new **rail** destination in the nav registry (`web/src/app/nav/registry.ts`), never
  as a hand-written link, with whichever of the three gates it needs (`surface`, `capability`,
  `flag`). `EntitlementGate` is the component form for wrapping a page. See
  [web/AGENTS.md](../../../web/AGENTS.md) for how the gates compose and where a capability
  the base build ships has to be declared. A route the *chrome* reaches is the exception and
  is a hand-written `Link` on purpose: `/docs` from the top bar and the account menu, and
  `/account` from that menu's first row. The registry is what the rails render, so an entry
  there would duplicate into the sidebar a row the design draws once. Adding to the sidebar
  has no exception.
- Add a Vitest test for any component, hook, or helper whose behavior you change (`Foo.tsx` →
  `Foo.test.tsx`, colocated), and a screenshot entry for any new page. The screenshot suite
  runs on demand rather than as a PR gate while the migration lands, so the entry is owed
  even though nothing fails without it. See [testing.md](./testing.md).

**Never:**

- New HeroUI **v2** patterns: granular imports, `HeroUIProvider`, `classNames={{ slot }}`,
  `onValueChange`, `color` on `Button`, or a `content1`/`content2` utility. v3 ignores some of
  these silently, which is why the full v2-to-v3 table is in [components.md](./components.md).
- Inline `style={{}}` or `<style>` tags. Use Tailwind utilities or a token. (The pre-paint
  block in `index.html` is the one exception, and [layout-stability.md](./layout-stability.md)
  says why.)
- A HeroUI `<Link href>` for an internal route: it is a full page reload. Use TanStack
  Router's `<Link to>`.
- Manual polling with bare `setInterval`/`setTimeout`. Use TanStack Query's `refetchInterval`
  (see `useDashboardBuild`).
- A raw `fetch()` for **authenticated** management requests. Go through `apiFetch`, which owns the
  Bearer key, error extraction, and 401/403 sign-out. `validateMasterKey` is the one sanctioned
  exception, and [data-fetching.md](./data-fetching.md) says why.
- Client-side filtering/sorting/pagination of large server datasets when the endpoint can do
  it. (Small, already-loaded lists rendered in a `Table` are fine.)
- Memoization by reflex. The React Compiler is enabled; add `useMemo`/`useCallback`/`memo`
  only with a measurement behind it. See [performance.md](./performance.md).
- A second export from a route file. It defeats `autoCodeSplitting` and lands the page in the
  entry chunk. See [component-architecture.md](./component-architecture.md).
- Barrel files, default exports, or namespace imports. See
  [imports-and-modules.md](./imports-and-modules.md).
- `getByTestId` when a semantic query (`getByRole`, `getByLabelText`, `getByText`) works.
- A page component that branches on the gateway's mode (reading `/v1/settings`'s `mode`, or
  probing whether a management endpoint 404s). The deployment bootstrap answers that once,
  for the whole shell; see the rule above.

## A note on status colors

The foundation names the status roles, so use them: `text-danger` / `bg-danger-subtle` for
errors, `text-warning` / `bg-warning-subtle` for caution, `text-success` / `bg-success-subtle`
for healthy, `text-info` / `bg-info-subtle` for neutral notices, and the separate
`bg-attention` family for "look here" (a required action, an unread marker) as distinct from
"be careful". Every one of them is defined for both themes.

Two pairings are worth knowing because the obvious guess is wrong. **A status word on its own
subtle fill wears the status color** (`text-danger` on `bg-danger-subtle`), and the light
theme's danger, warning, and attention values are a step darker than otari-ai's swatches for
exactly that reason: at the original values those pairings sat between 3.2:1 and 4.4:1, under
AA for the small text a pill uses. **Brand text on the brand tint does not follow that rule**:
`--color-primary` on `--color-primary-subtle` is 3.8:1, so a chip or an active nav item takes
`text-primary-subtle-foreground` instead.

## Topic guides

- [design-tokens.md](./design-tokens.md): the semantic tokens, the HeroUI mapping, the type scale, the chart palettes, and how to translate otari-ai's utility names.
- [components.md](./components.md): HeroUI v3 patterns, the order to reach for when customizing (variable, shared utility, prop, then a rule into the library's DOM), internal links, the shared UI primitives in `shared/components/`.
- [component-architecture.md](./component-architecture.md): what a page composes, what gets its own file, route files, no duplicated markup.
- [data-fetching.md](./data-fetching.md): TanStack Query conventions: query keys, `staleTime`, guards, invalidation, bounded pagination.
- [typescript-and-react.md](./typescript-and-react.md): strict TS, `undefined` over `null`, discriminated unions, hook and effect hygiene.
- [responsiveness.md](./responsiveness.md): breakpoints, touch targets, tables on a phone, `rem` over `px`.
- [layout-stability.md](./layout-stability.md): loading guards, skeletons, the pre-paint theme script, no reload-as-refresh.
- [performance.md](./performance.md): the React Compiler, code splitting, lazy loading, bundle watch, effect cleanup.
- [imports-and-modules.md](./imports-and-modules.md): named exports, no barrels, path aliases, the lint-enforced layer boundary, test mocking.
- [naming-conventions.md](./naming-conventions.md): files, variables, callbacks, constants, and the vocabulary that carries meaning.
- [formatting-and-i18n.md](./formatting-and-i18n.md): `Intl` for numbers, money, and dates, through `shared/helpers/format.ts`.
- [testing.md](./testing.md): Vitest conventions, the harnesses in `src/tests/`, the behavioral e2e suite, and the screenshot matrix.
