---
name: frontend-standards
description: Guidelines for the otari admin dashboard (`web/`), React 19 + TypeScript (strict) + HeroUI v3 + Tailwind v4 + TanStack Query. Use when writing or reviewing dashboard components, styling, data fetching, or tests.
---

# Frontend Standards: otari dashboard (`web/`)

`web/` is the standalone admin dashboard: a small React SPA that talks to the gateway's
management API (`/v1/models`, `/v1/pricing`, `/v1/aliases`, `/v1/settings`) with the master
key. It is a focused operator tool, not a general-purpose app; keep its footprint small and
its conventions consistent with what is already there.

Stack: React 19, TypeScript (`strict`), HeroUI v3 (`@heroui/react`), Tailwind CSS v4,
TanStack Query, Vite, Vitest + Testing Library. Package manager is **npm**. Layout is flat
(`web/src/{components,pages,api,lib,auth}`), there is no `features/`/`shared/` architecture,
no router, no Playwright/screenshot suite, and no analytics. Do not introduce those to match
some other repo; match what `web/` does.

Build and check from the repo root:

- `npm --prefix web ci && npm --prefix web run build` (output is committed to
  `src/gateway/static/dashboard/`; rebuild and commit after any change under `web/src`).
- `npm --prefix web run typecheck`
- `npm --prefix web test`

## Critical rules

**Always:**

- Reach for a HeroUI component prop (`variant`, `size`, `isDisabled`, `isPending`,
  `fullWidth`, `isInvalid`) before a `className`. `className` is for layout/position, not for
  restyling a component HeroUI already styles. See [components.md](./components.md).
- Use the `--otari-*` CSS variables for brand/surface/text color
  (`text-[var(--otari-muted)]`, `bg-[var(--otari-brand-tint)]`), defined once in
  `web/src/styles/globals.css`. If you need a new brand/surface color, add a token there
  rather than scattering a hex. See [design-tokens.md](./design-tokens.md).
- Fetch server state through TanStack Query hooks in `web/src/api/hooks.ts`. Keep query keys
  as module constants, set a deliberate `staleTime`, and invalidate the affected keys in a
  mutation's `onSuccess`. See [data-fetching.md](./data-fetching.md).
- Bound every paginated read. `fetchAllPricing` walks pages behind a hard `PRICING_MAX_PAGES`
  cap so a backend that ignores `skip` can't spin an unbounded loop, copy that shape for any
  new "fetch everything" hook.
- Prefer `undefined` over `null` for absent values in your own types (the API layer may return
  `null`; convert at the boundary).
- Add a Vitest test for any component or helper whose behavior you change (`Foo.tsx` →
  `Foo.test.tsx`, colocated). See [typescript-and-react.md](./typescript-and-react.md#testing).

**Never:**

- New HeroUI **v2** patterns: granular imports (`@heroui/button`), `HeroUIProvider`,
  `classNames={{ slot }}` objects, `onValueChange` on inputs, or `color` on `Button`. This is
  v3: unified `@heroui/react` import, compound components (`Card.Content`, `Card.Header`),
  `onChange`, `onPress`, `variant="primary"|"secondary"|"ghost"|"danger"|"danger-soft"`.
- Inline `style={{}}` or `<style>` tags. Use Tailwind utilities or a token.
- Manual polling with bare `setInterval`/`setTimeout`. Use TanStack Query's `refetchInterval`
  (see `useDashboardBuild`).
- A raw `fetch()` for the management API. Go through `apiFetch` in `web/src/api/client.ts`, so
  the Bearer key, error extraction, and 401/403 sign-out handling stay in one place.
- Client-side filtering/sorting/pagination of large server datasets when the endpoint can do
  it. (Small, already-loaded lists rendered in a `Table` are fine.)
- `getByTestId` when a semantic query (`getByRole`, `getByLabelText`, `getByText`) works.

## A note on status colors

`web/` uses raw Tailwind palette classes for one narrow case: semantic status surfaces
(`ErrorBanner` uses `border-red-200 bg-red-50 text-red-700`; `InfoBanner` uses `amber`). That
is the existing convention, so match it for new alert/status elements rather than reformatting
them into `--otari-*` tokens. Everything else (brand, surface, text, borders) uses the
`--otari-*` variables. Don't reach for `bg-white`/`text-gray-900` for general chrome.

## Topic guides

- [design-tokens.md](./design-tokens.md): the `--otari-*` variables, where they live, when to add one.
- [components.md](./components.md): HeroUI v3 patterns, props over `className`, the shared UI primitives in `components/ui.tsx`.
- [data-fetching.md](./data-fetching.md): TanStack Query conventions: query keys, `staleTime`, invalidation, bounded pagination.
- [typescript-and-react.md](./typescript-and-react.md): strict TS, `undefined` over `null`, hook/effect hygiene, and Vitest testing.
