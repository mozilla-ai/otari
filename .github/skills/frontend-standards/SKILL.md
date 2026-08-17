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
TanStack Query, TanStack Router (file-based, `web/src/routes/`), Vite, Vitest + Testing
Library, Playwright (`web/e2e/`). Package manager is **npm**. Layout mirrors
`otari-ai/frontend/src`, whose control-plane UI moves into this repo at M5:
`web/src/features/<domain>/` for a domain's page and the parts only it uses, `shared/`
for what no domain owns (`components/`, `helpers/`, `api/`), `app/` for the composition
root, plus `routes/`, `tests/` and the generated `client/`. Three import rules are lint-enforced
(a feature may not import `app/`; `shared/` may not import `features/` or `app/`;
nothing imports an overlay tree): run `npm --prefix web run lint`, and read the Layout
bullet in [web/AGENTS.md](../../../web/AGENTS.md) before adding a directory. There is
still no screenshot suite and no analytics; do not add them to match some other repo.
Routing conventions (one file per URL, a route
file exports `Route` and nothing else, the search-param codec, and `renderWithRouter` for
component tests) and the generated API client (`web/src/client`, regenerated from the
OpenAPI spec, never hand-edited) live in [web/AGENTS.md](../../../web/AGENTS.md).

Build and check from the repo root:

- `make dashboard` (`npm --prefix web ci && npm --prefix web run build`). Output goes to
  the gitignored `src/gateway/static/dashboard/`; there is nothing to commit. Build only
  when you need to run the dashboard locally; Docker builds it in its own Node stage.
- `npm --prefix web run lint` (the layer boundaries; see web/AGENTS.md)
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
- Fetch server state through TanStack Query hooks in `web/src/shared/api/hooks.ts`. Keep query keys
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
  `onChange`, `onPress`, and a `variant` (the dashboard uses `primary`, `ghost`, `outline`,
  `danger`, `danger-soft`) instead of `color` on `Button`.
- Inline `style={{}}` or `<style>` tags. Use Tailwind utilities or a token.
- Manual polling with bare `setInterval`/`setTimeout`. Use TanStack Query's `refetchInterval`
  (see `useDashboardBuild`).
- A raw `fetch()` for **authenticated** management requests. Go through `apiFetch` in
  `web/src/shared/api/client.ts`, so the Bearer key, error extraction, and 401/403 sign-out handling
  stay in one place. (The sanctioned exception is `validateMasterKey`, which raw-`fetch`es a
  master-key-gated endpoint to check a candidate key *before* it becomes the stored `masterKey`.)
- Client-side filtering/sorting/pagination of large server datasets when the endpoint can do
  it. (Small, already-loaded lists rendered in a `Table` are fine.)
- `getByTestId` when a semantic query (`getByRole`, `getByLabelText`, `getByText`) works.

## A note on status colors

`web/` uses raw Tailwind palette classes for one narrow case: semantic status surfaces. The
sanctioned triad is `red` for danger/error (`ErrorBanner` uses `border-red-200 bg-red-50
text-red-700`), `amber` for warning (`InfoBanner`), and `emerald` for healthy/success
(`StatCard` status pills and the overview all-clear strip). That is the existing convention, so
match it for new alert/status elements rather than reformatting them into `--otari-*` tokens.
Everything else (brand, surface, text, borders) uses the `--otari-*` variables. Don't reach for
`bg-white`/`text-gray-900` for general chrome.

## Topic guides

- [design-tokens.md](./design-tokens.md): the `--otari-*` variables, where they live, when to add one.
- [components.md](./components.md): HeroUI v3 patterns, props over `className`, the shared UI primitives in `shared/components/ui.tsx`.
- [data-fetching.md](./data-fetching.md): TanStack Query conventions: query keys, `staleTime`, invalidation, bounded pagination.
- [typescript-and-react.md](./typescript-and-react.md): strict TS, `undefined` over `null`, hook/effect hygiene, and Vitest testing.
