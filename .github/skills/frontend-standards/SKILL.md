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
- `npm --prefix web run lint` (Biome: formatting, recommended rules, layer boundaries; `lint:fix` writes)
- `npm --prefix web run typecheck`
- `npm --prefix web test`

## Critical rules

**Always:**

- Reach for a HeroUI component prop (`variant`, `size`, `isDisabled`, `isPending`,
  `fullWidth`, `isInvalid`) before a `className`. `className` is for layout/position, not for
  restyling a component HeroUI already styles. See [components.md](./components.md).
- Style from the semantic tokens in `web/src/styles/globals.css`, through the Tailwind
  utilities they back (`text-muted`, `bg-surface`, `border-border`, `bg-surface-alt`,
  `text-danger`, `text-heading`). They are the design foundation rehomed from
  `otari-ai/frontend`, and the HeroUI variable mapping in that file is what makes a bare
  `<Card>` or `<Chip>` wear the palette with no `className`. Never a raw hex, never a
  numbered Tailwind palette class. The `--otari-*` variables below the `MIGRATION BRIDGE`
  marker are the pre-rehome palette: existing call sites keep them, new code must not
  reach for one. See [design-tokens.md](./design-tokens.md).
- Fetch server state through TanStack Query hooks in `web/src/shared/api/hooks.ts`. Keep query keys
  as module constants, set a deliberate `staleTime`, and invalidate the affected keys in a
  mutation's `onSuccess`. See [data-fetching.md](./data-fetching.md).
- Bound every paginated read. `fetchAllPricing` walks pages behind a hard `PRICING_MAX_PAGES`
  cap so a backend that ignores `skip` can't spin an unbounded loop, copy that shape for any
  new "fetch everything" hook.
- Prefer `undefined` over `null` for absent values in your own types (the API layer may return
  `null`; convert at the boundary).
- Gate a deployment-dependent surface on the bootstrap, through `useDeployment()` /
  `useSurfaces()` (`web/src/shared/hooks/useDeployment.tsx`). It is the one place that
  knows which deployment served the page, and it is read before the first render. Note the
  word: a *surface* is the deployment axis, a *capability* is the entitlement axis.
- Declare a new destination in the nav registry (`web/src/app/nav/registry.ts`), never as a
  hand-written link in a component, and give it whichever of the three gates it needs:
  `surface`, `capability`, `flag`. `useNavVisibility` composes them as AND for the sidebar;
  `EntitlementGate` (`web/src/shared/components/EntitlementGate.tsx`) is the component form
  for wrapping a page. A capability the base build ships goes in `BASE_CAPABILITIES`
  (`web/src/shared/hooks/useEntitlements.tsx`); an overlay-only one goes in neither, which
  is what makes a gate on it hide here.
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
- A page component that branches on the gateway's mode (reading `/v1/settings`'s `mode`, or
  probing whether a management endpoint 404s). The deployment bootstrap answers that once,
  for the whole shell; see the rule above.

## A note on status colors

The foundation names the status roles, so use them: `text-danger` / `bg-danger-subtle` for
errors, `text-warning` / `bg-warning-subtle` for caution, `text-success` / `bg-success-subtle`
for healthy, `text-info` / `bg-info-subtle` for neutral notices, and the separate
`bg-attention` family for "look here" (a required action, an unread marker) as distinct from
"be careful". Every one of them is defined for both themes.

The bridge components (`ErrorBanner` with `border-red-200 bg-red-50 text-red-700`,
`InfoBanner` with `amber`, `StatCard`'s `emerald` status pills) still use raw Tailwind palette
classes. That is the palette this dashboard had before the rehome, not a sanctioned exception:
leave those call sites alone until their pages are rebuilt, and don't copy the pattern into
anything new.

## Topic guides

- [design-tokens.md](./design-tokens.md): the semantic tokens, the HeroUI mapping, the type scale, and the `--otari-*` bridge on its way out.
- [components.md](./components.md): HeroUI v3 patterns, props over `className`, the shared UI primitives in `shared/components/ui/`.
- [data-fetching.md](./data-fetching.md): TanStack Query conventions: query keys, `staleTime`, invalidation, bounded pagination.
- [typescript-and-react.md](./typescript-and-react.md): strict TS, `undefined` over `null`, hook/effect hygiene, and Vitest testing.
