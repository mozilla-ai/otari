---
name: frontend-standards
description: Guidelines for the otari admin dashboard (`web/`), React 19 + TypeScript (strict) + HeroUI v3 + Tailwind v4 + TanStack Query. Use when writing or reviewing dashboard components, styling, data fetching, or tests.
---

# Frontend Standards: otari dashboard (`web/`)

`web/` is the standalone admin dashboard: a small React SPA that talks to the gateway's
management API with the master key. It is a focused operator tool, not a general-purpose app;
keep its footprint small and its conventions consistent with what is already there. There is
still no screenshot suite and no analytics; do not add them to match some other repo.

Stack: React 19, TypeScript (`strict`), HeroUI v3 (`@heroui/react`), Tailwind CSS v4,
TanStack Query, TanStack Router (file-based, `web/src/routes/`), Vite, Vitest + Testing
Library, Playwright (`web/e2e/`). Package manager is **npm**.

[web/AGENTS.md](../../../web/AGENTS.md) owns the structure and is worth reading first: the
`features/` / `shared/` / `app/` layout it mirrors from `otari-ai/frontend`, the three
lint-enforced import rules, the routing conventions, the generated API client, and the design
foundation. This file is the house style for writing code inside that structure.

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
- Style from the semantic tokens in `web/src/styles/globals.css`, through the utilities they
  back (`text-muted`, `bg-surface`, `border-border`, `text-danger`, `text-heading`). Never a raw
  hex, never a numbered Tailwind palette class, and never an `--otari-*` variable from below the
  `MIGRATION BRIDGE` marker. [design-tokens.md](./design-tokens.md) has the families, the HeroUI
  mapping, and the near-synonym utilities HeroUI ships that look like ours and are not.
- Fetch server state through the TanStack Query hooks in `web/src/shared/api/hooks.ts`, and bound
  every "fetch everything" walk with a hard page cap. See [data-fetching.md](./data-fetching.md)
  for query keys, `staleTime`, invalidation, and the `fetchAllPricing` shape.
- Prefer `undefined` over `null` for absent values in your own types (the API layer may return
  `null`; convert at the boundary). See [typescript-and-react.md](./typescript-and-react.md).
- Gate a deployment-dependent surface through `useDeployment()` / `useSurfaces()`, the one place
  that knows which deployment served the page. Mind the vocabulary: a *surface* is the
  deployment axis, a *capability* is the entitlement axis.
- Declare a new destination in the nav registry (`web/src/app/nav/registry.ts`), never as a
  hand-written link, with whichever of the three gates it needs (`surface`, `capability`,
  `flag`). `EntitlementGate` is the component form for wrapping a page. See
  [web/AGENTS.md](../../../web/AGENTS.md) for how the gates compose and where a capability
  the base build ships has to be declared.
- Add a Vitest test for any component or helper whose behavior you change (`Foo.tsx` →
  `Foo.test.tsx`, colocated). See [typescript-and-react.md](./typescript-and-react.md#testing).

**Never:**

- New HeroUI **v2** patterns: granular imports, `HeroUIProvider`, `classNames={{ slot }}`,
  `onValueChange`, or `color` on `Button`. v3 ignores some of these silently, which is why the
  full v2-to-v3 table is in [components.md](./components.md).
- Inline `style={{}}` or `<style>` tags. Use Tailwind utilities or a token.
- Manual polling with bare `setInterval`/`setTimeout`. Use TanStack Query's `refetchInterval`
  (see `useDashboardBuild`).
- A raw `fetch()` for **authenticated** management requests. Go through `apiFetch`, which owns the
  Bearer key, error extraction, and 401/403 sign-out. `validateMasterKey` is the one sanctioned
  exception, and [data-fetching.md](./data-fetching.md) says why.
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
