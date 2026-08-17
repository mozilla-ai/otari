---
applyTo: "web/src/**/*.{ts,tsx,css}"
---

# Frontend Standards (dashboard)

These auto-apply when reviewing or editing the `web/` admin dashboard (React 19, TypeScript
strict, HeroUI v3, Tailwind v4, TanStack Query, Vitest). The full guidance, with worked
examples grounded in this dashboard's code, lives in the skill:
[.github/skills/frontend-standards/SKILL.md](../skills/frontend-standards/SKILL.md).

## Non-negotiables

1. **HeroUI v3 only.** Unified `@heroui/react` import, compound components (`Card.Content`,
   `Card.Header`), `onChange` (not `onValueChange`), `onPress` (not `onClick`), and a `variant`
   (not `color`) on `Button`. No v2 patterns: granular `@heroui/*` imports, `HeroUIProvider`,
   `classNames={{ slot }}` objects. See [components.md](../skills/frontend-standards/components.md).

2. **Props over `className`.** Use a component's prop (`variant`, `size`, `isDisabled`,
   `isPending`, `fullWidth`, `isInvalid`) before a `className`; reserve `className` for
   layout/position. Space siblings with `gap-*` on the parent, not `m-*` on children.

3. **Color comes from the `--otari-*` tokens** in `web/src/styles/globals.css`
   (`text-[var(--otari-muted)]`, `bg-[var(--otari-surface)]`). Add a token there rather than
   scattering a hex. Raw Tailwind palette classes are only for status surfaces (`ErrorBanner`,
   `InfoBanner`). See [design-tokens.md](../skills/frontend-standards/design-tokens.md).

4. **Server state goes through TanStack Query + `apiFetch`.** Fetch via the hooks in
   `web/src/shared/api/hooks.ts`; keep query keys as module constants, set a deliberate `staleTime`,
   and invalidate only the keys a mutation changes. Don't call `fetch()` directly for
   authenticated management requests (the one exception is pre-auth `validateMasterKey`), and
   never mirror server state into `useState`. Bound every "fetch all" loop with a hard page
   cap. See [data-fetching.md](../skills/frontend-standards/data-fetching.md).

5. **TypeScript + React hygiene.** `undefined` (not `null`) for absent values in your own
   types; named exports and named imports; correct effect dependency arrays with cleanup;
   derive from props/query data rather than duplicating into state. See
   [typescript-and-react.md](../skills/frontend-standards/typescript-and-react.md).

6. **New code lands in a layer.** A domain's page and the parts only it uses go in
   `web/src/features/<domain>/`; something no domain owns goes in `web/src/shared/`
   (`components/`, `helpers/`, `api/`); test harnesses go in `web/src/tests/`; only
   `web/src/app/` composes the tree. The layout mirrors `otari-ai/frontend/src`, so
   prefer its names for a new directory. A feature may not
   import `app/`, and `shared/` may not import `features/` or `app/`. `npm run lint` (Biome)
   rejects both, so flag placement in review rather than leaving it to the lint to reject
   after the fact. Adding a directory directly under `src/` needs a rule to go with it.

7. **Deployment differences come from the bootstrap.** A surface that only some deployments
   host gates on `useSurfaces()` / `useDeployment()` (`web/src/shared/hooks/useDeployment.tsx`),
   which `main.tsx` fills from `GET /v1/bootstrap` before the first render. Flag a page component
   that reads the gateway's mode itself, or infers it from an endpoint's 404: that is the
   scattered mode check the bootstrap replaced. Hiding a surface client-side is a convenience,
   never an authorization; the server still has to enforce it.

8. **Tests for changed behavior.** Colocated Vitest tests (`Foo.tsx` → `Foo.test.tsx`) that
   query the way a user would (`getByRole`/`getByLabelText`/`getByText`, not `getByTestId`),
   render real providers, and mock only the network boundary (`apiFetch`), not the hooks.
