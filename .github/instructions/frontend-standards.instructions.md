---
applyTo: "web/src/**/*.{ts,tsx,css},web/e2e/**/*.{ts,tsx},web/index.html,web/*.ts"
---

# Frontend Standards (dashboard)

These auto-apply when reviewing or editing the `web/` admin dashboard (React 19 with the React
Compiler, TypeScript strict, HeroUI v3, Tailwind v4, TanStack Query, Vitest, Playwright). The
full guidance, with worked examples grounded in this dashboard's code, lives in the skill:
[.github/skills/frontend-standards/SKILL.md](../skills/frontend-standards/SKILL.md).

## Non-negotiables

1. **HeroUI v3 only.** Unified `@heroui/react` import, compound components (`Card.Content`,
   `Card.Header`), `onChange` (not `onValueChange`), `onPress` (not `onClick`), and a `variant`
   (not `color`) on `Button`. No v2 patterns: granular `@heroui/*` imports, `HeroUIProvider`,
   `classNames={{ slot }}` objects, `bg-content1`. An internal link is TanStack Router's
   `<Link to>`; HeroUI's `<Link href>` is a full page reload.
   See [components.md](../skills/frontend-standards/components.md).

2. **Variables and props over hand-written CSS.** Start from a HeroUI component or a shared
   primitive rather than a native element, which arrives with the tokens and its states
   (pointer, focus ring, disabled dimming) already wired. Then four ways to change how it
   looks, in order: a variable (ours as a token, or one of HeroUI's own aliased onto ours;
   `--radius` drives its whole radius ramp, and `--cursor-interactive`, `--disabled-opacity`
   and the rest are in `@heroui/styles/dist/themes/`, none of them aliased in `globals.css`
   yet), a shared utility once the look repeats, the component's own prop (`variant`, `size`,
   `isDisabled`, `isPending`, `fullWidth`, `isInvalid`), and only then a rule reaching into
   HeroUI's DOM under the `.otari-*` namespace. A rule against an internal class
   (`.table__cell`, `.table__body tr:first-child td:first-child`) is a finding when a prop or
   a variable would do the same job, and so is a `className` re-skinning something the
   component already styles; reserve `className` for layout/position. Space siblings with
   `gap-*` on the parent, not `m-*` on children, and write arbitrary values in `rem`
   (`h-[20rem]`, not `h-[320px]`; a `1px` border is the exception).

3. **Color and type come from the semantic tokens** in `web/src/styles/globals.css`. The
   tokens are the design system; HeroUI and Tailwind consume it, so a utility that does not
   resolve to a `--color-*` token is outside the system and will not follow a retheme. Use the
   utilities the tokens back (`text-muted`, `bg-surface`, `border-border`, `text-danger`,
   `text-heading`), add a token there rather than scattering a hex, and add it to both theme
   blocks. A raw hex, a numbered Tailwind palette class, and `bg-white` / `text-black` all
   fail `web/src/styles/foundation.test.ts`, over the whole of `web/src`. Two pairings are
   easy to get wrong: a status word wears its own color on its own subtle fill
   (`text-danger` on `bg-danger-subtle`), but brand text on the brand tint takes
   `text-primary-subtle-foreground`, not `text-accent`. Type comes from the seven roles, so a
   `text-[11px]` or a `text-2xl font-bold` is a finding. See
   [design-tokens.md](../skills/frontend-standards/design-tokens.md).

4. **Server state goes through TanStack Query + `apiFetch`.** Fetch via the hooks in
   `web/src/shared/api/hooks.ts`; keep query keys as module constants, set a deliberate `staleTime`,
   and invalidate only the keys a mutation changes. Guard with `isPending && !data` (never bare
   `isPending`, never `isLoading`) and give a filtered or paginated query
   `placeholderData: (prev) => prev`, or the page blanks on every filter change. Don't call
   `fetch()` directly for authenticated management requests (the one exception is pre-auth
   `validateMasterKey`), never mirror server state into `useState`, and never swallow a
   mutation error. Bound every "fetch all" loop with a hard page cap. See
   [data-fetching.md](../skills/frontend-standards/data-fetching.md).

5. **TypeScript + React hygiene.** `undefined` (not `null`) for absent values in your own
   types; `unknown` plus a guard where a type is genuinely unknown, not `any`; a discriminated
   union rather than a bag of optionals; named exports and named imports, no barrel files;
   correct effect dependency arrays with cleanup; derive from props/query data rather than
   duplicating into state. The React Compiler is enabled, so hand-written
   `useMemo`/`useCallback`/`React.memo` needs a stated reason. See
   [typescript-and-react.md](../skills/frontend-standards/typescript-and-react.md) and
   [performance.md](../skills/frontend-standards/performance.md).

6. **New code lands in a layer.** A domain's page and the parts only it uses go in
   `web/src/features/<domain>/`; something no domain owns goes in `web/src/shared/`
   (`components/`, `helpers/`, `api/`); test harnesses go in `web/src/tests/`; only
   `web/src/app/` composes the tree. The layout mirrors `otari-ai/frontend/src`, so
   prefer its names for a new directory. A feature may not
   import `app/`, and `shared/` may not import `features/` or `app/`. `pnpm --dir web run lint` (Biome)
   rejects both, so flag placement in review rather than leaving it to the lint to reject
   after the fact. Adding a directory directly under `src/` needs a rule to go with it.

7. **A page composes; it does not also implement.** A dialog body, a second table, or a pure
   derivation added to a page file belongs in its own file in the same feature. A route file
   under `web/src/routes/` exports `Route` and nothing else, or `autoCodeSplitting` stops
   splitting it and the page lands in the entry chunk every visitor downloads. No IIFEs in
   JSX; no structural markup copy-pasted between files. See
   [component-architecture.md](../skills/frontend-standards/component-architecture.md).

8. **Navigation is data, and its three gates stay three.** A destination is declared in
   `web/src/app/nav/registry.ts` and nowhere else; flag a nav link hand-written into a
   component. An entry gates on `surface` (the deployment axis, from `GET /v1/bootstrap`
   via `useDeployment`), `capability` (the entitlement axis, via `useEntitlements`), and
   `flag` (the operational axis, valid only alongside a capability), composed as AND by
   `useNavVisibility`. Do not fold one into another, and do not reach past them: a page
   component that reads the gateway's mode itself, or infers it from an endpoint's 404, is
   the scattered mode check this replaced. A new base capability belongs in
   `BASE_CAPABILITIES`; an overlay's belongs in neither. Hiding a surface client-side is a
   convenience, never an authorization; the server still has to enforce it.

9. **Mobile is not optional.** The dashboard installs to a phone home screen and the shell
   already switches to a drawer below `md`. Touch targets ≥44px on the phone viewport, no
   hover-only controls (`opacity-0 group-hover:*` does not exist on a touch device), `min-w-0`
   on flex children that can overflow, no fixed-pixel layout containers, and a table needs a
   card list or its own horizontal scroll below `md`. See
   [responsiveness.md](../skills/frontend-standards/responsiveness.md).

10. **Tests for changed behavior.** Colocated Vitest tests (`Foo.tsx` → `Foo.test.tsx`) that
    query the way a user would (`getByRole`/`getByLabelText`/`getByText`, not `getByTestId`),
    render real providers, and mock only the network boundary (`apiFetch`), not the hooks. Each
    file restores the globals it overrode and carries no per-assertion timeout override. A new
    page also needs a screenshot entry in `web/e2e/screenshots/`, which is what will cover it
    at three viewports in both themes; that suite runs on demand today rather than as a PR
    gate, so the entry is owed even though nothing fails without it. See [testing.md](../skills/frontend-standards/testing.md).
