import type { ReactNode } from "react"

/**
 * Where the entitlement axis is resolved, and in this build it resolves to the
 * constant.
 *
 * `useEntitlements.tsx` describes the base answer as a build-time constant and
 * an overlay's as "its own answer by rendering `EntitlementProvider` above the
 * shell". This module is that mount point. Without it the sentence described a
 * place that did not exist: every seam this repository has contributes *inside*
 * the shell, so a build whose routes are entitled and serving had nowhere to
 * feed the server's answer in, and a nav row or page tagged with an overlay
 * capability hid in every build including that one (otari#758).
 *
 * The base default renders its children unchanged. That is the whole of it, and
 * it is doing something: with no provider here, every consumer falls through to
 * the context default in `useEntitlements.tsx`, which is `BASE_CAPABILITIES`.
 * Rendering `EntitlementProvider` with that same constant would look equivalent
 * and is not, because a provider shadows whatever is above it: a superset build
 * that wrapped the app in its own provider, and every test that does exactly
 * that today, would get the empty base answer back from this line.
 *
 * One export, and its shape is the contract: a component named
 * `EntitlementResolver` taking `{ children }`. A replacement fetches
 * `GET /v1/entitlements` (the superset serves it, resolved through
 * `EntitlementPort` per request; nothing in this repository serves that path)
 * and renders `EntitlementProvider` with what came back.
 *
 * **It should report `isLoading` rather than hold its children back**, and the
 * shell is built to be told. The two halves the axis gates read that state
 * differently, which is worth knowing before choosing: the rail is forgiving,
 * since `useNavVisibility` reads capabilities only, so a still-resolving answer
 * draws the base rows first and the contributed ones when they arrive. The route
 * gate is a predicate with no `loading` prop to pass, unlike `EntitlementGate`,
 * so it is `AppShell` that carries the branch: while `isLoading` is set it
 * renders the router's pending copy instead of the panel asserting the
 * deployment does not serve the page, which is a claim nothing can make yet. A
 * replacement that instead renders nothing until its query settles blanks the
 * whole dashboard on every load, which is why the shell takes the state rather
 * than making each replacement work around it.
 *
 * **Why a seam and not a probe in the base.** The alternative is for this
 * repository to request `/v1/entitlements` itself and fall back to the constant
 * on 404. It is fewer moving parts, and it is wrong here for a reason stronger
 * than taste: this gateway mounts no such route (`EntitlementPort` exists, and
 * `require_capability` is its only caller), so the request 404s on every load of
 * every OSS deployment. That is a wire path in the base for an endpoint the base
 * does not serve, plus a hand-written fetch, since `src/client/schema.ts` is
 * generated from `docs/public/openapi.json` and has no operation to type it
 * with. The constant is not a placeholder standing in for a fetch: it is the
 * honest core adapter ARCHITECTURE.md describes, and a build that resolves the
 * axis from a server is the build that owns the request for it.
 *
 * **Reached by its `@/app/overlayEntitlementResolver` specifier and never
 * relatively**, which is the seam rule rather than a style call:
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why. That specifier
 * is the alias key an `OVERLAY_MODULE_OVERRIDES` entry has to match, and it is
 * the path otari#758 names, so a replacement keyed on it lands. The module sits
 * in `src/app/` rather than beside `useEntitlements.tsx` in `src/shared/hooks/`
 * because it is composition and not a hook: the shell mounts it, in the way
 * `App.tsx` mounts `DeploymentProvider`.
 */
export function EntitlementResolver({ children }: { children: ReactNode }) {
  return <>{children}</>
}
