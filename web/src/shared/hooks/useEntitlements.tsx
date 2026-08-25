/**
 * The entitlement axis, and the seam an overlay resolves it through.
 *
 * One of the two gates a surface composes (ARCHITECTURE.md, "Deployment and
 * entitlements"). **Entitlement** is the licensing axis: is this capability
 * enabled for this deployment at all? Scoped per deployment, never per user,
 * and a standing property rather than a switch.
 *
 * It is never folded together with the deployment axis in `useDeployment`
 * (that one answers whether the process serving this page hosts the surface at
 * all).
 *
 * **The base build answers from a constant**, which is what the context's
 * default value is. That is the honest core adapter ARCHITECTURE.md describes:
 * it grants every capability the base ships and reports every overlay-only one
 * as absent. An overlay, or a later release that resolves this from the server,
 * supplies its own answer by rendering `EntitlementProvider` above the shell;
 * nothing that reads a capability changes.
 *
 * The hook returns `{ entitled, isLoading }` rather than a bare boolean,
 * matching `otari-ai/frontend/src/shared/hooks`, where it is a TanStack query.
 * `isLoading` is always false here. Keeping the shape is what lets the resolver
 * become asynchronous later without touching a call site, and it is why
 * `EntitlementGate` has a loading state to render at all.
 */

import type { ReactNode } from "react"
import { createContext, useContext } from "react"

/** What the entitlement axis resolves to for the current deployment. */
export interface Entitlements {
  /** Capability names this deployment is entitled to. */
  capabilities: readonly string[]
  /** Whether it is still resolving. Always false for the base answer. */
  isLoading: boolean
}

/**
 * The capabilities Otari's base build ships and therefore entitles.
 *
 * **Empty, because nothing in the base registry is gated on a capability yet.**
 * That is not an oversight: the one candidate is routing, and ARCHITECTURE.md
 * marks how far the core base extends before an overlay adapter takes over as
 * provisional and not a contributor's to assume. So the base withholds nothing
 * and declares nothing, and the axis waits for a real decision instead of
 * anticipating one.
 *
 * Add a name here when the base grows a capability with UI surface, at the same
 * time as the nav entry that gates on it; leave an overlay-only one (billing,
 * for example) out, which is what makes a gate on it hide in this build. A nav
 * entry gated on a capability that is not listed disappears from every
 * deployment, and the two lists live in different files, so `registry.test.ts`
 * fails when they disagree.
 *
 * This has a server-side twin in `src/gateway/adapters/entitlement_adapter.py`,
 * which is what gates a route rather than a link. The two are meant to agree,
 * and nothing checks that they do, so a capability the base grows is added to
 * both at once.
 */
export const BASE_CAPABILITIES: readonly string[] = []

const BASE_ENTITLEMENTS: Entitlements = {
  capabilities: BASE_CAPABILITIES,
  isLoading: false,
}

// The default value is the seam: with no provider above it, every consumer gets
// the base answer, so the base build wires nothing up and an overlay wires up
// one component.
const EntitlementContext = createContext<Entitlements>(BASE_ENTITLEMENTS)

export function EntitlementProvider({
  value,
  children,
}: {
  value: Entitlements
  children: ReactNode
}) {
  return (
    <EntitlementContext.Provider value={value}>
      {children}
    </EntitlementContext.Provider>
  )
}

/** Everything the entitlement axis resolved to. Prefer `useEntitlement`. */
export function useEntitlements(): Entitlements {
  return useContext(EntitlementContext)
}

/** Whether this deployment is entitled to a capability. */
export function useEntitlement(capability: string): {
  entitled: boolean
  isLoading: boolean
} {
  const { capabilities, isLoading } = useEntitlements()
  return { entitled: capabilities.includes(capability), isLoading }
}
