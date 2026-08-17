/**
 * The entitlement and feature-flag axes, and the seam an overlay resolves them
 * through.
 *
 * Two of the three gates a surface composes (ARCHITECTURE.md, "Deployment,
 * entitlements, and feature flags"):
 *
 * - **Entitlement** is the licensing axis: is this capability enabled for this
 *   deployment or org at all? Scoped per deployment or org, never per user, and
 *   a standing property rather than a switch.
 * - **Feature flag** is the operational axis: is this sub-feature of a
 *   capability I already hold turned on right now? Engineering's rollout
 *   switch, retired once the rollout lands.
 *
 * They are never folded together, and neither is the deployment axis in
 * `useDeployment` (that one answers whether the process serving this page hosts
 * the surface at all).
 *
 * **The base build answers both from a constant**, which is what the context's
 * default value is. That is the honest core adapter ARCHITECTURE.md describes:
 * it grants every capability the base ships and reports every overlay-only one
 * as absent, and the base ships no feature flags because a flag belongs to
 * whoever is rolling something out. An overlay, or a later release that resolves
 * these from the server, supplies its own answer by rendering
 * `EntitlementProvider` above the shell; nothing that reads a capability changes.
 *
 * The hooks return `{ entitled, isLoading }` / `{ enabled, isLoading }` rather
 * than a bare boolean, matching `otari-ai/frontend/src/shared/hooks`, where both
 * are TanStack queries. `isLoading` is always false here. Keeping the shape is
 * what lets the resolver become asynchronous later without touching a call site,
 * and it is why `EntitlementGate` has a loading state to render at all.
 */

import type { ReactNode } from "react"
import { createContext, useContext } from "react"

/** What the two axes resolve to for the current deployment. */
export interface Entitlements {
  /** Capability names this deployment or org is entitled to. */
  capabilities: readonly string[]
  /** Evaluated feature flags, by key. An absent key is off. */
  flags: Readonly<Record<string, boolean>>
  /** Whether either is still resolving. Always false for the base answer. */
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
 */
export const BASE_CAPABILITIES: readonly string[] = []

const BASE_ENTITLEMENTS: Entitlements = {
  capabilities: BASE_CAPABILITIES,
  flags: {},
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

/** Everything both axes resolved to. Prefer the two single-name hooks below. */
export function useEntitlements(): Entitlements {
  return useContext(EntitlementContext)
}

/** Whether this deployment or org is entitled to a capability. */
export function useEntitlement(capability: string): {
  entitled: boolean
  isLoading: boolean
} {
  const { capabilities, isLoading } = useEntitlements()
  return { entitled: capabilities.includes(capability), isLoading }
}

/** Whether a feature flag is on. An unknown key is off. */
export function useFeatureFlag(flag: string): {
  enabled: boolean
  isLoading: boolean
} {
  const { flags, isLoading } = useEntitlements()
  return { enabled: flags[flag] ?? false, isLoading }
}
