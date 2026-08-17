import type { ReactNode } from "react"

import { useEntitlement, useFeatureFlag } from "@/shared/hooks/useEntitlements"

interface EntitlementGateProps {
  /** The capability this deployment or org must be entitled to. */
  capability: string
  /**
   * An operational feature flag that must also be on. Entitlement (the
   * licensing axis) and flag (the operational axis) compose as
   * `entitled AND enabled`; they are never merged into one check.
   */
  flag?: string
  /** Rendered once the surface is both entitled and, if given, flagged on. */
  children: ReactNode
  /** Rendered when the surface is not entitled, or not flagged on. */
  fallback?: ReactNode
  /**
   * Rendered while the two axes are still resolving; defaults to `fallback`, so
   * a gate hides rather than flashes. Give a full-page gate whose fallback
   * asserts something ("not available here") its own loading state, so an
   * entitled user is never shown a false negative during a cold load. The base
   * build resolves synchronously and never renders this, but an overlay that
   * resolves from the server will.
   */
  loading?: ReactNode
}

/**
 * Renders its children only for a deployment entitled to `capability` and, when
 * a `flag` is given, only while that flag is on.
 *
 * The component form of the two gates in `useEntitlements`, for wrapping a page
 * or a block of markup. The sidebar does not use it: a section's heading has to
 * disappear along with its last visible item, which needs the gates as a
 * predicate rather than as a wrapper, so `app/nav/useNavVisibility.ts` composes
 * the same hooks instead.
 *
 * Hiding a surface here is a client-side convenience and never an
 * authorization. The server authorizes every request the children make,
 * whatever this decides.
 */
export function EntitlementGate({
  capability,
  flag,
  children,
  fallback,
  loading,
}: EntitlementGateProps) {
  const { entitled, isLoading: entitlementLoading } = useEntitlement(capability)
  // Reading the flag unconditionally is safe because both axes come from one
  // context; there is no request to avoid making, unlike otari.ai's version,
  // which has to keep an entitlement-only gate off the flags endpoint.
  const { enabled, isLoading: flagLoading } = useFeatureFlag(flag ?? "")

  if (entitlementLoading || (flag !== undefined && flagLoading)) {
    return <>{loading ?? fallback}</>
  }
  const flagSatisfied = flag === undefined ? true : enabled
  return <>{entitled && flagSatisfied ? children : fallback}</>
}
