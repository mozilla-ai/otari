import type { ReactNode } from "react"

import { useEntitlement } from "@/shared/hooks/useEntitlements"

interface EntitlementGateProps {
  /** The capability this deployment must be entitled to. */
  capability: string
  /** Rendered once the surface is entitled. */
  children: ReactNode
  /** Rendered when the surface is not entitled. */
  fallback?: ReactNode
  /**
   * Rendered while the axis is still resolving; defaults to `fallback`, so a
   * gate hides rather than flashes. Give a full-page gate whose fallback
   * asserts something ("not available here") its own loading state, so an
   * entitled user is never shown a false negative during a cold load. The base
   * build resolves synchronously and never renders this, but an overlay that
   * resolves from the server will.
   */
  loading?: ReactNode
}

/**
 * Renders its children only for a deployment entitled to `capability`.
 *
 * The component form of the gate in `useEntitlements`, for wrapping a page or a
 * block of markup. The sidebar does not use it: a section's heading has to
 * disappear along with its last visible item, which needs the gate as a
 * predicate rather than as a wrapper, so `app/nav/useNavVisibility.ts` composes
 * the same hooks instead.
 *
 * Hiding a surface here is a client-side convenience and never an
 * authorization. The server authorizes every request the children make,
 * whatever this decides.
 */
export function EntitlementGate({
  capability,
  children,
  fallback,
  loading,
}: EntitlementGateProps) {
  const { entitled, isLoading } = useEntitlement(capability)

  if (isLoading) {
    return <>{loading ?? fallback}</>
  }
  return <>{entitled ? children : fallback}</>
}
