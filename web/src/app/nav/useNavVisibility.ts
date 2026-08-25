/**
 * Composes the two gates a registry entry can declare.
 *
 * The predicate form, because a section's heading has to disappear along with
 * its last visible item, and a wrapper component (`EntitlementGate`) cannot tell
 * the section that happened. Both read the same hooks, so there is one answer
 * per axis, not two.
 *
 * The axes are independent and compose as AND, in the order they are cheapest to
 * be wrong about: a deployment that does not host the surface has no page to
 * show whatever the entitlement says. See ARCHITECTURE.md for why they stay two
 * mechanisms rather than one.
 */

import { useMemo } from "react"

import { useSurfaces } from "@/shared/hooks/useDeployment"
import { useEntitlements } from "@/shared/hooks/useEntitlements"

import type { NavItem } from "./types"

/**
 * A predicate over registry entries: true when this deployment hosts the
 * surface and is entitled to the capability. An axis the entry does not
 * declare is not a gate.
 *
 * Client-side only, so it can hide a destination and never grant one; the
 * server still authorizes every request the page behind it makes.
 */
export function useNavVisibility(): (item: NavItem) => boolean {
  const hostsSurface = useSurfaceVisibility()
  const { capabilities } = useEntitlements()

  return useMemo(() => {
    const entitled = new Set(capabilities)
    return (item: NavItem) =>
      hostsSurface(item) &&
      (item.capability === undefined || entitled.has(item.capability))
  }, [hostsSurface, capabilities])
}

/**
 * The deployment half of that predicate, alone.
 *
 * For the one caller that has to tell the two axes apart rather than compose
 * them: the entitlement axis can still be resolving while the deployment axis
 * has been settled since the bootstrap, so a surface this gateway does not host
 * is an answer available immediately, and `AppShell` says so rather than making
 * the reader wait on a query that cannot change it.
 *
 * Exported beside its composition rather than rebuilt at that call site, so
 * there is still one implementation of the axis and the two cannot disagree
 * about a destination.
 */
export function useSurfaceVisibility(): (item: NavItem) => boolean {
  // Through `useSurfaces` rather than reading `surfaces` off the bootstrap
  // directly, so the deployment axis has one implementation and a feature that
  // asks it outside the sidebar gets the same answer.
  const hostsSurface = useSurfaces()

  return useMemo(
    () => (item: NavItem) =>
      item.surface === undefined || hostsSurface(item.surface),
    [hostsSurface],
  )
}
