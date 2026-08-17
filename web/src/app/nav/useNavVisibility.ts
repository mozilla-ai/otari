/**
 * Composes the three gates a registry entry can declare.
 *
 * The predicate form, because a section's heading has to disappear along with
 * its last visible item, and a wrapper component (`EntitlementGate`) cannot tell
 * the section that happened. Both read the same hooks, so there is one answer
 * per axis, not two.
 *
 * The axes are independent and compose as AND, in the order they are cheapest to
 * be wrong about: a deployment that does not host the surface has no page to
 * show whatever the entitlement says, an unentitled capability is off however the
 * flag is set, and the flag narrows a capability already held. See
 * ARCHITECTURE.md for why they stay three mechanisms rather than one.
 */

import { useMemo } from "react"

import { useSurfaces } from "@/shared/hooks/useDeployment"
import { useEntitlements } from "@/shared/hooks/useEntitlements"

import type { NavItem } from "./types"

/**
 * A predicate over registry entries: true when this deployment hosts the
 * surface, is entitled to the capability, and has the flag on. An axis the
 * entry does not declare is not a gate.
 *
 * Client-side only, so it can hide a destination and never grant one; the
 * server still authorizes every request the page behind it makes.
 */
export function useNavVisibility(): (item: NavItem) => boolean {
  // Through `useSurfaces` rather than reading `surfaces` off the bootstrap
  // directly, so the deployment axis has one implementation and a feature that
  // asks it outside the sidebar gets the same answer.
  const hostsSurface = useSurfaces()
  const { capabilities, flags } = useEntitlements()

  return useMemo(() => {
    const entitled = new Set(capabilities)
    return (item: NavItem) =>
      (item.surface === undefined || hostsSurface(item.surface)) &&
      (item.capability === undefined || entitled.has(item.capability)) &&
      (item.flag === undefined || (flags[item.flag] ?? false))
  }, [hostsSurface, capabilities, flags])
}
