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
 * show whatever the entitlement says. See ARCHITECTURE.md for why the first two
 * stay two mechanisms rather than one. The third, `operatorOnly`, is not about
 * the deployment at all but about who is signed in, and it is answered by the
 * surface it gates rather than by a rule of its own.
 */

import { useMemo } from "react"

import { useDeploymentAdminAccess } from "@/shared/api/hooks"
import { useSurfaces } from "@/shared/hooks/useDeployment"
import { useEntitlements } from "@/shared/hooks/useEntitlements"

import type { NavItem } from "./types"

/**
 * A predicate over registry entries: true when this deployment hosts the
 * surface, is entitled to the capability, and (for an operator-only row) the
 * caller operates the deployment. An axis the entry does not declare is not a
 * gate. This is the *rail* predicate; the route behind a row is gated by
 * `useRouteVisibility`, which drops the caller axis for the reason stated there.
 *
 * Client-side only, so it can hide a destination and never grant one; the
 * server still authorizes every request the page behind it makes.
 */
export function useNavVisibility(): (item: NavItem) => boolean {
  const isRouteVisible = useRouteVisibility()
  // The caller axis. Asked unconditionally rather than only when a row declares
  // it, because a hook cannot be called from inside the predicate; it is one
  // cached read that every other row ignores. Undefined while it resolves and
  // false when the surface refused, so an operator-only row is absent until the
  // answer is yes, which is the safe direction for a destination that would
  // otherwise render its own refusal.
  const operator = useDeploymentAdminAccess()

  return useMemo(
    () => (item: NavItem) =>
      isRouteVisible(item) &&
      (item.operatorOnly === undefined || operator.data === true),
    [isRouteVisible, operator.data],
  )
}

/**
 * The two deployment-shaped axes, without the caller one.
 *
 * What decides whether the *route* renders, where `useNavVisibility` decides
 * whether the *rail row* does. The two answers differ on exactly one kind of
 * entry and for two reasons. An operator-only destination is answered by a
 * query, so composing it into the route gate would show the shell's "this
 * deployment does not serve that page" panel to a real operator for the length
 * of that request and then swap it for the page. And a caller who is genuinely
 * not an operator is not looking at a page this deployment does not serve: the
 * page is served, it is theirs to be refused, and it says so itself in words
 * that name the reason (`features/admin/DeploymentAccountsPage`). The shell's
 * panel would say something false instead.
 *
 * Both other axes are settled before the shell renders (the surface one from
 * the bootstrap) or settle without a page of their own to explain themselves
 * (the entitlement one, which `AppShell` waits on through `PendingPage`), which
 * is why they stay on the route gate.
 */
export function useRouteVisibility(): (item: NavItem) => boolean {
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
