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
 * server reporting the caller's own standing rather than by a rule of its own.
 */

import { useMemo } from "react"

import { isDeploymentOperator } from "@/features/organization/roles"
import { useOrganizationContext } from "@/shared/api/hooks"
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
  // The caller axis, and it rides on the organization context rather than on a
  // query of its own: `GET /v1/organizations/me` carries `deployment_operator`,
  // the same answer `/v1/admin/access` gives from the same server-side
  // predicate, and the shell reads that context anyway to decide whether to
  // offer the way into the organization rail. So the answer is here when the
  // chrome is, and no row is drawn on the strength of a question still in flight
  // (#836). Read for every row rather than only the ones that declare the axis,
  // because a hook cannot be called from inside the predicate; it is one cached
  // read the rest of them ignore.
  //
  // It also needs no deployment gate of its own any more. The request this used
  // to make had to be withheld from a gateway that does not host `/v1/admin`, so
  // its 404 could not become a second reading of `surfaces`; this read is not
  // that request, and each operator-only row still declares the surface it
  // needs, which `isRouteVisible` composes below.
  const organization = useOrganizationContext()
  // Through `roles.isDeploymentOperator` rather than reading the field, so the
  // client keeps one spelling of the predicate for the same reason the server
  // does: it already requires an explicit `true`, so an older gateway that omits
  // the field reads as not an operator rather than as an answer.
  const isOperator = isDeploymentOperator(organization.data)
  const answerUnavailable = organization.isError

  return useMemo(() => {
    // Both values wait for an explicit yes, so neither kind of row is shown and
    // then taken away. What still separates them is what a *failed* read means,
    // and the server's own refusal is what decides that: an "unlisted" row is
    // one the server 404s, so with no answer the rail may not reveal it either,
    // while a "refused" row is one the server 403s, whose existence is no secret
    // and whose destinations stay reachable by URL regardless. See `types.ts`
    // for the full note.
    const allowedByCaller = (item: NavItem) => {
      if (item.operatorOnly === undefined) {
        return true
      }
      return item.operatorOnly === "unlisted"
        ? isOperator
        : isOperator || answerUnavailable
    }

    return (item: NavItem) => isRouteVisible(item) && allowedByCaller(item)
  }, [isRouteVisible, isOperator, answerUnavailable])
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
