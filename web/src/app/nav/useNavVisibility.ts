/**
 * Composes the gates a registry entry can declare.
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
 * `reachableWithoutOrganization` rides on the same read: its 404 says the
 * caller belongs to no organization, and then only the rows that opt in show.
 *
 * The caller axis also decides who may open a page at all (`useRouteAccess`):
 * a page someone may not use is in none of their navigation, and typing its
 * address goes Home rather than to a refusal.
 */

import { type AnyRouter, useRouter } from "@tanstack/react-router"
import { useMemo } from "react"

import type { OrganizationContext } from "@/client"
import {
  canContributeToWorkspace,
  canManage,
  canManageWorkspace,
} from "@/features/organization/roles"
import {
  type DeploymentOperatorAnswer,
  deploymentOperatorAnswer,
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"
import { useEntitlements } from "@/shared/hooks/useEntitlements"

import { navContextForPath, navItemForPath } from "./registry"
import type { NavItem, Requirement } from "./types"

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
  // A row's `to` is a declared path already, so it needs no router to resolve.
  const { context, selectedWorkspaceRole } = useCaller()
  // The caller axis, and it rides on the organization context rather than on a
  // query of its own: `GET /v1/organizations/me` carries `deployment_operator`,
  // the same answer `/admin/access` gives from the same server-side
  // predicate, and the shell reads that context anyway to decide whether to
  // offer the way into the organization rail. So the answer is here when the
  // chrome is, and no row is drawn on the strength of a question still in flight
  // (#836). Read for every row rather than only the ones that declare the axis,
  // because a hook cannot be called from inside the predicate; it is one cached
  // read the rest of them ignore.
  //
  // It also needs no deployment gate of its own any more. The request this used
  // to make had to be withheld from a gateway that does not host `/admin`, so
  // its 404 could not become a second reading of `surfaces`; this read is not
  // that request, and each operator-only row still declares the surface it
  // needs, which `isRouteVisible` composes below.
  //
  // Through `useDeploymentOperator`, the one answer the pages read too, so a row
  // and the page behind it cannot disagree about a failed read (otari#876).
  const { answer, isOperator } = useDeploymentOperator()

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
        ? answer === "operator"
        : isOperator
    }

    // While the read is in flight, draw only the rows a caller with no
    // organization keeps. Right after an invitee's first sign-in that 404 is
    // the answer coming, and drawing the member rail first flashes every row
    // they are about to lose; a member instead sees the rest arrive late.
    if (answer === "no-organization" || answer === "pending") {
      return (item: NavItem) =>
        item.reachableWithoutOrganization === true && isRouteVisible(item)
    }
    const caller = { context, answer, selectedWorkspaceRole }
    return (item: NavItem) =>
      isRouteVisible(item) &&
      allowedByCaller(item) &&
      routeAccess({ route: item.to, params: {} }, caller) === "open"
  }, [isRouteVisible, answer, isOperator, context, selectedWorkspaceRole])
}

/**
 * The two deployment-shaped axes, without the caller one.
 *
 * What decides whether the shell answers a *route* with its "this deployment
 * does not serve that page" panel, where `useNavVisibility` decides whether the
 * *rail row* shows. The caller axis stays out of it: who may open a page is
 * `useRouteAccess`, which waits for its answer and sends someone it refuses
 * Home, while the panel is a claim about the deployment and would be false
 * said to a caller.
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

export type { Requirement } from "./types"

/** Whether the caller may open a page, or that it is not known yet. */
export type Access = "open" | "blocked" | "pending"

/** What is known about the caller, from the queries the shell holds. */
export interface Caller {
  /** The organization context, once read. */
  context: OrganizationContext | undefined
  /** How the context read went, as `useDeploymentOperator` reports it. */
  answer: DeploymentOperatorAnswer
  /** The caller's role in the selected workspace: null in none, undefined while unknown. */
  selectedWorkspaceRole: string | null | undefined
}

/**
 * What the router serves at an address: the route by the path it declares, and
 * its params as the address wrote them. Judged by the router's own matching, so
 * whatever the router forgives (the case of a letter) the rules forgive too:
 * `/Organization/Members` is the members page, and is gated as one.
 */
export interface Target {
  /** The declared path without a trailing slash, or undefined for the catch-all. */
  route: string | undefined
  params: Readonly<Record<string, string>>
  /** What the route itself declares (`staticData.access`), if anything. */
  access?: Requirement | "anyone"
}

export function targetOf(router: AnyRouter, pathname: string): Target {
  const [, params, found] = router.getMatchedRoutes(pathname)
  const declared = found?.fullPath as string | undefined
  return {
    route:
      declared === undefined || declared === "/$"
        ? undefined
        : declared.replace(/(.)\/$/, "$1"),
    params,
    access: found?.options.staticData?.access,
  }
}

/**
 * What a route asks of the caller, read off the registry entry that governs
 * it: its `operatorOnly`, its `requires`, or the organization rail it sits in,
 * which is a manager's. One route asks less than its entry: a workspace's own
 * page opens to that workspace's owner or admin, who may not open the list.
 */
export function requirementFor(route: string): Requirement | undefined {
  if (route === "/workspaces/$workspaceId") return "workspace-manager"
  const item = navItemForPath(route)
  if (!item) return undefined
  if (item.operatorOnly) return "operator"
  if (item.requires) return item.requires
  return navContextForPath(route) === "organization" ? "manager" : undefined
}

/**
 * Whether `caller` may open what `target` serves. `pending` until what the
 * requirement asks about is known, so nobody is sent away from a page they may
 * open while their role is still loading. A failed read opens the page, as the
 * shell's Organization row fails open: the page then shows the server's own
 * answer.
 */
export function routeAccess(target: Target, caller: Caller): Access {
  const requirement =
    target.access ??
    (target.route === undefined ? undefined : requirementFor(target.route))
  const { answer, context } = caller
  if (requirement === undefined || requirement === "anyone") {
    // Someone in no organization has only the rows that opt in for them, so a
    // page the rail hides from them does not open by its address either.
    const item =
      target.route === undefined ? undefined : navItemForPath(target.route)
    return answer === "no-organization" &&
      requirement === undefined &&
      item !== undefined &&
      !item.reachableWithoutOrganization
      ? "blocked"
      : "open"
  }
  const verdict = (allowed: boolean): Access => (allowed ? "open" : "blocked")
  if (answer === "pending") return "pending"
  if (answer === "unavailable") return "open"
  if (answer === "no-organization") return "blocked"
  switch (requirement) {
    case "operator":
      return verdict(answer === "operator")
    case "manager":
      return verdict(canManage(context))
    case "workspace-manager": {
      // From the address as written: an id keeps its case.
      const role = context?.workspace_memberships?.find(
        (one) => one.workspace_id === target.params.workspaceId,
      )?.role
      return verdict(canManageWorkspace(context, role))
    }
    case "contributor":
      if (caller.selectedWorkspaceRole === undefined) return "pending"
      return verdict(
        canContributeToWorkspace(
          context,
          caller.selectedWorkspaceRole ?? undefined,
        ),
      )
  }
}

/** The caller as `routeAccess` reads them. */
export function useCaller(): Caller {
  const context = useOrganizationContext()
  const { selected, isSettled } = useSelectedWorkspace()
  return {
    context: context.data,
    answer: deploymentOperatorAnswer(context),
    selectedWorkspaceRole: isSettled ? (selected?.role ?? null) : undefined,
  }
}

/**
 * Whether the caller may open the page at an address. Stable while the
 * caller's answers are, since the rail and its memory are memos and effects
 * keyed on it. Client-side only, like every gate here: it decides what to
 * offer, and the server still authorizes every request a page makes.
 */
export function useRouteAccess(): (pathname: string) => Access {
  const router = useRouter()
  const { context, answer, selectedWorkspaceRole } = useCaller()
  return useMemo(
    () => (pathname: string) =>
      routeAccess(targetOf(router, pathname), {
        context,
        answer,
        selectedWorkspaceRole,
      }),
    [router, context, answer, selectedWorkspaceRole],
  )
}
