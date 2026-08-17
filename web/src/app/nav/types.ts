/**
 * The shapes the sidebar registry is written in.
 *
 * Deliberately the same vocabulary as `otari-ai/frontend/src/app/nav/types.ts`,
 * because that tree's pages move into this one at M5 and the two registries have
 * to compose rather than be reconciled. Three fields, three independent gates:
 * `surface` is the deployment axis, `capability` the entitlement axis, and
 * `flag` the operational axis. See ARCHITECTURE.md.
 */

import type { LinkProps } from "@tanstack/react-router"
import type { ReactNode } from "react"

/**
 * A route the sidebar can link to.
 *
 * `LinkProps["to"]` is resolved against the generated route tree, so a path that
 * is not a real route fails to type-check at the registry rather than 404ing at
 * runtime. Stripped of the `undefined` a `Link` allows (it means "stay here"),
 * because an entry with no destination is not a destination.
 */
export type NavPath = NonNullable<LinkProps["to"]>

/** Fields every sidebar link carries. */
interface NavItemBase {
  to: NavPath
  label: string
  /**
   * A decorative glyph. `otari-ai/frontend` passes a `react-icons` `IconType`
   * here; this dashboard draws its own inline SVGs and has no react-icons
   * dependency, so this is the one field of the two registries that differs.
   */
  icon: ReactNode
  /**
   * The management surface this destination needs, from the deployment
   * bootstrap (`GET /v1/bootstrap`). The topology axis: does the process
   * serving this page host the surface at all? A missing one is ungated, which
   * is right for the Overview index: it is the deployment's own front page and
   * reads whatever it is allowed to.
   */
  surface?: string
}

/**
 * Entitlement and feature-flag gating for a sidebar link.
 *
 * A flag is a rollout switch beneath a capability, so it is only valid
 * alongside one: an item is either ungated on these two axes, or gated on a
 * `capability` and optionally narrowed further by a `flag`, composed as AND. A
 * flag without a capability is not representable. Copied from otari.ai's
 * registry, including this constraint.
 */
type NavItemGating =
  | { capability?: undefined; flag?: undefined }
  | { capability: string; flag?: string }

/** One sidebar link with its deployment, entitlement, and feature-flag gating. */
export type NavItem = NavItemBase & NavItemGating

/**
 * A group of links under a shared heading.
 *
 * A section with no `label` renders as a divider instead of a heading, which is
 * what sets the index and the system group off from the labeled ones. A section
 * whose items are all gated away renders nothing at all, heading included.
 */
export interface NavSection {
  id: string
  label?: string
  items: readonly NavItem[]
}
