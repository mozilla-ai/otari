/**
 * The shapes the sidebar registry is written in.
 *
 * Deliberately the same vocabulary as `otari-ai/frontend/src/app/nav/types.ts`,
 * because that tree's pages move into this one at M5 and the two registries have
 * to compose rather than be reconciled. Two fields, two independent gates:
 * `surface` is the deployment axis and `capability` the entitlement axis. See
 * ARCHITECTURE.md.
 */

import type { LinkProps } from "@tanstack/react-router"
import type { IconType } from "react-icons"

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
   * A decorative glyph: a `react-icons` component, the same Feather set and the
   * same `IconType` reference `otari-ai/frontend/src/app/nav/registry.ts`
   * declares, so a destination that exists in both rails wears the same mark.
   * The shell renders it; the registry only names it.
   */
  icon: IconType
  /**
   * The management surface this destination needs, from the deployment
   * bootstrap (`GET /v1/bootstrap`). The topology axis: does the process
   * serving this page host the surface at all? A missing one is ungated, which
   * is right for the Overview index: it is the deployment's own front page and
   * reads whatever it is allowed to.
   */
  surface?: string
  /**
   * The capability this deployment must be entitled to. The licensing
   * axis, resolved through `shared/hooks/useEntitlements`. A missing one is
   * ungated, which is the ordinary case in this build.
   */
  capability?: string
}

/** One sidebar link with its deployment and entitlement gating. */
export type NavItem = NavItemBase & {
  /**
   * Destinations nested under this one, rendered as a collapsible group.
   *
   * A child declares no gating of its own and inherits the parent's: the
   * group exists because the pages belong together, and a deployment that
   * hosts the surface hosts all of them.
   */
  children?: readonly NavChild[]
}

/** A destination nested under another, gated with its parent by default. */
export interface NavChild {
  to: NavPath
  label: string
  /**
   * The child's own glyph, as `otari-ai/frontend`'s nested leaves carry one.
   * The rail does not draw it on an expanded row, where the indent marks the
   * nesting, but the collapsed flyout does: there the rows hang off an icon in
   * a menu rather than under a parent, and have no indent to read.
   */
  icon: IconType
  /**
   * The surface this destination needs, when it is not the parent's.
   *
   * Grouping is an editorial choice and gating is a fact about the deployment,
   * so the two can disagree: Guardrails is grouped under Routing, where the
   * navigation prototype puts it, but the page is served by the tools surface.
   * Omitted, the child inherits the parent's, which is the ordinary case.
   */
  surface?: string
}

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

/**
 * A rename of one base section's labels: its heading, and the labels of the
 * disclosures inside it.
 *
 * How an overlay relabels an entry the base declares, rather than contributing
 * one of its own. Labels only: a different destination, icon, or gate is a
 * section of the overlay's own through `overlaySections.ts`.
 *
 * Same name as `otari-ai/frontend/src/app/nav/types.ts`'s `NavLabelOverride`,
 * with its two fields pointing at this registry's label sites. Both differ in
 * name for the reason the shapes differ: a section's heading is `label` here and
 * `header` there, and nesting is `children` here where there it is a single
 * `NavDisclosure`, so a section here can hold several disclosures (Gateway holds
 * Routing and Tools) and a rename has to say which.
 */
export interface NavLabelOverride {
  /** Id of the base `NavSection` this override targets. */
  sectionId: string
  /** Replaces the section's heading when set. */
  label?: string
  /**
   * Replaces the label of the nested group at this path, when set.
   *
   * Keyed by the group's own `to`, the same field the registry declares it
   * under, so an override that names a path no longer nested under this section
   * is dropped rather than applied to the wrong row.
   */
  disclosureLabels?: Readonly<Partial<Record<NavPath, string>>>
}
