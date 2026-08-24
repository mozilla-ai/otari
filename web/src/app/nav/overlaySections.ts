import type { NavSection } from "./types"

/**
 * Nav sections contributed on top of the base workspace sidebar.
 *
 * Empty in this build, and meant to stay empty here. An overlay owns its own
 * pages in its own tree and replaces this module at build time to register
 * them; the registry appends whatever it exports after the base sections. This
 * is the seam that lets an overlay add navigation without editing an Otari
 * source file (ARCHITECTURE.md, "cardinal rules for contributors", rule 6).
 *
 * Same shape and same name as `otari-ai/frontend/src/app/nav/overlaySections.ts`,
 * so the two registries compose when the control-plane UI converges at M5.
 */
export const OVERLAY_NAV_SECTIONS: readonly NavSection[] = []

/**
 * Nav sections contributed on top of the organization rail.
 *
 * The second rail needs its own seam, because the destinations an overlay adds
 * are not all workspace-scoped: some belong to the tenant rather than to one
 * workspace. Without this an overlay would have to edit `registry.ts` to
 * register one, which rule 6 rules out.
 *
 * A whole section is the coarser of the two contribution seams, and suits a
 * destination an overlay owns outright. A destination with base neighbours goes
 * through `overlayNavItems.ts` instead: Billing sits inside "Cost & billing" and
 * Gateways inside "Gateway", so contributing either as a section of its own
 * would put a second heading of that name on the rail (otari#737).
 */
export const OVERLAY_ORG_NAV_SECTIONS: readonly NavSection[] = []
