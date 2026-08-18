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
 * are not all workspace-scoped: Billing is ARCHITECTURE.md's canonical
 * overlay-only capability and it belongs to the tenant, not to one workspace,
 * as do the other three the navigation prototype shows and this build leaves
 * out (Gateways, Guardrail ceiling, and an org-scoped provider-credentials
 * view). Without this an overlay would have to edit `registry.ts` to register
 * any of them, which rule 6 rules out.
 */
export const OVERLAY_ORG_NAV_SECTIONS: readonly NavSection[] = []
