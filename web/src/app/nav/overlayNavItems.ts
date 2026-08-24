import type { NavItemContribution } from "./types"

/**
 * Rows contributed into sections the base registry declares.
 *
 * Empty in this build, and meant to stay empty here, like the two seams beside
 * it: an overlay replaces this module at build time and the registry appends
 * whatever it exports into the section each contribution names.
 *
 * The seam `overlaySections.ts` cannot cover, and the reason this third module
 * exists. That one appends whole sections, which suits a destination an overlay
 * owns outright; these are the ones that belong *inside* a section the base owns
 * and among rows the base declares. Billing is the canonical case
 * (ARCHITECTURE.md's capability table makes it the overlay-only capability, and
 * it belongs under "Cost & billing" beside `/budgets` and
 * `/organization/pricing`), with Gateways the same shape under the organization
 * rail's "Gateway" heading. Contributing either as a section of its own would
 * put a second "Cost & billing" on the rail; contributing it here puts the row
 * where the navigation design draws it, and neither costs an edit to a base
 * source file (ARCHITECTURE.md, "cardinal rules for contributors", rule 6).
 *
 * One list for both rails, as `OVERLAY_NAV_LABEL_OVERRIDES` is: `sectionId` is
 * unique across the two (`registry.test.ts` pins that), so an overlay has one
 * module to replace rather than one per rail.
 */
export const OVERLAY_NAV_ITEMS: readonly NavItemContribution[] = []
