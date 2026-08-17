import type { NavSection } from "./types"

/**
 * Nav sections contributed on top of the base sidebar.
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
