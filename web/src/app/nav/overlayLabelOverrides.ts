import type { NavLabelOverride } from "./types"

/**
 * Renames applied to the base sidebar's section headings and disclosure labels.
 *
 * Empty in this build, and meant to stay empty here: every base label renders
 * exactly as `registry.ts` declares it. An overlay replaces this module at build
 * time to relabel a base entry it inherits rather than one it contributes, which
 * is the case `overlaySections.ts` cannot cover: appending a section of its own
 * leaves the base wording alone, and editing the base wording means editing an
 * Otari source file (ARCHITECTURE.md, "cardinal rules for contributors", rule 6).
 *
 * One list for both rails, because a section id is unique across the two.
 *
 * Same shape and same name as
 * `otari-ai/frontend/src/app/nav/overlayLabelOverrides.ts`, so the two
 * registries compose when the control-plane UI converges at M5.
 */
export const OVERLAY_NAV_LABEL_OVERRIDES: readonly NavLabelOverride[] = []
