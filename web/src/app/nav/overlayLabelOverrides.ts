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
 * The module name, this export's name, and the type and helper behind it are
 * `otari-ai/frontend/src/app/nav/overlayLabelOverrides.ts`'s, so the two
 * registries compose when the control-plane UI converges at M5. The entries
 * themselves are not interchangeable: two of `NavLabelOverride`'s field names
 * differ from the platform's, because the registries' shapes differ, and its
 * docstring in `types.ts` says which and why. Read that before porting a list
 * across.
 */
export const OVERLAY_NAV_LABEL_OVERRIDES: readonly NavLabelOverride[] = []
