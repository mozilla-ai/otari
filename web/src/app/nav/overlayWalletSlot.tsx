/**
 * The balance the design draws at the right end of the top bar.
 *
 * Renders nothing in this build, and is meant to keep rendering nothing here:
 * this gateway meters spend but holds no wallet, so there is no figure to
 * report. A build that does hold one replaces this module at build time and
 * renders its chip; that version owns everything the chip needs (its query, its
 * add-funds modal, its own entitlement gate), so the base chrome holds no state
 * for a capability it does not have.
 *
 * This is `overlaySections.ts`'s seam one grain finer. That one lets an overlay
 * append a destination to a rail the base owns; this one lets an overlay put
 * something *inside* a piece of chrome the base owns, which is the case a rail
 * cannot cover, because a chip in the top bar is not a destination and can be
 * contributed from nowhere else (ARCHITECTURE.md, "cardinal rules for
 * contributors", rule 6).
 *
 * **`TopBarActions.tsx` must reach this by its `@/app/nav/overlayWalletSlot`
 * specifier, not relatively.** That specifier is the key a superset build's
 * alias matches; a relative `./overlayWalletSlot` resolves to a file path the
 * alias never sees, so the contribution would vanish with no error, in a build
 * whose only symptom is the empty default it wanted anyway. `overlaySeams.test.ts`
 * fails on a seam module reached relatively, so that mistake lands on the
 * contributor rather than on a release.
 *
 * Same name and same shape as `otari-ai/frontend`'s, so the two compose when the
 * control-plane UI converges at M5. One deliberate difference: the platform's
 * base module pairs the slot with a no-op `useRefreshWallet`, because its
 * post-checkout return is handled in `src/routes/_layout.tsx`, a base file.
 * There is no such call site here, and the slot renders on every page of the
 * shell, so the overlay's own component can own that effect. A no-op hook with
 * no base caller would read as wired while reaching nothing.
 */
export function WalletNavSlot() {
  return null
}
