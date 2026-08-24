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
 * **Reached by its `@/app/nav/overlayWalletSlot` specifier and never
 * relatively**, which is the seam rule and not a style call; `overlaySeams.test.ts`
 * enforces it and web/AGENTS.md says why.
 *
 * Two things a build writing the replacing half needs to know, because both are
 * places this seam can disappoint quietly rather than fail:
 *
 * - **The alias key is `@/app/nav/overlayWalletSlot`.** `otari-ai/frontend` has
 *   no counterpart to this module today: its three live seams are all under
 *   `src/app/nav/` (`overlaySections.ts`, `overlayLabelOverrides.ts`,
 *   `overlayAdminTabs.ts`), which is why this one is there too, but issue #736
 *   describes the reference as `src/app/overlayWalletSlot.tsx`. An
 *   `OVERLAY_MODULE_OVERRIDES` entry keyed on that shorter path matches nothing
 *   and leaves this empty default in place.
 * - **The slot inherits the cluster's `hidden … md:flex`, so a contributed chip
 *   is desktop-only.** So is otari.ai's own balance: `WalletBalance` renders at
 *   `frontend/src/app/Navbar.tsx:118`, inside a `hidden md:flex` cluster. That
 *   makes this known scope rather than an oversight, and it does mean a phone
 *   reaches no add-funds affordance through this seam; giving it one is another
 *   seam, not an edit to the top bar.
 *
 * One export, where #736 describes the reference as pairing the slot with a
 * no-op `useRefreshWallet`. That pairing is not arbitrary and it does not
 * transfer. In otari-ai the hook is real (`frontend/src/features/wallet/hooks/
 * useWallet.ts:81`) and its caller is `frontend/src/routes/_layout.tsx:53`,
 * which reads a `?wallet=success` or `?wallet=cancelled` return from Stripe: a
 * base file, so the platform needs a no-op there for a build without the
 * wallet. Nothing here has that shape. There is no checkout to return from and
 * no equivalent call site, and this slot renders on every page of the shell, so
 * the replacing component can own that effect itself. A no-op hook with no base
 * caller would read as wired while reaching nothing, which is what the comment
 * this module replaced was warning about.
 */
export function WalletNavSlot() {
  return null
}
