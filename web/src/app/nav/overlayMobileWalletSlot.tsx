import type { ReactNode } from "react"

/**
 * The balance on a phone, in the mobile header beside the drawer toggle.
 *
 * `overlayWalletSlot` sits in `TopBarActions`, which is `hidden md:flex`, so a
 * build that holds a wallet showed no balance below `md`. This is that seam's
 * phone half: a separate module rather than a prop on the first, because the
 * header has room for a compact chip and nothing more, and the replacing build
 * decides what compact means.
 *
 * Renders nothing in this build, which holds no wallet. The shell hides the
 * slot at `md` and up, so a replacement does not hide itself, and it must stay
 * short: everything in that header truncates rather than wrapping, and the
 * drawer is offset by the header's fixed height.
 *
 * **Reached by its `@/app/nav/overlayMobileWalletSlot` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why. That specifier
 * is the alias key an `OVERLAY_MODULE_OVERRIDES` entry has to match.
 */
export function MobileWalletNavSlot(): ReactNode {
  return null
}
