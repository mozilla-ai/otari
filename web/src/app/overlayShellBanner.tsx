import type { ReactNode } from "react"

/**
 * A band across the top of the shell, on every signed-in page.
 *
 * Renders nothing in this build, which has no account state worth interrupting
 * every page for. A build that does (otari.ai's wallet running dry, or an auto
 * top-up that stopped) replaces this module at build time and renders its band
 * while the condition holds and nothing otherwise. The replacement owns the
 * query, the copy and whatever action it offers.
 *
 * **The position is the contract.** `AppShell` mounts it once, beside
 * `PricingWarning`: inside `PostSignInGate` and `EntitlementResolver`, so a
 * replacement may read the session and gate on a capability, and above the row
 * the header leads, in flow. The mobile drawer is positioned against that row
 * rather than the viewport, so a band here moves the drawer down with the
 * header instead of covering the control that closes it. `PostSignInGate`
 * cannot host this: it renders outside that layout.
 *
 * **Render nothing while resolving.** A band that paints and then vanishes on
 * every load is worse than one that arrives a moment late.
 *
 * **Reached by its `@/app/overlayShellBanner` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why. That specifier
 * is the alias key an `OVERLAY_MODULE_OVERRIDES` entry has to match.
 */
export function ShellBanner(): ReactNode {
  return null
}
