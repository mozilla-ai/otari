/**
 * Something an edition draws in the header of the public sign-in pages, beside
 * the theme control.
 *
 * Renders nothing in this build: a deployment that serves its own dashboard has
 * nothing to say about itself on the way in. A build whose one dashboard
 * reaches several deployments replaces this module at build time and renders
 * the control that says which one (and lets the visitor change it); that
 * version owns the choice and what it does with it, so the shell here holds no
 * state for a topology it does not have.
 *
 * `page` says which page the shell is framing, `null` for one a control like
 * that does not belong on, and `isBusy` whether a request of that page's own is
 * in flight, so a contributed control can hold still while an answer is
 * pending. The props are `AuthShellSlotProps` in `./authShellSlot`, kept off
 * this module so the replacing one can import them.
 *
 * **Reached by its `@/features/auth/overlayAuthShellSlot` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import type { AuthShellSlotProps } from "./authShellSlot"

export function AuthShellSlot(_props: AuthShellSlotProps) {
  return null
}
