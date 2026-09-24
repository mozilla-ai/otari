/**
 * Fields an edition adds above the address on the public sign-in pages.
 *
 * Renders nothing in this build: a deployment that serves its own dashboard
 * asks for nothing before the address. A build whose one dashboard reaches
 * several deployments replaces this module at build time and renders the
 * control that picks one; that version owns the choice and what it does with
 * it, so the pages here hold no state for a topology they do not have.
 *
 * Rendered by the four pages that post an address somewhere: sign-in, signup,
 * password recovery and resending a verification link. The pages that spend a
 * token from a link do not render it, because the link already names where
 * it came from. `page` says which one is asking and `isBusy` whether a request
 * of that page's own is in flight, so a contributed control can hold still
 * while an answer is pending.
 *
 * **Reached by its `@/features/auth/overlayPublicAuthFields` specifier and
 * never relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

export type PublicAuthFieldsPage =
  | "login"
  | "signup"
  | "recover-password"
  | "resend-verification"

export interface PublicAuthFieldsProps {
  page: PublicAuthFieldsPage
  isBusy: boolean
}

export function PublicAuthFields(_props: PublicAuthFieldsProps) {
  return null
}
