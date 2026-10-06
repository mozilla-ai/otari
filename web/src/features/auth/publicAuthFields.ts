/**
 * What `overlayPublicAuthFields` is handed, off the seam so a build that
 * replaces it can import the props; see `shared/api/requestPolicy.ts` for why.
 */

export type PublicAuthFieldsPage =
  | "login"
  | "signup"
  | "recover-password"
  | "resend-verification"

export interface PublicAuthFieldsProps {
  /** Which page is asking. */
  page: PublicAuthFieldsPage
  /** Whether a request of the page's own is in flight. */
  isBusy: boolean
}
