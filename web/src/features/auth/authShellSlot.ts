/**
 * What `overlayAuthShellSlot` is handed, off the seam so a build that replaces
 * it can import the props; see `shared/api/requestPolicy.ts` for why.
 */

/**
 * The pages whose address is typed into a form, which is where a control that
 * says which deployment the address belongs to has a job. The pages that spend
 * a token from a link (verify, reset, check-email, an OAuth callback) are not
 * here: the link already names where it came from, and a control that could
 * move the visitor elsewhere would only spend the token in the wrong place.
 */
export type AuthShellPage =
  | "login"
  | "signup"
  | "recover-password"
  | "resend-verification"

export interface AuthShellSlotProps {
  /** Which page the shell is framing, or `null` for any other. */
  page: AuthShellPage | null
  /** Whether a request of the page's own is in flight. */
  isBusy: boolean
}

const SHELL_PAGES: Record<string, AuthShellPage> = {
  "": "login",
  "/": "login",
  "/signup": "signup",
  "/recover-password": "recover-password",
  "/resend-verification": "resend-verification",
}

/** The shell page a `location.hash` is, or `null` when it is not one of them. */
export function authShellPage(hash: string): AuthShellPage | null {
  const path = hash.replace(/^#/, "").split("?")[0] ?? ""
  return SHELL_PAGES[path] ?? null
}
