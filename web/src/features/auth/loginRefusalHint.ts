/**
 * What `overlayLoginRefusalHint` is handed, off the seam so a build that
 * replaces it can import the props; see `shared/api/requestPolicy.ts` for why.
 */

export interface LoginRefusalHintProps {
  /** The address typed in the form, trimmed. */
  email: string
  /** Whether the gateway just refused the password with its 401. */
  refused: boolean
}
