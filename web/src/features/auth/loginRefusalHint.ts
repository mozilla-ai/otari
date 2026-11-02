/**
 * What `overlayLoginRefusalHint` is handed, off the seam so a build that
 * replaces it can import the props; see `shared/api/requestPolicy.ts` for why.
 */

export interface LoginRefusalHintProps {
  /**
   * The address typed in the form, trimmed. A contributed control that moves
   * the visitor somewhere that reloads the page can hand it on with
   * `rememberSignInAddress`, so it is not typed twice.
   */
  email: string
  /**
   * Whether the gateway just refused the password with its 401. The seam is
   * rendered whether or not this is true, so a contributed control can keep a
   * live region mounted and have it announce the refusal when it arrives.
   */
  refused: boolean
}
