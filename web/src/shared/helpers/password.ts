/**
 * The client half of the server's password policy
 * (`gateway.services.password_service`), not a second authority: it disables a
 * save the gateway would refuse instead of round-tripping to a 400.
 *
 * Shared because the same policy is enforced at three doors that mint or
 * replace a password: the account page's card (`PUT /v1/auth/password`),
 * signup (`POST /v1/auth/signup`), and a password reset
 * (`POST /v1/auth/password/reset/confirm`). One copy, because a rule stated
 * three times is a rule two of them will stop matching.
 */

// The ceiling is counted in bytes because bcrypt's is, so an accented
// character spends more than one and a character count would let a
// 72-character password through to the refusal this exists to pre-empt.
export const MIN_PASSWORD_LENGTH = 8
export const MAX_PASSWORD_BYTES = 72

// Both counts are the server's, and neither is `String.length`. Python's `len`
// counts code points where JavaScript counts UTF-16 units, so seven emoji are
// 14 to `.length` and 7 to the gateway: the minimum would pass here and be
// refused there. bcrypt's ceiling is bytes, which is a third number again.
export function passwordLength(password: string): number {
  return [...password].length
}

export function passwordByteLength(password: string): number {
  return new TextEncoder().encode(password).length
}

/**
 * Why the new password cannot be saved yet, or null when it can.
 *
 * Returns the empty-field case as null rather than as a complaint: a form
 * nobody has typed in yet is not wrong, it is unfinished, and the disabled
 * submit already says so.
 */
export function newPasswordProblem(
  password: string,
  confirm: string,
): string | null {
  if (password === "") {
    return null
  }
  if (passwordLength(password) < MIN_PASSWORD_LENGTH) {
    return `At least ${MIN_PASSWORD_LENGTH} characters.`
  }
  if (passwordByteLength(password) > MAX_PASSWORD_BYTES) {
    return `At most ${MAX_PASSWORD_BYTES} bytes; accented characters count for more than one.`
  }
  if (confirm !== "" && confirm !== password) {
    return "The two passwords do not match."
  }
  return null
}
