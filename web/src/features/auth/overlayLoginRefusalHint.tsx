/**
 * Something an edition draws under the password field when a password sign-in
 * was refused, for a refusal this build cannot say more about.
 *
 * Renders nothing in this build: a deployment that serves its own dashboard
 * has one set of accounts, so "incorrect email or password" is the whole
 * answer. A build whose one dashboard reaches several deployments replaces
 * this module at build time and says which deployment the attempt went to,
 * and how to try another; that version owns what it says and where it goes,
 * so the form here holds nothing for a topology it does not have.
 *
 * Rendered inside the password field's grid, on the password form only. `refused`
 * is true after a password sign-in the gateway answered with its 401, the one
 * refusal that cannot say whether the address or the password was wrong; never
 * for an unverified address (which has its own remedy), a network failure, or a
 * passkey or provider sign-in. It goes false once the visitor edits either box,
 * as the refusal itself does. The props are
 * `LoginRefusalHintProps` in `./loginRefusalHint`, kept off this module so the
 * replacing one can import them.
 *
 * **Reached by its `@/features/auth/overlayLoginRefusalHint` specifier and
 * never relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import type { LoginRefusalHintProps } from "./loginRefusalHint"

/**
 * The `id` of the element this draws, which the password input names in
 * `aria-describedby` while the refusal stands. `undefined` here: nothing is
 * drawn, so there is nothing to describe the input with.
 */
export const LOGIN_REFUSAL_HINT_ID: string | undefined = undefined

export function LoginRefusalHint(_props: LoginRefusalHintProps) {
  return null
}
