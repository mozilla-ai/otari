/**
 * Something an edition draws under the password field after a refused sign-in.
 * Renders nothing in this build.
 *
 * Rendered inside the password field's grid, refused or not, so a contributed
 * live region is mounted before the refusal it announces. `LOGIN_REFUSAL_HINT_ID`
 * is the `id` of the element it draws, which the password input names in
 * `aria-describedby` while the refusal stands.
 *
 * **Reached by its `@/features/auth/overlayLoginRefusalHint` specifier and
 * never relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import type { LoginRefusalHintProps } from "./loginRefusalHint"

export const LOGIN_REFUSAL_HINT_ID: string | undefined = undefined

export function LoginRefusalHint(_props: LoginRefusalHintProps) {
  return null
}
