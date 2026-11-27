/**
 * The reason the sign-in screen gives in place of its default description, or
 * null for the default. Here, the one a page in front of the session
 * remembered (`signInReason.ts`); an edition replaces it to resolve reasons of
 * its own, such as an intent it stored before sending the visitor to sign in,
 * and may fall back to `readSignInReason`.
 *
 * Read once per mount, so the screen does not change its description while
 * somebody is reading it.
 *
 * **Reached by its `@/features/auth/overlaySignInReason` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import { useState } from "react"

import { readSignInReason } from "./signInReason"

export function useSignInReason(): string | null {
  const [reason] = useState(readSignInReason)
  return reason
}
