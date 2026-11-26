/**
 * Why somebody was sent to the sign-in screen, said in place of its default
 * description: "Sign in to join Acme." rather than "One account for every
 * model.". A page in front of the session remembers it just before it sends
 * the visitor to sign in; `overlaySignInReason` is what the screen asks.
 *
 * In memory, like `signInNotice`: it describes this navigation only. Read
 * without being removed, because React may run a state initializer twice; the
 * sign-in screen forgets it once it has mounted.
 */
let reason: string | null = null

export function rememberSignInReason(text: string): void {
  reason = text
}

export function readSignInReason(): string | null {
  return reason
}

export function forgetSignInReason(): void {
  reason = null
}
