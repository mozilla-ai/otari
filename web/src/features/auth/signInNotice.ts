/**
 * One line a page in front of the session hands to the sign-in screen, saying
 * why the visitor landed there instead of where their link pointed.
 *
 * In memory rather than session storage: it describes this navigation only.
 * Read without being removed, because React may run a state initializer twice;
 * the sign-in screen forgets it once it has mounted.
 */
let notice: string | null = null

export function rememberSignInNotice(message: string): void {
  notice = message
}

export function readSignInNotice(): string | null {
  return notice
}

export function forgetSignInNotice(): void {
  notice = null
}
