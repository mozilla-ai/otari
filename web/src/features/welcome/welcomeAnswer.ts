import { useSyncExternalStore } from "react"

/**
 * How the operator answered the welcome's "where do models come from", kept in
 * this browser. A timestamp for Close or a completed Connect, `OWN_KEYS` for
 * Skip. Any other value (the e2e suite writes `"e2e"`) is an answer that chose
 * neither.
 */
export const WELCOME_ANSWERED_KEY = "otari.welcome.answered"

export const OWN_KEYS = "own-keys"

const listeners = new Set<() => void>()

/** The answer where storage refused it, so it holds for this tab only. */
let unsaved: string | null = null

export function readWelcomeAnswer(): string | null {
  try {
    return localStorage.getItem(WELCOME_ANSWERED_KEY) ?? unsaved
  } catch {
    return unsaved
  }
}

export function writeWelcomeAnswer(answer: string) {
  try {
    localStorage.setItem(WELCOME_ANSWERED_KEY, answer)
  } catch {
    unsaved = answer
  }
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  window.addEventListener("storage", listener)
  return () => {
    listeners.delete(listener)
    window.removeEventListener("storage", listener)
  }
}

/** The answer, live: a page already open follows a Skip pressed over it. */
export function useWelcomeAnswer(): string | null {
  return useSyncExternalStore(subscribe, readWelcomeAnswer)
}
