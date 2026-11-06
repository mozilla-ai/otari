import { useSyncExternalStore } from "react"

import { defaultFaceFor, FACE_COUNT } from "./faces"

/**
 * The face drawn for a person: the one they picked, else the one their user
 * id gives them (`defaultFaceFor`), so everybody has a face from the start.
 *
 * A pick is kept in this browser only, per user: the gateway has no column for
 * it yet, so it does not follow anyone to another device. Moving it onto the
 * profile is the whole of making it shared; every reader goes through
 * `useAvatar`.
 */
// v2: the generated set. A v1 pick indexed the retired 72-face set.
const PREFIX = "otari.avatar.v2:"
const listeners = new Set<() => void>()

export function readAvatarChoice(userId: string): number | null {
  try {
    const raw = window.localStorage.getItem(PREFIX + userId)
    if (raw === null) return null
    const id = Number(raw)
    return Number.isInteger(id) && id >= 0 && id < FACE_COUNT ? id : null
  } catch {
    return null
  }
}

export function saveAvatarChoice(userId: string, id: number): void {
  try {
    window.localStorage.setItem(PREFIX + userId, String(id))
  } catch {
    // Storage refused (private mode, quota): the pick lasts this page view.
  }
  for (const notify of listeners) notify()
}

function subscribe(notify: () => void): () => void {
  listeners.add(notify)
  const onStorage = (event: StorageEvent) => {
    if (event.key?.startsWith(PREFIX)) notify()
  }
  window.addEventListener("storage", onStorage)
  return () => {
    listeners.delete(notify)
    window.removeEventListener("storage", onStorage)
  }
}

/** The face for `userId`, or `null` while the identity is still loading. */
export function useAvatar(userId: string | undefined): number | null {
  const picked = useSyncExternalStore(
    subscribe,
    () => (userId ? readAvatarChoice(userId) : null),
    () => null,
  )
  if (!userId) return null
  return picked ?? defaultFaceFor(userId)
}
