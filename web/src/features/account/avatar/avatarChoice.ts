import { useEffect } from "react"

import { useSessionCaller, useUpdateAvatar } from "@/shared/api/auth"

import { defaultFaceFor, FACE_COUNT } from "./faces"

/**
 * The face drawn for a person: the one they picked, which the gateway stores on
 * their identity and publishes wherever they are listed, else the one their
 * user id gives them (`defaultFaceFor`), so everybody has a face from the start.
 */
export function avatarFace(
  userId: string,
  picked: number | null | undefined,
): number {
  return picked ?? defaultFaceFor(userId)
}

// Where a pick lived before the gateway stored it: this browser, per user.
// Read only to carry such a pick onto the profile once (`useAdoptBrowserAvatar`)
// and to draw it until that lands, then removed.
const LEGACY_PREFIX = "otari.avatar:"

export function readBrowserAvatar(userId: string): number | null {
  try {
    const raw = window.localStorage.getItem(LEGACY_PREFIX + userId)
    if (raw === null) return null
    const id = Number(raw)
    return Number.isInteger(id) && id >= 0 && id < FACE_COUNT ? id : null
  } catch {
    return null
  }
}

function forgetBrowserAvatar(userId: string): void {
  try {
    window.localStorage.removeItem(LEGACY_PREFIX + userId)
  } catch {
    // Storage refused: the profile holds a pick now, and it wins over this one.
  }
}

/**
 * The signed-in person's face, or `null` while the identity is still loading.
 *
 * A pick still waiting in this browser is drawn until it reaches the profile,
 * so nobody sees their face change back to the default for a round trip.
 */
export function useOwnAvatar(): number | null {
  const caller = useSessionCaller()
  if (!caller?.user_id) return null
  return avatarFace(
    caller.user_id,
    caller.avatar_face ?? readBrowserAvatar(caller.user_id),
  )
}

// Users whose pick this page view has already tried to carry. Once each: a
// write that fails puts the old caller back, which would otherwise ask again
// on the next render, and the pick stays in this browser for the next load.
const attempted = new Set<string>()

/**
 * Carry a pick made before the gateway stored avatars onto the profile, once.
 *
 * A browser holding such a pick uploads it the first time it loads for that
 * user and the profile has none, then forgets it. If the profile already has
 * one, it was made since, so it wins and the browser's copy is dropped. A
 * failed write leaves the copy where it is to try again on the next load. An
 * older gateway that does not publish `avatar_face` at all is left alone.
 */
export function useAdoptBrowserAvatar(): void {
  const caller = useSessionCaller()
  const { mutate } = useUpdateAvatar()
  const userId = caller?.user_id
  const stored = caller?.avatar_face

  useEffect(() => {
    if (!userId || stored === undefined || attempted.has(userId)) return
    const pick = readBrowserAvatar(userId)
    if (pick === null) return
    if (stored !== null) {
      forgetBrowserAvatar(userId)
      return
    }
    attempted.add(userId)
    mutate(pick, { onSuccess: () => forgetBrowserAvatar(userId) })
  }, [userId, stored, mutate])
}
