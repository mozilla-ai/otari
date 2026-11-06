import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { readAvatarChoice, saveAvatarChoice, useAvatar } from "./avatarChoice"
import { defaultFaceFor, FACE_COUNT } from "./faces"

const ADA = "33333333-3333-3333-3333-333333333333"
const GRACE = "44444444-4444-4444-4444-444444444444"

afterEach(() => {
  window.localStorage.clear()
  vi.restoreAllMocks()
})

describe("useAvatar", () => {
  it("draws the face the user id gives before anything is picked", () => {
    const { result } = renderHook(() => useAvatar(ADA))
    expect(result.current).toBe(defaultFaceFor(ADA))
  })

  it("answers null while the identity is still loading", () => {
    const { result } = renderHook(() => useAvatar(undefined))
    expect(result.current).toBeNull()
  })

  it("follows a pick made after it rendered", () => {
    const { result } = renderHook(() => useAvatar(ADA))
    const other = (defaultFaceFor(ADA) + 1) % FACE_COUNT
    act(() => saveAvatarChoice(ADA, other))
    expect(result.current).toBe(other)
  })

  it("keeps one person's pick off another person's account", () => {
    saveAvatarChoice(ADA, 5)
    expect(readAvatarChoice(GRACE)).toBeNull()
    const { result } = renderHook(() => useAvatar(GRACE))
    expect(result.current).toBe(defaultFaceFor(GRACE))
  })
})

describe("readAvatarChoice", () => {
  it("ignores a stored value that is not a face", () => {
    window.localStorage.setItem(`otari.avatar.v1:${ADA}`, "not-a-face")
    expect(readAvatarChoice(ADA)).toBeNull()
    window.localStorage.setItem(`otari.avatar.v1:${ADA}`, String(FACE_COUNT))
    expect(readAvatarChoice(ADA)).toBeNull()
  })

  it("falls back to the default when storage is refused", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError")
    })
    expect(readAvatarChoice(ADA)).toBeNull()
    const { result } = renderHook(() => useAvatar(ADA))
    expect(result.current).toBe(defaultFaceFor(ADA))
  })
})
