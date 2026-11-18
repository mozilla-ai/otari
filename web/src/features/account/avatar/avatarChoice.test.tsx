import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { CallerIdentity } from "@/client"
import * as apiClient from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

import {
  avatarFace,
  readBrowserAvatar,
  useAdoptBrowserAvatar,
  useOwnAvatar,
} from "./avatarChoice"
import { defaultFaceFor, FACE_COUNT } from "./faces"

const ADA = "33333333-3333-3333-3333-333333333333"

// The adoption runs once per user per page view, so every test of it signs in
// a user of its own.
let nextUser = 0
function newUserId(): string {
  nextUser += 1
  return `00000000-0000-4000-8000-${String(nextUser).padStart(12, "0")}`
}

function remember(userId: string, face: number) {
  window.localStorage.setItem(`otari.avatar:${userId}`, String(face))
}

/**
 * The gateway as the shell sees it: the membership context, and the avatar
 * write, which the context reports from then on. `stored` undefined is a
 * gateway older than the field, which does not send it at all.
 */
function mockApi(
  userId: string,
  {
    stored,
    failWrite = false,
  }: { stored: number | null | undefined; failWrite?: boolean },
) {
  const written: Array<number | null> = []
  let caller: CallerIdentity = {
    user_id: userId,
    email: "ada@example.com",
    full_name: "Ada",
    has_password: true,
    claims_deployment: false,
    ...(stored === undefined ? {} : { avatar_face: stored }),
  }
  vi.spyOn(apiClient, "apiFetch").mockImplementation(
    async (path: string, init?: RequestInit) => {
      if (path === "/organizations/me") {
        return organizationContext({ caller }) as never
      }
      if (path === "/auth/profile/avatar") {
        const { face } = JSON.parse(String(init?.body)) as {
          face: number | null
        }
        written.push(face)
        if (failWrite) throw new Error("The avatar could not be saved.")
        caller = { ...caller, avatar_face: face }
        return caller as never
      }
      throw new Error(`unexpected request: ${path}`)
    },
  )
  return written
}

function Shell() {
  useAdoptBrowserAvatar()
  const face = useOwnAvatar()
  return <output>{face === null ? "loading" : String(face)}</output>
}

function renderShell() {
  return render(
    <AppProviders>
      <Shell />
    </AppProviders>,
  )
}

async function drawnFace(): Promise<number> {
  await waitFor(() =>
    expect(screen.getByRole("status")).not.toHaveTextContent("loading"),
  )
  return Number(screen.getByRole("status").textContent)
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20))
  })
}

afterEach(() => {
  window.localStorage.clear()
  vi.restoreAllMocks()
})

describe("avatarFace", () => {
  it("draws the pick when there is one", () => {
    expect(avatarFace(ADA, 42)).toBe(42)
    expect(avatarFace(ADA, 0)).toBe(0)
  })

  it("draws the face the user id gives when there is none", () => {
    expect(avatarFace(ADA, null)).toBe(defaultFaceFor(ADA))
    expect(avatarFace(ADA, undefined)).toBe(defaultFaceFor(ADA))
  })
})

describe("readBrowserAvatar", () => {
  it("ignores a stored value that is not a face", () => {
    remember(ADA, FACE_COUNT)
    expect(readBrowserAvatar(ADA)).toBeNull()
    window.localStorage.setItem(`otari.avatar:${ADA}`, "not-a-face")
    expect(readBrowserAvatar(ADA)).toBeNull()
  })

  it("answers null when storage is refused", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError")
    })
    expect(readBrowserAvatar(ADA)).toBeNull()
  })
})

describe("useOwnAvatar", () => {
  it("draws the face the user id gives before anything is picked", async () => {
    const user = newUserId()
    mockApi(user, { stored: null })
    renderShell()
    expect(await drawnFace()).toBe(defaultFaceFor(user))
  })

  it("draws the pick the profile holds", async () => {
    const user = newUserId()
    mockApi(user, { stored: 1234 })
    renderShell()
    expect(await drawnFace()).toBe(1234)
  })
})

describe("useAdoptBrowserAvatar", () => {
  it("carries a pick made in this browser onto the profile, then forgets it", async () => {
    const user = newUserId()
    remember(user, 4242)
    const written = mockApi(user, { stored: null })
    renderShell()

    expect(await drawnFace()).toBe(4242)
    await waitFor(() => expect(written).toEqual([4242]))
    await waitFor(() => expect(readBrowserAvatar(user)).toBeNull())
    expect(await drawnFace()).toBe(4242)
  })

  it("draws the browser's pick while the write is on its way, never the default", async () => {
    const user = newUserId()
    remember(user, 4242)
    mockApi(user, { stored: null })
    renderShell()

    const seen: string[] = []
    const observer = new MutationObserver(() =>
      seen.push(screen.getByRole("status").textContent ?? ""),
    )
    observer.observe(screen.getByRole("status"), {
      childList: true,
      characterData: true,
      subtree: true,
    })
    await waitFor(() => expect(readBrowserAvatar(user)).toBeNull())
    observer.disconnect()
    expect(seen).not.toContain(String(defaultFaceFor(user)))
  })

  it("keeps the profile's pick over an older one in this browser, and drops the older one", async () => {
    const user = newUserId()
    remember(user, 4242)
    const written = mockApi(user, { stored: 7 })
    renderShell()

    expect(await drawnFace()).toBe(7)
    await waitFor(() => expect(readBrowserAvatar(user)).toBeNull())
    expect(written).toEqual([])
  })

  it("keeps the browser's pick for the next load when the write fails, and asks once", async () => {
    const user = newUserId()
    remember(user, 4242)
    const written = mockApi(user, { stored: null, failWrite: true })
    renderShell()

    await waitFor(() => expect(written).toEqual([4242]))
    await settle()
    expect(written).toEqual([4242])
    expect(readBrowserAvatar(user)).toBe(4242)
    expect(await drawnFace()).toBe(4242)
  })

  it("leaves the browser's pick alone on a gateway that does not store avatars", async () => {
    const user = newUserId()
    remember(user, 4242)
    const written = mockApi(user, { stored: undefined })
    renderShell()

    expect(await drawnFace()).toBe(4242)
    await settle()
    expect(written).toEqual([])
    expect(readBrowserAvatar(user)).toBe(4242)
  })

  it("writes nothing when this browser holds no pick", async () => {
    const user = newUserId()
    const written = mockApi(user, { stored: null })
    renderShell()

    expect(await drawnFace()).toBe(defaultFaceFor(user))
    await settle()
    expect(written).toEqual([])
  })
})
