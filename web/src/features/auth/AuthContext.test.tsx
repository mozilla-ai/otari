import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useAuth } from "@/features/auth/AuthContext"
import { AppProviders } from "@/tests/providers"

function Harness() {
  const { isAuthenticated, isSigningOut, logout } = useAuth()
  return isAuthenticated ? (
    <button type="button" onClick={logout}>
      Sign out
    </button>
  ) : (
    <div>{isSigningOut ? "SIGNED OUT (revocation pending)" : "SIGNED OUT"}</div>
  )
}

function RepeatableLogoutHarness() {
  // Unlike Harness, this exposes a "Trigger logout" button that stays
  // rendered regardless of isAuthenticated, so a test can call logout()
  // more than once - Harness's button disappears the instant the first
  // call flips isAuthenticated false.
  const { isSigningOut, logout } = useAuth()
  return (
    <div>
      <button type="button" onClick={logout}>
        Trigger logout
      </button>
      <div>{isSigningOut ? "SIGNING OUT" : "IDLE"}</div>
    </div>
  )
}

describe("AuthProvider", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("restores the signed-in state on load without asking for the key again", () => {
    // The session credential is an HttpOnly cookie the page cannot read; the
    // persisted marker is what tells a fresh tab/restart it is signed in.
    window.localStorage.setItem("otari.dashboard.hasSession", "1")

    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    )

    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument()
  })

  it("starts signed out when no session marker is present", () => {
    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    )

    expect(screen.getByText("SIGNED OUT")).toBeInTheDocument()
  })

  it("revokes the server-side session and drops the marker on sign-out", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(null, { status: 204 }))
    const user = userEvent.setup()

    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    )

    await user.click(screen.getByRole("button", { name: "Sign out" }))

    expect(screen.getByText("SIGNED OUT")).toBeInTheDocument()
    expect(window.localStorage.getItem("otari.dashboard.hasSession")).toBeNull()
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([url]) => url === "/v1/auth/session",
      )
      expect(call?.[1]?.method).toBe("DELETE")
    })
  })

  it("marks isSigningOut while the revocation is in flight, and clears it once resolved", async () => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
    let resolveDelete!: () => void
    const deletePending = new Promise<Response>((resolve) => {
      resolveDelete = () => resolve(new Response(null, { status: 204 }))
    })
    vi.spyOn(globalThis, "fetch").mockReturnValue(deletePending)
    const user = userEvent.setup()

    render(
      <AppProviders>
        <Harness />
      </AppProviders>,
    )

    await user.click(screen.getByRole("button", { name: "Sign out" }))

    // Local sign-out is synchronous and unconditional...
    expect(
      screen.getByText("SIGNED OUT (revocation pending)"),
    ).toBeInTheDocument()

    // ...but isSigningOut only clears once the DELETE actually resolves.
    resolveDelete()
    await waitFor(() => {
      expect(screen.getByText("SIGNED OUT")).toBeInTheDocument()
    })
  })

  it("keeps isSigningOut true until every concurrent logout's revocation settles", async () => {
    // A manual sign-out and a stray 401-triggered auto-logout can both call
    // logout() close together. If the flag cleared on whichever deleteSession
    // resolves first, an earlier call finishing first would reopen the #557
    // race while a later, still-pending one could still land and clobber a
    // fresh sign-in's cookie.
    let resolveFirst!: () => void
    let resolveSecond!: () => void
    const firstPending = new Promise<Response>((resolve) => {
      resolveFirst = () => resolve(new Response(null, { status: 204 }))
    })
    const secondPending = new Promise<Response>((resolve) => {
      resolveSecond = () => resolve(new Response(null, { status: 204 }))
    })
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockReturnValueOnce(firstPending)
      .mockReturnValueOnce(secondPending)
    const user = userEvent.setup()

    render(
      <AppProviders>
        <RepeatableLogoutHarness />
      </AppProviders>,
    )

    const trigger = screen.getByRole("button", { name: "Trigger logout" })
    await user.click(trigger)
    await user.click(trigger)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(screen.getByText("SIGNING OUT")).toBeInTheDocument()

    // The second (later) call resolves first: the flag must stay true, since
    // the first call is still pending. Flush the resolved promise's
    // microtasks (its .finally()) without waiting on a condition that's
    // already true, so this actually exercises the settle path rather than
    // passing trivially.
    resolveSecond()
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(screen.getByText("SIGNING OUT")).toBeInTheDocument()

    // Only once the first call also resolves does the flag clear.
    resolveFirst()
    await waitFor(() => {
      expect(screen.getByText("IDLE")).toBeInTheDocument()
    })
  })
})
