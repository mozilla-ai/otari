import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CallerIdentity, OrganizationContext } from "@/client"
import { ProfileCard } from "@/features/account/ProfileCard"
import * as apiClient from "@/shared/api/client"
import { useOrganizationContext } from "@/shared/api/organizations"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

const CALLER: CallerIdentity = {
  user_id: "33333333-3333-3333-3333-333333333333",
  email: "ada@example.com",
  full_name: "Operator",
}

/**
 * Spies on the transport rather than the hooks, so the real query keys, the
 * cache seeding and the invalidation all run. `saved` is what the PATCH answers
 * with and what `/organizations/me` reports from then on, which is how a test
 * can tell a card that re-read the server from one that kept its own draft.
 */
function mockApi(saved?: CallerIdentity) {
  const patched: CallerIdentity[] = []
  let caller = CALLER
  vi.spyOn(apiClient, "apiFetch").mockImplementation(
    async (path: string, init?: RequestInit) => {
      if (path === "/organizations/me") {
        return organizationContext({ caller }) as never
      }
      if (path === "/auth/profile") {
        patched.push(JSON.parse(String(init?.body)) as CallerIdentity)
        caller = saved ?? { ...caller, full_name: null }
        return caller as never
      }
      throw new Error(`unexpected request: ${path}`)
    },
  )
  return patched
}

/** Reads the cached context back, which is where the sidebar takes the name from. */
function ContextProbe() {
  const { data } = useOrganizationContext()
  return (
    <output>
      {(data as OrganizationContext | undefined)?.caller?.full_name ??
        "unnamed"}
    </output>
  )
}

function renderCard() {
  return render(
    <AppProviders>
      <ProfileCard />
      <ContextProbe />
    </AppProviders>,
  )
}

describe("ProfileCard", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("seeds the field with the name the membership context reports", async () => {
    mockApi()
    renderCard()

    expect(await screen.findByLabelText("Display name")).toHaveValue("Operator")
  })

  it("cannot be saved until the name actually changes", async () => {
    mockApi()
    renderCard()
    await screen.findByLabelText("Display name")

    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
  })

  it("sends the new name and reports it", async () => {
    const patched = mockApi({ ...CALLER, full_name: "Ada Lovelace" })
    const user = userEvent.setup()
    renderCard()

    await user.clear(await screen.findByLabelText("Display name"))
    await user.type(screen.getByLabelText("Display name"), "Ada Lovelace")
    await user.click(screen.getByRole("button", { name: "Save name" }))

    await waitFor(() => {
      expect(patched).toEqual([{ full_name: "Ada Lovelace" }])
    })
    expect(
      await screen.findByText("Saved. You are shown as Ada Lovelace."),
    ).toBeInTheDocument()
  })

  it("renames the person the rest of the shell draws, without waiting for a refetch", async () => {
    mockApi({ ...CALLER, full_name: "Ada Lovelace" })
    const user = userEvent.setup()
    renderCard()

    await user.clear(await screen.findByLabelText("Display name"))
    await user.type(screen.getByLabelText("Display name"), "Ada Lovelace")
    await user.click(screen.getByRole("button", { name: "Save name" }))

    expect(await screen.findByText("Ada Lovelace")).toBeInTheDocument()
  })

  it("clears the name when the field is emptied", async () => {
    const patched = mockApi()
    const user = userEvent.setup()
    renderCard()

    await user.clear(await screen.findByLabelText("Display name"))
    await user.click(screen.getByRole("button", { name: "Save name" }))

    await waitFor(() => {
      expect(patched).toEqual([{ full_name: null }])
    })
    expect(
      await screen.findByText(
        "Saved. You are shown by your sign-in address from now on.",
      ),
    ).toBeInTheDocument()
  })

  it("drops the line reporting the last save as soon as the field is edited again", async () => {
    mockApi({ ...CALLER, full_name: "Ada Lovelace" })
    const user = userEvent.setup()
    renderCard()

    await user.clear(await screen.findByLabelText("Display name"))
    await user.type(screen.getByLabelText("Display name"), "Ada Lovelace")
    await user.click(screen.getByRole("button", { name: "Save name" }))
    await screen.findByText("Saved. You are shown as Ada Lovelace.")

    await user.type(screen.getByLabelText("Display name"), " Byron")

    expect(
      screen.queryByText("Saved. You are shown as Ada Lovelace."),
    ).not.toBeInTheDocument()
  })

  it("refuses a name past the length the gateway stores, before sending it", async () => {
    const patched = mockApi()
    const user = userEvent.setup()
    renderCard()

    await user.clear(await screen.findByLabelText("Display name"))
    await user.paste("a".repeat(256))

    expect(
      screen.getByText("A name can be at most 255 characters."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save name" })).toBeDisabled()
    expect(patched).toEqual([])
  })

  it("says why it could not save", async () => {
    vi.spyOn(apiClient, "apiFetch").mockImplementation(async (path: string) => {
      if (path === "/organizations/me") {
        return organizationContext({ caller: CALLER }) as never
      }
      throw new apiClient.ApiError(400, "That name is not allowed.")
    })
    const user = userEvent.setup()
    renderCard()

    await user.type(await screen.findByLabelText("Display name"), "!")
    await user.click(screen.getByRole("button", { name: "Save name" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "That name is not allowed.",
    )
  })
})
