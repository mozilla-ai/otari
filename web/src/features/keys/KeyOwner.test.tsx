import { Outlet } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { KeyOwner } from "@/features/keys/KeyOwner"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext, organizationMember } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const WEI = organizationMember({
  attribution_user_id: "wei",
  full_name: "Wei Zhang",
})

async function renderOwner(roster: "refused" | "read", userId = "wei") {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === `${API_ROOT}/organizations/me`) {
      return Response.json(organizationContext({ role: "owner" }))
    }
    if (url.startsWith(`${API_ROOT}/organizations/me/members`)) {
      return roster === "refused"
        ? Response.json({ detail: "Forbidden" }, { status: 403 })
        : Response.json({ data: [WEI], count: 1 })
    }
    return Response.json({ data: [], count: 0 })
  })
  await renderWithRouter(<KeyOwner userId={userId} />, {
    shell: (
      <Provider>
        <Outlet />
      </Provider>
    ),
  })
}

describe("a key's owner", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("is named, in full on hover, once the roster is read", async () => {
    await renderOwner("read")
    expect(await screen.findByText("Wei Zhang")).toHaveAttribute(
      "title",
      "Wei Zhang",
    )
  })

  it("is the id the key bills to when the roster does not hold them", async () => {
    await renderOwner("read", "ci-bot")
    expect(await screen.findByText("ci-bot")).toHaveAttribute(
      "title",
      "Not in the organization",
    )
  })

  it("is unknown, not outside the organization, when the roster could not be read", async () => {
    await renderOwner("refused")
    expect(await screen.findByText("—")).toBeInTheDocument()
    expect(screen.queryByTitle("Not in the organization")).toBeNull()
  })
})
