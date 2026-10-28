import { useLocation } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { mockApi, renderPage } from "@/tests/activity"
import { organizationMember } from "@/tests/fixtures"
import { useObserveScope } from "./useObserveScope"

afterEach(() => {
  vi.restoreAllMocks()
})

// The context mock names the caller's membership "om-1".
const ME = organizationMember({
  organization_member_id: "om-1",
  attribution_user_id: "me-id",
})
const NO_IDENTITY = organizationMember({
  organization_member_id: "om-1",
  attribution_user_id: null,
})

function Probe({ active = true }: { active?: boolean }) {
  const state = useObserveScope(active)
  const { searchStr } = useLocation()
  return (
    <>
      <output aria-label="State">
        {JSON.stringify({
          scope: state.scope,
          resolved: state.isResolved,
          users: state.userIds,
          hasOwner: state.hasOwner,
          carry: state.scopeSearch,
        })}
      </output>
      <output aria-label="Search">{searchStr}</output>
      <button type="button" onClick={() => state.setScope("mine")}>
        pick mine
      </button>
      <button type="button" onClick={() => state.setScope("everyone")}>
        pick everyone
      </button>
    </>
  )
}

const state = () => JSON.parse(screen.getByLabelText("State").textContent ?? "")

async function settled(route = "/") {
  renderPage(<Probe />, route)
  await waitFor(() => expect(state().resolved).toBe(true))
  // An operator resolves before the roster has been read.
  await waitFor(() => expect(state().hasOwner).toBe(true))
}

describe("useObserveScope", () => {
  it("defaults an operator to Everyone and the URL's users", async () => {
    mockApi({ members: [ME] })
    await settled("/?user_id=u1")

    expect(state()).toMatchObject({
      scope: "everyone",
      users: ["u1"],
      hasOwner: true,
    })
  })

  it("defaults an organization admin to Everyone", async () => {
    mockApi({ deploymentOperator: false, role: "admin", members: [ME] })
    await settled()

    expect(state().scope).toBe("everyone")
  })

  it("defaults a member to Mine, fetching with their owner id", async () => {
    mockApi({ deploymentOperator: false, role: "member", members: [ME] })
    await settled("/?user_id=u1")

    expect(state()).toMatchObject({
      scope: "mine",
      users: ["me-id"],
      carry: { scope: "mine" },
    })
  })

  it("holds back while the role and the owner id are pending", async () => {
    const { mock } = mockApi({
      deploymentOperator: false,
      role: "viewer",
      members: [ME],
    })
    const answer = mock.getMockImplementation()
    let release = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    mock.mockImplementation(async (input, init) => {
      await gate
      return answer?.(input, init) as Promise<Response>
    })
    renderPage(<Probe />, "/")

    // Nothing has answered: no scope is trusted, so nothing may fetch.
    expect((await screen.findByLabelText("State")).textContent).toContain(
      '"resolved":false',
    )
    release()
    await waitFor(() => expect(state().resolved).toBe(true))
    expect(state().scope).toBe("mine")
  })

  it("falls back to Everyone for a member with no usage identity", async () => {
    mockApi({
      deploymentOperator: false,
      role: "member",
      members: [NO_IDENTITY],
    })
    renderPage(<Probe />, "/?scope=mine")
    await waitFor(() => expect(state().resolved).toBe(true))
    await waitFor(() => expect(state().scope).toBe("everyone"))

    expect(state()).toMatchObject({ scope: "everyone", hasOwner: false })
  })

  it("settles on Everyone when the roster cannot be read", async () => {
    const { mock } = mockApi({
      deploymentOperator: false,
      role: "member",
      members: [ME],
    })
    const answer = mock.getMockImplementation()
    mock.mockImplementation(async (input, init) =>
      String(input).includes("/organizations/me/members")
        ? new Response("{}", { status: 500 })
        : (answer?.(input, init) as Promise<Response>),
    )
    renderPage(<Probe />, "/")

    // The owner id cannot be known, so the page may fetch, as Everyone.
    await waitFor(() => expect(state().resolved).toBe(true))
    expect(state()).toMatchObject({ scope: "everyone", hasOwner: false })
  })

  it("lets a member choose Everyone", async () => {
    mockApi({ deploymentOperator: false, role: "member", members: [ME] })
    await settled("/?scope=everyone")
    expect(state()).toMatchObject({ scope: "everyone", resolved: true })
  })

  it("lets an admin choose Mine from the URL", async () => {
    mockApi({ deploymentOperator: false, role: "admin", members: [ME] })
    await settled("/?scope=mine")
    expect(state()).toMatchObject({ scope: "mine", users: ["me-id"] })
  })

  it("clears the user filter when switching to Mine", async () => {
    mockApi({ members: [ME] })
    await settled("/?user_id=u1&scope=everyone")
    const user = userEvent.setup()

    await user.click(screen.getByRole("button", { name: "pick mine" }))
    await waitFor(() =>
      expect(screen.getByLabelText("Search")).toHaveTextContent("scope=mine"),
    )
    expect(screen.getByLabelText("Search")).not.toHaveTextContent("user_id")
  })

  it("reads the URL's users and fetches nothing extra when inactive", async () => {
    const { calls } = mockApi({ members: [ME] })
    renderPage(<Probe active={false} />, "/organization/usage?user_id=u1")
    await waitFor(() => expect(state().users).toEqual(["u1"]))

    expect(state()).toMatchObject({ scope: "everyone", resolved: true })
    expect(calls.some((call) => call.url.includes("/members"))).toBe(false)
  })
})
